import time
import argparse
import ray
import wandb
import numpy as np
import os
import csv
import json
from collections import deque
from typing import Dict, Tuple

import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.utils import flatten_dict

from GraphJSSPEnv import GraphJSSPEnv
from models import GNNMaskedActionsModel
from CustomCallbacks import CustomCallbacks

_exclude_results = ["done", "should_checkpoint", "config"]
_config_results = [
    "trial_id", "experiment_tag", "node_ip",
    "experiment_id", "hostname", "pid", "date",
]

parser = argparse.ArgumentParser(
    description="PPO + GNN trainer for JSSP Taillard instances."
)
parser.add_argument("--instances", type=str, required=True, help="path/to/taXX-taYY")
parser.add_argument("--iters", type=int, default=300, help="Max training iterations.")
parser.add_argument("--out", type=str, required=True, help="Output folder for checkpoints and CSV.")
parser.add_argument("--bks", type=str, required=True, help="Path to bks.json")
args = parser.parse_args()


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")
    for k, v in flat_result.items():
        if any(k.startswith(item + "/") or k == item for item in _config_results):
            config_update[k] = v
        elif any(k.startswith(item + "/") or k == item for item in _exclude_results):
            continue
        elif not wandb_tune._is_allowed_type(v):
            continue
        else:
            log[k] = v
    config_update.pop("callbacks", None)
    return log, config_update


def train_func(instance_path, num_iterations, save_dir, bks_path):
    with open(bks_path, "r") as f:
        bks_map = json.load(f)

    # ---- Register env & model (once each) ----
    register_env("GraphJSS", lambda config: GraphJSSPEnv(config))
    ModelCatalog.register_custom_model("gnn_masked_model", GNNMaskedActionsModel)

    lr_start, lr_end = 3e-4, 1e-5
    entropy_start, entropy_end = 0.01, 1e-3

    default_config = {
        "env": "GraphJSS",
        "seed": 42,
        "framework": "torch",
        "log_level": "WARN",
        "num_gpus": 1,
        "env_config": {
            "instance_path": instance_path,
            "bks_map": bks_map,
        },
        # ---- Workers ----
        "num_workers": 4,
        "num_envs_per_worker": 1,
        # ---- Batch sizes ----
        "train_batch_size": 12000,
        "rollout_fragment_length": 512,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 15,
        # ---- PPO hyperparams (tuned) ----
        "clip_param": 0.2,
        "vf_clip_param": 50.0,       # Finite clip (was 100k — effectively disabled)
        "vf_loss_coeff": 0.5,        # Standard value
        "kl_coeff": 0.0,             # Disable adaptive KL (rely on clip)
        "lambda": 0.95,              # GAE lambda (was 1.0 — no bootstrapping)
        "gamma": 0.999,              # Slight discounting (was 1.0)
        "grad_clip": 0.5,            # Gradient clipping (critical for stability)
        # ---- Schedules ----
        "lr_schedule": [
            [0, lr_start],
            [2_000_000, lr_end],
        ],
        "entropy_coeff_schedule": [
            [0, entropy_start],
            [2_000_000, entropy_end],
        ],
        # ---- Model ----
        "model": {
            "custom_model": "gnn_masked_model",
        },
        "metrics_smoothing_episodes": 50,
    }

    print("Initializing Ray...")
    ray.init(
        ignore_reinit_error=True,
        num_cpus=20,  # 1 driver + 4 workers + 1 spare
        _memory=30 * 1024**3,
        object_store_memory=10 * 1024**3,
    )

    print("Connecting to Weights & Biases...")
    wandb.init(project="JSS_PPO_GNN", config=default_config)

    config = with_common_config(default_config)
    config["callbacks"] = CustomCallbacks

    print("Building PPO Trainer...")
    trainer = PPOTrainer(config=config)
    print("Trainer built. Starting training loop...")

    gap_window = deque(maxlen=10)
    csv_history = []
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, "results.csv")

    best_gap = float("inf")

    for iteration in range(1, num_iterations + 1):
        t0 = time.time()
        result = trainer.train()
        dt = time.time() - t0

        result_clean = wandb_tune._clean_log(result)
        log, _ = _handle_result(result_clean)
        wandb.log(log)

        metrics = result.get("custom_metrics", {})
        makespan = metrics.get("make_span_mean", 0)
        iter_gap = metrics.get("optimality_gap_mean")
        ep_reward = result.get("episode_reward_mean", "N/A")

        if iter_gap is not None:
            gap_window.append(iter_gap)
            rolling_avg = np.mean(gap_window)

            csv_history.append({
                "iteration": iteration,
                "makespan": f"{makespan:.2f}",
                "gap": f"{iter_gap:.2f}",
                "rolling_gap": f"{rolling_avg:.2f}",
                "reward": f"{ep_reward:.4f}" if isinstance(ep_reward, float) else str(ep_reward),
                "time_s": f"{dt:.1f}",
            })

            print(
                f"Iter {iteration:4d} | reward={ep_reward:>10} | "
                f"gap={iter_gap:.2f}% | rolling={rolling_avg:.2f}% | "
                f"time={dt:.1f}s"
            )

            # Save best model
            if iter_gap < best_gap:
                best_gap = iter_gap
                best_path = trainer.save(os.path.join(save_dir, "best"))
                print(f"  ★ New best gap {best_gap:.2f}% — saved to {best_path}")

            if len(gap_window) >= 10 and rolling_avg <= 15.0:
                print(
                    f"Target reached! Stable gap at {rolling_avg:.2f}%. "
                    f"Saving and exiting."
                )
                trainer.save(save_dir)
                break
        else:
            print(f"Iter {iteration}: reward={ep_reward}")

        if iteration % 25 == 0:
            ckpt = trainer.save(save_dir)
            print(f"  Checkpoint at iter {iteration}: {ckpt}")

    # Write CSV
    fieldnames = ["iteration", "makespan", "gap", "rolling_gap", "reward", "time_s"]
    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_history)

    wandb.finish()
    ray.shutdown()
    print(f"Done. Results saved to {csv_file_path}")


if __name__ == "__main__":
    train_func(args.instances, args.iters, args.out, args.bks)
