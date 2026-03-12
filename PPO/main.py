import os
import csv
import time
import argparse
import random
import multiprocessing as mp
from typing import Dict, Tuple, Optional, List

import numpy as np
import ray
import wandb

import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf

from CustomCallbacks import *
from models import *

tf1, tf, tfv = try_import_tf()

_exclude_results = ["done", "should_checkpoint", "config"]

# Use these result keys to update wandb.config
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]


parser = argparse.ArgumentParser(description="Iteration-based PPO trainer for Taillard instances.")
parser.add_argument(
    "--instances",
    type=str,
    default="instances/ta52",
    help="Path to the Taillard instance directory or file."
)
parser.add_argument(
    "--iters",
    type=int,
    default=200,
    help="Maximum number of PPO training iterations."
)
parser.add_argument(
    "--out",
    type=str,
    default="checkpoint_results",
    help="Path to save checkpoints and the summary CSV."
)
parser.add_argument(
    "--bks",
    type=float,
    required=True,
    help="Best Known Solution (BKS) value used to compute the optimality gap."
)
args = parser.parse_args()


def _handle_result(result: Dict) -> Tuple[Dict, Dict]:
    """Prepare a Ray RLlib result dict for wandb logging."""
    config_update = result.get("config", {}).copy()
    log: Dict = {}
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

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_objective_from_result(result: Dict) -> Optional[float]:
    """
    Try to extract a *minimization* objective (e.g., makespan) from RLlib results.

    Priority order:
      1) custom_metrics/makespan_min
      2) custom_metrics/makespan_mean
      3) any custom metric containing 'makespan' with a numeric value
      4) episode_reward_mean (assumes reward = -makespan if reward is negative)
    """
    cm = result.get("custom_metrics", {}) or {}

    # 1) / 2) common custom metrics patterns
    for key in ("makespan_min", "makespan_mean"):
        if key in cm:
            v = _safe_float(cm.get(key))
            if v is not None:
                return v

    # 3) any makespan-like custom metric
    for k, v in cm.items():
        if "makespan" in str(k).lower():
            fv = _safe_float(v)
            if fv is not None:
                return fv

    # 4) fallback: infer from reward (common in scheduling: reward = -makespan)
    r = _safe_float(result.get("episode_reward_mean", None))
    if r is None:
        return None
    if r < 0:
        return -r

    # Otherwise cannot safely infer objective direction/value
    return None


def _optimality_gap(obj: float, bks: float) -> float:
    """
    Optimality gap for minimization:
      gap = max(0, (obj - bks) / bks)
    """
    if bks <= 0:
        raise ValueError("BKS must be > 0 to compute an optimality gap.")
    gap = (obj - bks) / bks
    return max(0.0, gap)


def _list_instances(instances_arg: str) -> List[str]:
    """
    If instances_arg is a file -> return [file].
    If it's a directory -> return sorted list of files inside (non-recursive).
    """
    p = os.path.expanduser(instances_arg)
    if os.path.isdir(p):
        files = []
        for name in os.listdir(p):
            fp = os.path.join(p, name)
            if os.path.isfile(fp):
                files.append(fp)
        return sorted(files)
    return [p]


def train_single_instance(
    instance_path: str,
    num_iterations: int,
    save_dir: str,
    bks: float,
    target_gap: float = 0.10
) -> Tuple[str, float, Optional[float], Optional[float], float]:
    """
    Train PPO on a single instance, attempting to reach target_gap before stopping.

    Returns:
      (instance_name, bks, final_makespan_or_None, final_gap_or_None, wall_time_seconds)
    """
    os.makedirs(save_dir, exist_ok=True)
    instance_name = os.path.basename(instance_path)

    default_config = {
        "env": "JSSEnv:jss-v1",
        "seed": 0,
        "framework": "tf",
        "log_level": "WARN",
        "num_gpus": 1,
        "instance_path": instance_path,
        "evaluation_interval": None,
        "metrics_smoothing_episodes": 2000,
        "gamma": 1.0,
        "num_workers": mp.cpu_count(),
        "train_batch_size": 4000,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 704,  # TO TUNE
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,  # epochs
        "clip_param": 0.5,
        "vf_loss_coeff": 0.8,
        "kl_coeff": 0.5,
        "lambda": 1.0,
        "entropy_start": 2.0e-3,
        "entropy_end": 2.5e-4,
        "lr_start": 6.6e-4,
        "lr_end": 7.8e-5,
        "batch_mode": "truncate_episodes",
        "grad_clip": None,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_share_layers": False,
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "_fake_gpus": False,
    }

    # Seed everything deterministically.
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Init wandb per instance so that runs are separated.
    wandb.init(config=default_config, reinit=True)
    ray.init(ignore_reinit_error=True)

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config["model"] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    }
    config["env_config"] = {"env_config": {"instance_path": config["instance_path"]}}

    config = with_common_config(config)
    config["callbacks"] = CustomCallbacks

    config["lr"] = config["lr_start"]
    config["lr_schedule"] = [
        [0, config["lr_start"]],
        [1_000_000, config["lr_end"]],
    ]

    config["entropy_coeff"] = config["entropy_start"]
    config["entropy_coeff_schedule"] = [
        [0, config["entropy_start"]],
        [1_000_000, config["entropy_end"]],
    ]

    # Remove non-RLlib keys.
    config.pop("instance_path", None)
    config.pop("lr_start", None)
    config.pop("lr_end", None)
    config.pop("entropy_start", None)
    config.pop("entropy_end", None)

    checkpoint_freq = 25
    trainer = PPOTrainer(config=config)

    print(f"Starting PPO training for up to {num_iterations} iterations on {instance_path} (BKS={bks})...")
    start_time = time.time()

    final_gap: Optional[float] = None
    final_makespan: Optional[float] = None

    # Set up iteration-level progress CSV
    progress_csv_path = os.path.join(save_dir, f"{instance_name}_progress.csv")
    with open(progress_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "episode_reward_mean", "makespan", "optimality_gap", "time_elapsed_sec"])

    try:
        for iteration in range(1, num_iterations + 1):
            result = trainer.train()
            result = wandb_tune._clean_log(result)
            log, _ = _handle_result(result)
            wandb.log(log)

            obj = _extract_objective_from_result(result)
            ep_reward = result.get("episode_reward_mean", "N/A")
            elapsed_time = time.time() - start_time

            if obj is not None:
                gap = _optimality_gap(obj, bks)
                final_gap = gap
                final_makespan = obj
                print(
                    f"Iteration {iteration}/{num_iterations}: "
                    f"makespan={obj:.6g}, gap={gap*100:.2f}%"
                )
                # Log iteration data to the progress CSV
                with open(progress_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, ep_reward, obj, gap, elapsed_time])
            else:
                # Format the strings safely in case a makespan hasn't been found at all yet
                ms_str = f"{final_makespan:.6g}" if final_makespan is not None else "N/A"
                gap_str = f"{final_gap*100:.2f}%" if final_gap is not None else "N/A"
                
                print(
                    f"Iteration {iteration}/{num_iterations}: reward_mean={ep_reward} "
                    f"| current_makespan={ms_str} | current_gap={gap_str} (new makespan not found in logs)"
                )
                
                # Log to progress CSV using the last known values (or empty strings if None)
                with open(progress_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iteration, 
                        ep_reward, 
                        final_makespan if final_makespan is not None else "", 
                        final_gap if final_gap is not None else "", 
                        elapsed_time
                    ])
            if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
                ckpt_path = trainer.save(save_dir)
                print(f"Checkpoint saved at iteration {iteration}: {ckpt_path}")

            # Early stop criterion: attempt to achieve <= target gap.
            if final_gap is not None and final_gap <= target_gap:
                print(
                    f"Early stopping: reached gap {final_gap*100:.2f}% "
                    f"(<= {target_gap*100:.2f}%)."
                )
                break

    finally:
        wall_time = time.time() - start_time
        try:
            trainer.stop()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass

    return instance_name, float(bks), final_makespan, final_gap, wall_time


def save_summary_csv(rows: List[Tuple[str, float, Optional[float], Optional[float], float]], out_dir: str) -> str:
    """
    Save summary rows to CSV under out_dir/summary.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_name", "bks", "makespan", "optimality_gap", "time_taken_sec"])
        for instance_name, bks, makespan, gap, tsec in rows:
            writer.writerow([
                instance_name, 
                bks, 
                "" if makespan is None else makespan, 
                "" if gap is None else gap, 
                tsec
            ])

    return csv_path


if __name__ == "__main__":
    instance_list = _list_instances(args.instances)

    all_rows: List[Tuple[str, float, Optional[float], Optional[float], float]] = []
    for instance_path in instance_list:
        row = train_single_instance(
            instance_path=instance_path,
            num_iterations=args.iters,
            save_dir=args.out,
            bks=args.bks,
            target_gap=0.15,
        )
        all_rows.append(row)

        instance_name, bks, makespan, gap, wall_time = row
        gap_str = "N/A" if gap is None else f"{gap*100:.2f}%"
        makespan_str = "N/A" if makespan is None else f"{makespan:.6g}"
        print(
            f"[SUMMARY] instance={instance_name} | bks={bks} | makespan={makespan_str} | gap={gap_str} | time_taken={wall_time:.2f}s"
        )

    csv_path = save_summary_csv(all_rows, args.out)
    print(f"Saved summary CSV to: {csv_path}")
