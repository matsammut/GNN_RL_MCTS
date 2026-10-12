import time
import argparse
import ray
import wandb
import random
import numpy as np
import os
import csv
import json
from collections import deque
from typing import Dict, Tuple, List
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from CustomCallbacks import *
from models import *
from MultiInstanceJSSEnv import MultiInstanceJSSEnv

import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

_exclude_results = ["done", "should_checkpoint", "config"]
_config_results = ["trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname", "pid", "date"]

parser = argparse.ArgumentParser(description="Iteration-based PPO trainer with Rolling Window.")
parser.add_argument("--instances", type=str, required=True, help="path/to/taXX-taYY")
parser.add_argument("--iters", type=int, default=200, help="Max iterations.")
parser.add_argument("--out", type=str, required=True, help="Folder for checkpoints and CSV.")
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
    with open(bks_path, 'r') as f:
        bks_map = json.load(f)

    register_env("MultiJSS", lambda config: MultiInstanceJSSEnv(config))

    # Re-integrating your full list of parameters
    lr_start, lr_end = 6.6e-4, 7.8e-5
    entropy_start, entropy_end = 2.0e-3, 2.5e-4

    default_config = {
        'env': 'MultiJSS',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 1,
        'env_config': {
            'instance_path': instance_path,
            'bks_map': bks_map
        },
        'num_workers': 32,
        'train_batch_size': 32000,
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,
        'sgd_minibatch_size': 256,
        'num_sgd_iter': 20,
        
        # PPO Specific Parameters from your list
        'clip_param': 0.2,
        'vf_loss_coeff': 0.8,
        'kl_coeff': 0.5,
        'lambda': 1.0,
        'gamma': 1.0,

        # Learning Rate Schedule
        'lr_schedule': [
            [0, lr_start],
            [1000000, lr_end],
        ],

        # Entropy Schedule
        'entropy_coeff_schedule': [
            [0, entropy_start],
            [1000000, entropy_end],
        ],

        'model': {
            "fcnet_activation": "relu",
            "custom_model": "fc_masked_model_tf",
            "fcnet_hiddens": [256, 256],
            "vf_share_layers": False,
        },
        
        # Metrics setup
        'metrics_smoothing_episodes': 100, 
    }
    
    ray.init(ignore_reinit_error=True)
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    wandb.init(project="JSS_PPO", config=default_config)
    config = wandb.config
    
    config = with_common_config(default_config)
    config['callbacks'] = CustomCallbacks 
    trainer = PPOTrainer(config=config)
    
    # Rolling Window Logic
    gap_window = deque(maxlen=5)
    csv_history = []
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, "results.csv")

    for iteration in range(1, num_iterations + 1):
        result = trainer.train()
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
                "bks": "Dynamic",
                "optimality gap": f"{iter_gap:.2f}"
            })

            print(f"Iter {iteration}:  reward_mean={ep_reward} | Gap={iter_gap:.2f}% | Rolling Avg={rolling_avg:.2f}%")

            # Rolling Window Stopping Condition
            if len(gap_window) >= 5 and rolling_avg <= 20.0:
                print(f"Goal Reached! Stable gap at {rolling_avg:.2f}%. Saving and exiting.")
                trainer.save(save_dir)
                break
        else:
            print(f"Iteration {iteration}: Mean Reward={result.get('episode_reward_mean')}")

        if iteration % 25 == 0:
            ckpt_path = trainer.save(save_dir)
            print(f"Checkpoint saved at iteration {iteration}: {ckpt_path}")

    # Save CSV to the --out directory
    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "makespan", "bks", "optimality gap"])
        writer.writeheader()
        writer.writerows(csv_history)
    
    ray.shutdown()

if __name__ == "__main__":
    train_func(args.instances, args.iters, args.out, args.bks)
