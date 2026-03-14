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
from GraphJSSPEnv import GraphJSSPEnv
from models import GNNMaskedActionsModel

from CustomCallbacks import *
# Make sure GNNMaskedActionsModel is in your models.py
from models import GNNMaskedActionsModel 
from MultiInstanceJSSEnv import MultiInstanceJSSEnv

from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.utils import flatten_dict

_exclude_results = ["done", "should_checkpoint", "config"]
_config_results = ["trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname", "pid", "date"]

parser = argparse.ArgumentParser(description="Iteration-based PPO trainer with GNN and Rolling Window.")
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

    register_env("MultiJSS", lambda config: GraphJSSPEnv(config))

    lr_start, lr_end = 1.0e-4, 1.0e-5
    entropy_start, entropy_end = 2.0e-3, 2.5e-4

    default_config = {
        'env': 'MultiJSS',
        'seed': 0,
        # 1. CHANGED FRAMEWORK TO PYTORCH
        'framework': 'torch', 
        'log_level': 'INFO',
        'num_gpus': 1,
        'env_config': {
            'instance_path': instance_path,
            'bks_map': bks_map
        },
        'num_workers': 4,
        'train_batch_size': 8000,
        'num_envs_per_worker': 1,
        'rollout_fragment_length': 704,
        'sgd_minibatch_size': 256,
        'num_sgd_iter': 10,
        'vf_clip_param': 100000.0,
        
        'clip_param': 0.2,
        'vf_loss_coeff': 0.8,
        'kl_coeff': 0.2,
        'lambda': 1.0,
        'gamma': 1.0,

        'lr_schedule': [
            [0, lr_start],
            [1000000, lr_end],
        ],

        'entropy_coeff_schedule': [
            [0, entropy_start],
            [1000000, entropy_end],
        ],

        'model': {
            "custom_model": "gnn_masked_model", 
        },
        
        'metrics_smoothing_episodes': 100, 
    }
    print("Initializing Ray with strict memory limits...")
    ray.init(
        ignore_reinit_error=True,
        num_cpus=25, # Restrict Ray from seeing all 35 CPUs
        # Cap Ray's total memory usage to ~40 GB (leaving room for Ubuntu OS)
        _memory=40 * 1024 * 1024 * 1024, 
        # Cap the Object Store (where Ray shares data between workers) to 10 GB
        object_store_memory=10 * 1024 * 1024 * 1024 
    )
    ModelCatalog.register_custom_model("gnn_masked_model", GNNMaskedActionsModel)    
    # 3. REGISTERED THE NEW GNN MODEL
    ModelCatalog.register_custom_model("gnn_masked_model", GNNMaskedActionsModel)
    print("Connecting to Weights & Biases...")
    wandb.init(project="JSS_PPO_GNN", config=default_config)
    config = wandb.config
    
    config = with_common_config(default_config)
    config['callbacks'] = CustomCallbacks 
    print("Building PPO Trainer (This may take a minute as workers spawn)...")
    trainer = PPOTrainer(config=config)
    print("Trainer built successfully! Starting training loop...")
    
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

            if len(gap_window) >= 5 and rolling_avg <= 20.0:
                print(f"Goal Reached! Stable gap at {rolling_avg:.2f}%. Saving and exiting.")
                trainer.save(save_dir)
                break
        else:
            print(f"Iteration {iteration}: Mean Reward={result.get('episode_reward_mean')}")

        if iteration % 25 == 0:
            ckpt_path = trainer.save(save_dir)
            print(f"Checkpoint saved at iteration {iteration}: {ckpt_path}")

    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "makespan", "bks", "optimality gap"])
        writer.writeheader()
        writer.writerows(csv_history)
    
    ray.shutdown()

if __name__ == "__main__":
    train_func(args.instances, args.iters, args.out, args.bks)
