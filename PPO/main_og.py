import time
import argparse
import ray
import wandb
import random
import numpy as np
import os
import csv

import ray.tune.integration.wandb as wandb_tune
from ray.rllib.agents.ppo import PPOTrainer

from CustomCallbacks import *
from models import *

from typing import Dict, Tuple
import multiprocessing as mp
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog

from ray.tune.utils import flatten_dict
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

_exclude_results = ["done", "should_checkpoint", "config"]

_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]

parser = argparse.ArgumentParser(description="Iteration-based PPO trainer for Taillard instances.")
parser.add_argument("--instances", type=str, default="instances/ta52",help="Path to the Taillard instance directory or file.")
parser.add_argument("--iters", type=int, default=200,help="Number of PPO training iterations.")
parser.add_argument("--out", type=str, default="checkpoint_results",help="Path to save the checkpoints and results to")
# Added BKS parser
parser.add_argument("--bks", type=float, required=True, help="Best Known Solution to calculate the optimality gap.")
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

def train_func(instance_path: str, num_iterations: int, save_dir: str, bks: float):
    default_config = {
        'env': 'JSSEnv:jss-v1',
        'seed': 0,
        'framework': 'tf',
        'log_level': 'WARN',
        'num_gpus': 1,
        'instance_path': instance_path,
        'evaluation_interval': None,
        'metrics_smoothing_episodes': 2000,
        'gamma': 1.0,
        'num_workers': mp.cpu_count(),
        'train_batch_size':4000,
        'num_envs_per_worker': 4,
        'rollout_fragment_length': 704,  
        'sgd_minibatch_size': 128,
        'num_sgd_iter': 10,          
        'clip_param': 0.5,
        'vf_loss_coeff': 0.8,
        'kl_coeff': 0.5,
        'lambda': 1.0,
        'entropy_start': 2.0e-3,
        'entropy_end': 2.5e-4,
        'lr_start': 6.6e-4,
        'lr_end': 7.8e-5,
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

    wandb.init(config=default_config)
    ray.init()
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = wandb.config

    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    }
    config['env_config'] = {
        'env_config': {'instance_path': config['instance_path']}
    }

    config = with_common_config(config)
    config['callbacks'] = CustomCallbacks

    config['lr'] = config['lr_start']
    config['lr_schedule'] = [
        [0, config['lr_start']],
        [1_000_000, config['lr_end']],
    ]

    config['entropy_coeff'] = config['entropy_start']
    config['entropy_coeff_schedule'] = [
        [0, config['entropy_start']],
        [1_000_000, config['entropy_end']],
    ]
    config.pop('instance_path', None)
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    checkpoint_freq = 25
    trainer = PPOTrainer(config=config)
    print(f"Starting PPO training on {instance_path}... (BKS: {bks})")

    csv_data = []

    for iteration in range(1, num_iterations + 1):
        result = trainer.train()
        print(result.get("custom_metrics", {}).keys())
        result_clean = wandb_tune._clean_log(result)
        log, _ = _handle_result(result_clean)
        wandb.log(log)

        ep_reward = result.get("episode_reward_mean", "N/A")
        
        makespan = result.get("custom_metrics", {}).get("make_span_mean", None)
        print(f"Iteration {iteration}/{num_iterations}: reward_mean={ep_reward}")

        if makespan is not None:
            # Calculate gap as a percentage
            optimality_gap = ((makespan - bks) / bks) * 100
            
            csv_data.append({
                "iteration": iteration,
                "makespan": makespan,
                "bks": bks,
                "optimality gap": optimality_gap
            })
            
            print(f"  -> Makespan: {makespan:.2f} | Optimality Gap: {optimality_gap:.2f}%")

            if optimality_gap <= 20.0:
                print(f"Goal Reached! Optimality gap is {optimality_gap:.2f}% (<= 20%). Stopping early.")
                trainer.save(save_dir) # Save final model before exiting
                break
        else:
            print("  -> Makespan metric not found in custom_metrics. Cannot calculate gap.")

        if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
            ckpt_path = trainer.save(save_dir)
            print(f"Checkpoint saved at iteration {iteration}: {ckpt_path}")

    # Export CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(save_dir, f"results_gap_{int(time.time())}.csv")
    
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "makespan", "bks", "optimality gap"])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"Training summary saved to {csv_file_path}")

    ray.shutdown()

if __name__ == "__main__":
    train_func(args.instances, args.iters, args.out, args.bks)
