import argparse
import os
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import tensorflow as tf
import numpy as np

# ====== IMPORT YOUR ENV AND MODEL ======
from models import FCMaskedActionsModelTF


# ============================
# ENV CREATION WRAPPER
# ============================
def create_env(env_config):
    return TaillardEnv(env_config)


# ============================
# TRAINING FUNCTION (NO EVAL)
# ============================
def train_func(args):

    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # PPO CONFIGURATION
    config = {
        "env": "TaillardEnv",
        "num_workers": 0,
        "num_gpus": 0,

        # PPO Settings
        "gamma": 1.0,
        "vf_clip_param": 100000.0,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.0,
        "lr": 6.6e-4,   # as used in Maharjan et al.

        # Model config
        "model": {
            "custom_model": "fc_masked_actions_tf",
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },

        "framework": "tf",
        "log_level": "WARN",

        # Episode length limit
        "env_config": {},
    }

    # Run training for each instance
    for instance_path in args.instances:

        print(f"\n=== Training PPO on instance: {instance_path} ===\n")

        config["env_config"] = {
            "instance_path": instance_path
        }

        trainer = PPOTrainer(config=config)

        # Training loop
        for i in range(args.iters):
            result = trainer.train()
            reward_mean = result["episode_reward_mean"]
            print(f"[{instance_path}] Iter {i+1}/{args.iters} reward_mean={reward_mean}")

        # Save checkpoint
        save_dir = f"ppo_mcts_results/{os.path.basename(instance_path)}"
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = trainer.save(save_dir)
        print(f"Checkpoint saved: {checkpoint}")

    ray.shutdown()


# ============================
# ARGUMENT PARSER
# ============================
if __name__ == "__main__":
    DEFAULT_INSTANCES = [
        "instances/ta42",
        "instances/ta52",
        "instances/ta62",
        "instances/ta72"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()
    train_func(args)
