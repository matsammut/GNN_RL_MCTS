import os
import json
import gym
import pandas as pd
import numpy as np
from models import *
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import ray


if not ray.is_initialized():
    ray.init()

with open("bks.json", "r") as f:
    bks_data = json.load(f)

def load_ppo_model(checkpoint_path, params_path):
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    with open(params_path, "r") as f:
        config = json.load(f)
    
    if "callbacks" in config:
        del config["callbacks"]
        
    config["num_workers"] = 0 
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_path)
    return trainer

def solve_instance(trainer, instance_id):
    # Path construction based on your default_config.py structure
    instance_path = f"instances/{instance_id}"
    env = gym.make('JSSEnv:jss-v1', env_config={'instance_path': instance_path})
    
    state = env.reset()
    done = False
    while not done:
        # Pass the observation and the action mask to the trainer
        action = trainer.compute_action(state, explore=False)
        state, reward, done, _ = env.step(action)
    
    # Extract makespan from the environment as seen in FIFO.py
    makespan = env.last_time_step
    return makespan

# https://scheduleopt.github.io/benchmarks/jsplib/#best-known-solutions---jsplib

model_folders = ["PPO_ta42_checkpoint_07032026", "PPO_ta52_checkpoint_28012026", "PPO_ta62_checkpoint_30012026", "PPO_ta72_checkpoint_31012026"]
base_dir = "checkpoint_results"
all_results = []

for folder in model_folders:
    checkpoint_path = os.path.join(base_dir, folder, "final_checkpoint", "final-checkpoint")
    params_path = os.path.join(base_dir, folder, "params.json")
    
    print(f"Evaluating model trained on {folder}...")
    trainer = load_ppo_model(checkpoint_path, params_path)
    
    # Determine target instance range (e.g., ta42 model evaluates ta41-ta50)
    group_num = int(folder[6]) # extract '4' from 'ta42'
    instance_range = [f"ta{i}" for i in range(group_num * 10 + 1, group_num * 10 + 11)]
    
    for inst in instance_range:
        achieved = solve_instance(trainer, inst)
        bks = bks_data[inst]
        all_results.append({
            "Model Group": f"{folder} [ta{group_num*10+1}-ta{group_num*10+10}]",
            "Instance": inst,
            "BKS": bks,
            "Achieved Result": achieved,
            "Gap (%)": round(((achieved - bks) / bks) * 100, 2)
        })

# Create and export Table
df = pd.DataFrame(all_results)
df.to_csv("tabulated_ppo_results_07032026.csv", index=False)
print("Evaluation complete. Results saved to tabulated_ppo_results_07032026.csv.")
