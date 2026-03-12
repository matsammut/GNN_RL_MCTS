import json
import ray
from ray.rllib.agents.ppo import PPOTrainer
from JSSPHelper import solve_taillard
import torch
import torch.nn as nn
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from models import *
import numpy as np

# Initialize ray if it's not already running
if not ray.is_initialized():
    ray.init()

def load_ppo_model(checkpoint_path, params_path):
    # 1. Load the configuration
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    with open(params_path, "r") as f:
        config = json.load(f)
    
    # 2. CLEANUP: Remove training-only configurations that cause errors in inference
    # This fixes the "ValueError: callbacks must be a callable"
    if "callbacks" in config:
        del config["callbacks"]
        
    # 3. Initialize Trainer
    # This will still use Ray to build the model architecture automatically
    trainer = PPOTrainer(config=config)
    
    # 4. Restore the weights
    trainer.restore(checkpoint_path)
    return trainer

# Paths based on your image
checkpoint_path = "PPO_ta42_checkpoint_07032026/final_checkpoint/final-checkpoint"
params_path = "PPO_ta42_checkpoint_07032026/params.json"

ppo_trainer = load_ppo_model(checkpoint_path, params_path)

class RayMCTSWrapper(nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.policy = trainer.get_policy()
        self.model = self.policy.model
        self.preprocessor = ModelCatalog.get_preprocessor_for_space(
            self.policy.observation_space
        )

    def forward(self, state):
        """
        Takes a numpy state and returns (probs, value)
        """
        # Convert numpy state to the format RLlib expects
        # Ray 1.1.0 expects a batch dimension: [1, state_dim]
        #obs = torch.from_numpy(state_array).float().unsqueeze(0)
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy().squeeze()
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = state.squeeze()
        # 3. MANUALLY PREPROCESS (Flatten the dict to the 241-vector)
        flattened_obs = self.preprocessor.transform(state)
        print(f"3DEBUG: state shape {flattened_obs}")
        # 4. COMPUTE ACTIONS
        # We pass the flattened vector in a list (batch of 1)
        out = self.policy.compute_actions([flattened_obs], explore=False)
        
        # Extract Logits (priors)
        logits = out[2]['action_dist_inputs'][0]
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)
        # Extract Value (V)
        value = out[2]["vf_preds"][0]
        return probs, value

# Initialize the wrapper
n_mcts_model = RayMCTSWrapper(ppo_trainer)

# Inside your MCTS Search loop, the expansion/evaluation step looks like this:
def evaluate_node(node, n_mcts_model):
    # n_mcts_model(node.state) now returns (priors, value) directly
    priors, value = n_mcts_model(node.state)
    return priors, value

# Call the solver
iterations=5000
final_env = solve_taillard("instances/ta41", n_mcts_model, iterations)
makespan = final_env.get_makespan()

with open("logs.txt", "a") as f:
    f.write(f"Final Makespan: {makespan}\n")
print(f"Final Makespan: {makespan}")
torch.save(n_mcts_model.state_dict(), "best_model_ta41.pth")
