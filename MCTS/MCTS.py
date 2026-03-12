import math
import numpy as np
import torch
from MCTSNode import MCTSNode

class MCTS:
    def __init__(self, model, env, c_puct=0.2):
        self.model = model
        self.env = env
        self.c_puct = c_puct

    def search(self, root_state, n_iterations):
        # root_state is the dict from JSSPEnv.reset()
        root = MCTSNode(root_state)
        
        for _ in range(n_iterations):
            node = root
            search_env = self.env.copy()

            # 1. SELECTION
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                new_state, _ = search_env.step(action)
                if node.state is None:
                    node.state = new_state

            # 2. EXPANSION & EVALUATION
            if not search_env.done:
                # FIX: Extract 'observation' from the dictionary state
                # The model expects the raw numerical vector
                mask = node.state['action_mask'].flatten().astype(np.float32)
                obs_data = node.state['real_obs'].flatten().astype(np.float32)
                #print(mask)
                #print(obs_data)
                flat_obs = np.concatenate([mask, obs_data])
                print(f"New shape: {flat_obs.shape}")
                state_tensor = torch.FloatTensor(flat_obs).unsqueeze(0)
                #print(state_tensor)
                with torch.no_grad():
                    # Get policy priors and value estimate from the Neural Net
                    priors, value = self.model(state_tensor)
                
                legal_actions = search_env.get_legal_actions()
                # 3. MASKING & EXPANSION
                node.expand(legal_actions, priors.cpu().numpy())
                reward = value.item()
            else:
                reward = self.calculate_reward(search_env.get_makespan())

            # 4. BACKPROPAGATION
            while node is not None:
                node.value_sum += reward
                node.visit_count += 1
                node = node.parent
            
        return root

#    def calculate_reward(self, makespan):
        # Standardize reward for MCTS: smaller makespan = higher reward
#        return 1000.0 / max(makespan, 1)
    def calculate_reward(self, makespan):
    # k is a sensitivity parameter. 
    # Try values between 0.001 and 0.01 depending on your makespan range.
        k = 0.005 
        return np.exp(-k * (makespan - 1926)) # 1926 is the BKS for ta41
