import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class GNNMaskedActionsModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs)
        nn.Module.__init__(self)

        # GNN Hyperparameters
        self.node_feature_dim = 4 # e.g., [is_completed, is_ready, proc_time, machine_id]
        self.hidden_dim = 64
        self.num_actions = action_space.n
        
        # 1. Graph Convolutional Layers (Message Passing)
        # H^(l+1) = ReLU( A * H^(l) * W )
        self.gcn1 = nn.Linear(self.node_feature_dim, self.hidden_dim)
        self.gcn2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # 2. Policy Head (Actor)
        # Maps the global graph embedding to action logits
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        
        # 3. Value Head (Critic)
        # Maps the global graph embedding to a single value estimate
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        # 1. Extract Graph Data from Observation
        # These will be provided by your updated JSSPEnv
        node_features = input_dict["obs"]["node_features"] # Shape: (Batch, Num_Nodes, Features)
        adj_matrix = input_dict["obs"]["adj_matrix"]       # Shape: (Batch, Num_Nodes, Num_Nodes)
        action_mask = input_dict["obs"]["action_mask"]     # Shape: (Batch, Num_Actions)

        # 2. Graph Convolution Step 1
        # Multiply Adjacency Matrix by Node Features to aggregate neighbors
        x = torch.bmm(adj_matrix, node_features) 
        x = torch.relu(self.gcn1(x))
        
        # 3. Graph Convolution Step 2
        x = torch.bmm(adj_matrix, x)
        x = torch.relu(self.gcn2(x))
        
        # 4. Global Graph Pooling (Mean Pooling)
        # Collapse the node dimension to get a single vector representing the whole graph
        # Shape goes from (Batch, Num_Nodes, Hidden_Dim) -> (Batch, Hidden_Dim)
        graph_embedding = torch.mean(x, dim=1)
        
        # 5. Compute Value Function (Critic)
        self._cur_value = self.value_head(graph_embedding).squeeze(1)
        
        # 6. Compute Action Logits (Actor)
        raw_logits = self.policy_head(graph_embedding)
        
        # 7. Apply Action Masking
        # Set illegal actions to a very large negative number
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = raw_logits + inf_mask
        
        return masked_logits, state

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
