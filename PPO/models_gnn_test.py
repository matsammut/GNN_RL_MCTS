import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class GCNLayer(nn.Module):
    """Single GCN layer: X' = ReLU(LayerNorm(A · X · W)) with residual connection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        # Residual projection if dimensions differ
        self.residual = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, adj):
        # x: (B, N, F_in), adj: (B, N, N) — already normalised
        h = torch.bmm(adj, x)          # neighbourhood aggregation
        h = self.linear(h)             # learnable transform
        h = self.norm(h)               # layer norm (stabilises training)
        h = h + self.residual(x)       # residual connection
        return torch.relu(h)


class GNNMaskedActionsModel(TorchModelV2, nn.Module):
    """
    GNN policy/value model with:
      - 3 GCN layers with residual connections and LayerNorm
      - Masked mean pooling (ignores padding nodes)
      - Separate actor/critic heads
      - Proper weight initialisation
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.node_feature_dim = 8   # Must match env
        self.hidden_dim = 128
        self.num_actions = action_space.n

        # ---- GCN Backbone (3 layers) ----
        self.gcn1 = GCNLayer(self.node_feature_dim, self.hidden_dim)
        self.gcn2 = GCNLayer(self.hidden_dim, self.hidden_dim)
        self.gcn3 = GCNLayer(self.hidden_dim, self.hidden_dim)

        # ---- Policy Head (Actor) ----
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions),
        )

        # ---- Value Head (Critic) ----
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self._cur_value = None

        # ---- Orthogonal Initialisation (best practice for PPO) ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for policy output (exploration)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        # Smaller init for value output
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        node_features = obs["node_features"]    # (B, N, 8)
        adj_matrix = obs["adj_matrix"]          # (B, N, N) — pre-normalised
        node_mask = obs["node_mask"]            # (B, N)
        action_mask = obs["action_mask"]        # (B, A)

        # ---- GCN Forward ----
        x = self.gcn1(node_features, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        x = self.gcn3(x, adj_matrix)

        # ---- Masked Mean Pooling (ignore padding) ----
        # node_mask: (B, N) → (B, N, 1) for broadcasting
        mask = node_mask.unsqueeze(-1)                    # (B, N, 1)
        x_masked = x * mask                               # zero out padding nodes
        graph_embedding = x_masked.sum(dim=1) / (
            mask.sum(dim=1).clamp(min=1.0)                # (B, hidden_dim)
        )

        # ---- Value ----
        self._cur_value = self.value_head(graph_embedding).squeeze(-1)

        # ---- Policy with Action Masking ----
        raw_logits = self.policy_head(graph_embedding)
        inf_mask = torch.clamp(torch.log(action_mask + 1e-8), min=-1e10)
        masked_logits = raw_logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
