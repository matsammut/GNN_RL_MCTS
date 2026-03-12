import torch.nn as nn
import torch.nn.functional as F

class JSSPNet(nn.Module):
    def __init__(self, input_dim, n_jobs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Policy head: Probability of selecting each job
        self.policy_head = nn.Linear(256, n_jobs)
        # Value head: Estimated makespan (or negative makespan)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        pi = F.softmax(self.policy_head(x), dim=-1)
        v = self.value_head(x)
        return pi, v