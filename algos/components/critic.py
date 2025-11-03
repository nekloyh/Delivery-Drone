
import torch, torch.nn as nn, torch.nn.functional as F

def _mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, out_dim),
    )

class TwinCritic(nn.Module):
    """Two independent Q networks for TD3-style critics."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, 1)
        self.q2 = _mlp(obs_dim + act_dim, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def q_min(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)
