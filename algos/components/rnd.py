
import torch, torch.nn as nn, torch.nn.functional as F

class RNDTarget(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x): return self.net(x)

class RNDPredictor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )
    def forward(self, x): return self.net(x)
