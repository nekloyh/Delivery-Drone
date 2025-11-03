
import torch, torch.nn as nn, torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

    def forward(self, obs):
        mu = self.net(obs)
        std = self.log_std.exp().clamp(1e-3, 1.0)
        return mu, std

    def act(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        if deterministic:
            a = mu
        else:
            a = mu + std * torch.randn_like(mu)
        a = torch.tanh(a)  # bound to [-1,1]
        # scale to action space
        scaled = (a + 1) * 0.5 * (self.act_high - self.act_low) + self.act_low
        return scaled
