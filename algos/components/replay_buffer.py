
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity, 1), dtype=np.float32)
        self.utils = np.zeros((self.capacity, 1), dtype=np.float32)  # per-step utility in [0,1]
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs, act, rew, utility, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.rews[i] = rew
        self.utils[i] = utility
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def size(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size):
        n = self.size()
        idx = np.random.randint(0, n, size=batch_size)
        return (
            self.obs[idx], self.acts[idx], self.rews[idx], self.utils[idx],
            self.next_obs[idx], self.dones[idx]
        )
