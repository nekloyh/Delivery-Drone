
import os, math, random, numpy as np, torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

def fanin_init(tensor, fanin=None):
    fanin = fanin or tensor.size(0)
    bound = 1. / math.sqrt(fanin)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
