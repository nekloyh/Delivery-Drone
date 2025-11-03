
import os, argparse, yaml, numpy as np, torch

from envs.triq_indoor_drone_env import IndoorDroneEnv
from algos.deep_triple_q import DeepTripleQ

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cpu"))

    env_cfg = cfg.get("env", {})
    env_cfg["seed"] = cfg.get("algo", {}).get("seed") # Pass seed for reproducibility
    env = IndoorDroneEnv(cfg=env_cfg)

    agent = DeepTripleQ(env, cfg, device=device)
    agent.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/indoor.yaml")
    args = parser.parse_args()
    main(args)