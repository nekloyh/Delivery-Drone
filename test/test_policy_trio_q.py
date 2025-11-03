
import os
import argparse
import yaml
import numpy as np
import torch
from envs.triq_indoor_drone_env import IndoorDroneEnv
from algos.deep_triple_q import DeepTripleQ

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg.get("device", "cpu"))

    env = IndoorDroneEnv(
        map_path=cfg["env"]["map_path"],
        resolution=cfg["env"]["resolution"],
        width=cfg["env"]["width"], height=cfg["env"]["height"],
        obstacle_density=cfg["env"]["obstacle_density"],
        radius=cfg["env"]["radius"],
        max_speed=cfg["env"]["max_speed"],
        dt=cfg["env"]["dt"],
        max_steps=cfg["env"]["max_steps"],
        reach_dist=cfg["env"]["reach_dist"],
        utility_penalty_near=cfg["env"]["utility_penalty_near"],
        near_margin=cfg["env"]["near_margin"],
        seed=cfg["algo"]["seed"],
    )
    agent = DeepTripleQ(env, cfg, device=device)
    agent.load(args.checkpoint)

    ep, total_ret, total_util, success, viol = 0, 0.0, 0.0, 0, 0
    N = args.episodes
    for i in range(N):
        obs, _ = env.reset()
        ep_ret, ep_util = 0.0, 0.0
        while True:
            # deterministic act
            agent.actor.eval()
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a = agent.actor.act(obs_t, deterministic=True)[0].cpu().numpy()
            agent.actor.train()

            next_obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            ep_util += info.get("utility", 1.0)
            obs = next_obs
            if term or trunc:
                success += int(info.get("reached", False))
                viol += int(info.get("collided", False))
                total_ret += ep_ret
                total_util += ep_util
                break

    print(f"Eval over {N} episodes: return={total_ret/N:.2f} utility={total_util/N:.2f} success={success/N:.2%} viol={viol/N:.2%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/indoor.yaml")
    p.add_argument("--episodes", type=int, default=50)
    main(p.parse_args())
