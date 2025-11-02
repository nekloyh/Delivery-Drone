"""PPO training entrypoint for the indoor drone environment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List

import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.indoor_drone_env import IndoorDroneEnv


DEFAULT_REWARD = {
    "goal_bonus": 500.0,
    "dist_coeff": -5.0,
    "dt_coeff": -0.1,
    "act_quadratic": -0.01,
    "collision": -1000.0,
    "jerk": -10.0,
}

DEFAULT_LIMITS = {"vxy": 1.0, "vz": 0.5, "yaw_rate": 0.5, "rate_limit": True}


def _parse_actor_names(raw: str) -> List[str]:
    names = [name.strip() for name in raw.split(",")]
    names = [name for name in names if name]
    if not names:
        raise argparse.ArgumentTypeError("--target-actor-names must contain at least one entry")
    return names


def _build_env_config(args, fixed_cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_config: Dict[str, Any] = {
        "dt": args.dt,
        "airsim_ip": args.airsim_ip,
        "airsim_port": args.airsim_port,
        "spawn_actor_name": args.spawn_actor_name,
        "target_actor_names": args.target_actor_names,
    }
    if args.feature_addr:
        env_config["feature_addr"] = args.feature_addr

    reward_cfg = dict(DEFAULT_REWARD)
    reward_cfg.update(fixed_cfg.get("reward", {}))
    env_config["reward"] = reward_cfg

    limits_cfg = dict(DEFAULT_LIMITS)
    limits_cfg.update(fixed_cfg.get("limits", {}))
    env_config["limits"] = limits_cfg

    spawn_cfg = fixed_cfg.get("spawn")
    if isinstance(spawn_cfg, dict):
        if {"x", "y", "z"} <= set(spawn_cfg):
            env_config["spawn_xyz_abs"] = [spawn_cfg["x"], spawn_cfg["y"], spawn_cfg["z"]]
            env_config["spawn_use_abs"] = True
        elif "xyz" in spawn_cfg and len(spawn_cfg["xyz"]) == 3:
            env_config["spawn_xyz_abs"] = list(spawn_cfg["xyz"])
            env_config["spawn_use_abs"] = True
        if "yaw" in spawn_cfg:
            env_config["spawn_yaw"] = spawn_cfg["yaw"]

    targets_cfg = fixed_cfg.get("targets")
    if isinstance(targets_cfg, list) and targets_cfg:
        fixed_targets = []
        for target in targets_cfg:
            if isinstance(target, dict):
                if {"x", "y", "z"} <= set(target):
                    fixed_targets.append([target["x"], target["y"], target["z"]])
                elif "xyz" in target and len(target["xyz"]) == 3:
                    fixed_targets.append(list(target["xyz"]))
        if fixed_targets:
            env_config["fixed_target_positions"] = fixed_targets

    if isinstance(fixed_cfg.get("env"), dict):
        env_config.update(fixed_cfg["env"])

    return env_config


def make_env(args, fixed_cfg):
    base_cfg = _build_env_config(args, fixed_cfg)

    def _f():
        return IndoorDroneEnv(copy.deepcopy(base_cfg))

    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--feature-addr", default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--fixed-ned-json", required=True)
    ap.add_argument("--ppo-config", default="configs/ppo_config.yaml", help="Path to PPO hyperparameter YAML config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn-actor-name", default="DroneSpawn", help="Name of Actor to spawn in UE.")
    ap.add_argument(
        "--target-actor-names",
        type=_parse_actor_names,
        default=_parse_actor_names("TargetSpawn_1,TargetSpawn_2,TargetSpawn_3"),
        help="Comma separated list of UE target actor names.",
    )

    args = ap.parse_args()

    fixed_cfg = json.load(open(args.fixed_ned_json, "r"))

    # Load PPO hyperparameters from YAML
    ppo_cfg = {}
    if os.path.exists(args.ppo_config):
        with open(args.ppo_config, "r") as f:
            ppo_cfg = yaml.safe_load(f) or {}
    else:
        print(f"[WARNING] PPO config file not found: {args.ppo_config}, using defaults")

    # Build policy_kwargs from config
    activation_fn_name = ppo_cfg.get("activation_fn", "Tanh")
    activation_fn = getattr(nn, activation_fn_name, nn.Tanh)
    
    net_arch = ppo_cfg.get("net_arch", [256, 256, 256])
    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=[dict(pi=net_arch.copy(), vf=net_arch.copy())],
        ortho_init=ppo_cfg.get("ortho_init", False),
        log_std_init=ppo_cfg.get("log_std_init", -1.0),
    )

    env = DummyVecEnv([make_env(args, fixed_cfg)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        use_sde=ppo_cfg.get("use_sde", False),
        verbose=1,
        seed=args.seed,
    )

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    finally:
        env.close()

    model.save("ppo_model.zip")
    env.save("vecnorm_env.pkl")


if __name__ == "__main__":
    main()