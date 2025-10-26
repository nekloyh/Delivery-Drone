# training/train_fixed_ppo.py
import argparse
import json
import itertools
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.indoor_drone_env import IndoorDroneEnv

def make_env(args, fixed_cfg):
    def _f():
        env_config = {
            "dt": args.dt,
            "airsim_ip": args.airsim_ip, 
            "airsim_port": args.airsim_port,
            "feature_addr": args.feature_addr or None,
            "reward": {
                "goal_bonus":500.0, 
                "dist_coeff":-5.0, 
                "dt_coeff":-0.1,
                "act_quadratic":-0.01, 
                "collision":-1000.0,
                "jerk":-10.0
                },
            "limits": {"vxy":1.0, "vz":0.5, "yaw_rate":0.5, "rate_limit":True},
            "curriculum": True,
            "spawn_actor_name": args.spawn_actor_name,
            "target_actor_names": args.target_actor_names.split(','),
            # --------------------------------
        }
        env = IndoorDroneEnv(env_config)
        return env
    return _f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airsim-ip", default="127.0.0.1")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--feature-addr", default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--fixed-ned-json", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn-actor-name", default="DroneSpawn", help="Name of Actor to spawn in UE.")
    ap.add_argument("--target-actor-names", default="TargetSpawn_1,TargetSpawn_2,TargetSpawn_3", help="List of target Actor names, separated by commas.")
    # ---------------------------------------
    
    args = ap.parse_args()

    fixed_cfg = json.load(open(args.fixed_ned_json, "r"))
    # Giữ lại dòng này nếu bạn muốn kiểm tra khung tọa độ, nhưng nó không còn quan trọng vì chúng ta dùng Actor
    # assert fixed_cfg.get("frame") == "ned_m", "Expected NED meters frame"

    env = DummyVecEnv([make_env(args, fixed_cfg)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[256,256,256], vf=[256,256,256])],
        ortho_init=False, log_std_init=-1.0
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save("ppo_model.zip")
    env.save("vecnorm_env.pkl")

if __name__ == "__main__":
    main()