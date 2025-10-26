# evaluation/eval_fixed.py
import argparse, json, time, numpy as np, pandas as pd, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from itertools import cycle

try:
    from envs.indoor_drone_env import IndoorDroneEnv
except Exception as e:
    raise ImportError("Cannot import IndoorDroneEnv. Ensure envs/indoor_drone_env.py exists.") from e

def make_env(args, fixed_cfg):
    goals = cycle(fixed_cfg["targets"])
    def _f():
        env = IndoorDroneEnv({
            "dt": args.dt, "airsim_ip": args.airsim_ip, "airsim_port": args.airsim_port,
            "feature_addr": args.feature_addr or None, "eval_mode": True,
            "limits": {"vxy":1.0, "vz":0.5, "yaw_rate":0.5, "rate_limit":True},
            "curriculum": False, "fixed_mode": True
        })
        if hasattr(env, "set_manual_spawn"):
            s=fixed_cfg["spawn"]; env.set_manual_spawn(s["x"],s["y"],s["z"],s.get("yaw",0.0))
        if hasattr(env, "set_next_goal_provider"):
            def next_goal():
                g=next(goals); return g["x"], g["y"], g["z"], g.get("yaw", 0.0)
            env.set_next_goal_provider(next_goal)
        return env
    return _f

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--airsim-ip", default="127.0.0.1"); ap.add_argument("--airsim-port", type=int, default=41451)
ap.add_argument("--feature-addr", default=None); ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--fixed-ned-json", required=True)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--log", default="out/eval_fixed.csv"); ap.add_argument("--parquet", default="out/eval_fixed.parquet")
args = ap.parse_args()

fixed_cfg = json.load(open(args.fixed_ned_json,"r"))
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([make_env(args, fixed_cfg)])
env = VecNormalize.load("vecnorm_fixed.pkl", env); env.training=False; env.norm_reward=False
model = PPO.load(args.ckpt, env=env, device="auto")

rows=[]
for ep in range(args.episodes):
    obs = env.reset(); done=False; info={}
    t0=time.time()
    while not done:
        action,_ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        info = infos[0]
    t1=time.time()
    rows.append({
      "episode_id": ep, "success": info.get("success", False),
      "collisions": info.get("collisions", 0), "time_s": t1-t0,
      "energy_j": info.get("energy", np.nan), "path_len_m": info.get("path_len", np.nan),
      "mean_jerk": info.get("mean_jerk", np.nan), "max_jerk": info.get("max_jerk", np.nan)
    })
df=pd.DataFrame(rows); os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
df.to_csv(args.log, index=False); 
try: df.to_parquet(args.parquet); except: pass
print(df.describe(include="all"))
