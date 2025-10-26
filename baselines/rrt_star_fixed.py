# baselines/rrt_star_fixed.py
import argparse, json
from baselines.rrt_star_baseline import run_rrt_star_episodes  # assume your repo has this

ap = argparse.ArgumentParser()
ap.add_argument("--airsim-ip", default="127.0.0.1")
ap.add_argument("--airsim-port", type=int, default=41451)
ap.add_argument("--dt", type=float, default=0.1)
ap.add_argument("--vxy-max", type=float, default=1.0)
ap.add_argument("--vz-max", type=float, default=0.5)
ap.add_argument("--yawrate-max", type=float, default=0.5)
ap.add_argument("--max-iters", type=int, default=20000)
ap.add_argument("--rewire-radius", type=float, default=1.5)
ap.add_argument("--step-size", type=float, default=0.3)
ap.add_argument("--fixed-ned-json", required=True)
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--log", default="out/rrt_star_fixed.parquet")
args=ap.parse_args()

cfg=json.load(open(args.fixed_ned_json,"r"))
starts=[cfg["spawn"]]*args.episodes
goals=(cfg["targets"]*(args.episodes//len(cfg["targets"])+1))[:args.episodes]

df = run_rrt_star_episodes(args.airsim_ip, args.airsim_port, starts, goals, args.max_iters,
                           args.rewire_radius, args.step_size, dt=args.dt,
                           vxy=args.vxy_max, vz=args.vz_max, yawrate=args.yawrate_max)
df.to_parquet(args.log); print("[OK] Wrote", args.log)
