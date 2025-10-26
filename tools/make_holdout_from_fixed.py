# tools/make_holdout_from_fixed.py
import json, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument(
    "--fixed-ned-json",
    required=True,
    help="maps/start_goal_fixed_ned.json (frame=ned_m)",
)
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--out", default="maps/holdout_pairs.json")
args = ap.parse_args()

with open(args.fixed_ned_json, "r") as f:
    cfg = json.load(f)
assert cfg.get("frame") == "ned_m", "Expected NED meters frame"

spawn = cfg["spawn"]  # {"x":..., "y":..., "z":..., "yaw":... (optional)}
targets = cfg["targets"]  # list of {"x","y","z",...}

pairs = []
for i in range(args.episodes):
    t = targets[i % len(targets)]
    pairs.append(
        {
            "start_xyz": [spawn["x"], spawn["y"], spawn["z"]],
            "goal_xyz": [t["x"], t["y"], t["z"]],
        }
    )

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(pairs, f, indent=2)
print("[OK] Wrote", args.out, "total episodes:", len(pairs))
