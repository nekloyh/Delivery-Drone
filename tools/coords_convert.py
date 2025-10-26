# tools/coords_convert.py
"""
Convert Unreal world coordinates (centimeters, Z up) to AirSim NED meters (Z down).
Usage:
  python tools/coords_convert.py --scenario configs/scenarios/fixed_unreal_cm.json         --playerstart-xcm <X0> --playerstart-ycm <Y0> --playerstart-zcm <Z0>         --out maps/start_goal_fixed_ned.json
If your numbers are already relative to PlayerStart, pass X0=Y0=Z0=0.
"""
import json, argparse, math, os

def ue_cm_to_ned_m(pt_cm, ps_cm):
    # shift by PlayerStart, convert cm->m, flip Z
    x_m = (pt_cm["x"] - ps_cm["x"])/100.0
    y_m = (pt_cm["y"] - ps_cm["y"])/100.0
    z_m = -(pt_cm["z"] - ps_cm["z"])/100.0
    yaw = math.radians(pt_cm.get("yaw_deg", 0.0))
    return dict(x=x_m, y=y_m, z=z_m, yaw=yaw)

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True)
ap.add_argument("--playerstart-xcm", type=float, default=0.0)
ap.add_argument("--playerstart-ycm", type=float, default=0.0)
ap.add_argument("--playerstart-zcm", type=float, default=0.0)
ap.add_argument("--out", default="maps/start_goal_fixed_ned.json")
args = ap.parse_args()

with open(args.scenario,"r") as f:
    sc = json.load(f)

ps = dict(x=args.playerstart_xcm, y=args.playerstart_ycm, z=args.playerstart_zcm)
spawn_ned = ue_cm_to_ned_m(sc["spawn"], ps)
goals_ned  = [ue_cm_to_ned_m(t, ps) for t in sc["targets"]]

out = {
  "frame": "ned_m",
  "spawn": spawn_ned,
  "targets": goals_ned
}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w") as f:
    json.dump(out, f, indent=2)
print("[OK] Wrote", args.out)
