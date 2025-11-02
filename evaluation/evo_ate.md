# Trajectory error evaluation

The `evo_ate.sh` script is a convenience wrapper around the
[`evo`](https://github.com/MichaelGrupp/evo) toolbox, which computes
trajectory accuracy metrics such as Absolute Trajectory Error (ATE)
and Relative Pose Error (RPE).

## Inputs

The script expects two arguments:

1. **Reference pose file** — a TUM format trajectory representing the
   ground truth.  In our experiments this could be the pose logged
   directly from AirSim when using the ground truth track.
2. **Estimated pose file** — a TUM format trajectory representing the
   estimated poses from ORB‑SLAM3.

You can generate these files using the built‑in logging functions
available in `airsim_ros_pkgs` and `orb_slam3_ros`.  Make sure that
both trajectories are time‑aligned and sampled at similar rates.  You
may need to use the `--align` flag when running `evo_ape` to align
the coordinate frames.

## Commands used

The script runs the following commands:

```bash
evo_ape tum <ref> <est> --align --plot --plot_mode xz
evo_rpe tum <ref> <est> --delta 1 --relative --plot --plot_mode xz
```

* `evo_ape` computes the absolute trajectory error between two
  trajectories, aligning them if necessary.  The `--plot` flag
  produces a plot of the error over time and `--plot_mode xz` limits
  the plot to the x‑z plane for clarity.
* `evo_rpe` computes the relative pose error over a fixed time
  interval (`--delta 1` second).  The `--relative` flag means the
  error is computed relative to the preceding pose rather than to a
  global frame.

## Extending the script

You may wish to modify or extend this script to compute additional
statistics or to batch process multiple trajectories.  Consult the
`evo` documentation for many more options (e.g. exporting results to
CSV).
