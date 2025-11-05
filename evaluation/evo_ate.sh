#!/usr/bin/env bash
# Compute Absolute Trajectory Error (ATE) and Relative Pose Error (RPE)
# using the evo toolbox.
#
# Usage:
#   bash evaluation/evo_ate.sh <reference_pose_file> <estimated_pose_file>
#
# The `reference_pose_file` should be a trajectory in TUM format
# representing the ground truth (e.g. AirSim GT) and `estimated_pose_file`
# should be the output from ORB‑SLAM3.  See the evo documentation for
# accepted formats.

set -e

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <reference_pose> <estimated_pose>" >&2
  exit 1
fi

ref=$1
est=$2

# Compute Absolute Trajectory Error (ATE)
echo "Computing ATE..."
evo_ape tum "$ref" "$est" --align --plot --plot_mode xz

# Compute Relative Pose Error (RPE)
echo "Computing RPE..."
evo_rpe tum "$ref" "$est" --delta 1 --relative --plot --plot_mode xz
