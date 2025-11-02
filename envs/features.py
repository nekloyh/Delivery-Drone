"""Feature extraction for drone observation space.

This module constructs the observation vector from sensor data, including:
- Position and velocity
- Goal vector
- Battery level
- Occupancy histogram from map points
- Relative Pose Error (RPE) from SLAM
"""

from typing import Tuple
import numpy as np


def occupancy_histogram(
    points_xyz: np.ndarray, 
    nbins: int = 24, 
    radius: float = 5.0
) -> np.ndarray:
    """Compute angular occupancy histogram from local map points.
    
    Divides 360° around the drone into bins and counts nearby obstacles
    in each direction. This provides a compact representation of the local
    environment for obstacle avoidance.
    
    Args:
        points_xyz: Nx3 array of points in drone local frame
        nbins: Number of angular bins (default: 24 for 15° resolution)
        radius: Maximum distance to consider points (meters)
        
    Returns:
        Normalized histogram of length nbins
    """
    if points_xyz.size == 0:
        return np.zeros(nbins, dtype=np.float32)

    xy = points_xyz[:, :2]
    distances = np.linalg.norm(xy, axis=1)
    mask = distances <= radius
    
    if not np.any(mask):
        return np.zeros(nbins, dtype=np.float32)

    angles = np.arctan2(xy[mask, 1], xy[mask, 0])
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    hist, _ = np.histogram(angles, bins=bins)
    hist = hist.astype(np.float32)
    
    total = hist.sum()
    return hist / total if total > 0 else hist


def build_state(
    position: Tuple[float, float, float],
    velocity: Tuple[float, float, float],
    goal_vector: Tuple[float, float, float],
    battery_frac: float,
    map_points_local: np.ndarray,
    rpe: float,
) -> np.ndarray:
    """Build observation vector from sensor data.
    
    Observation structure (35 dimensions):
    - position (3): current x, y, z in NED frame
    - velocity (3): current vx, vy, vz
    - goal_vector (3): direction to goal
    - battery (1): remaining battery fraction [0, 1]
    - occupancy_histogram (24): angular distribution of obstacles
    - rpe (1): SLAM relative pose error
    
    Args:
        position: Current position (x, y, z)
        velocity: Current velocity (vx, vy, vz)
        goal_vector: Vector pointing to goal
        battery_frac: Battery level [0, 1]
        map_points_local: Nx3 array of nearby map points
        rpe: Relative pose error from SLAM
        
    Returns:
        Observation vector of shape (35,)
    """
    occ = occupancy_histogram(map_points_local, nbins=24, radius=5.0)

    state = np.concatenate([
        np.asarray(position, dtype=np.float32),
        np.asarray(velocity, dtype=np.float32),
        np.asarray(goal_vector, dtype=np.float32),
        np.asarray([battery_frac], dtype=np.float32),
        occ,
        np.asarray([rpe], dtype=np.float32),
    ])
    
    assert state.shape[0] == 35, f"Expected state length 35, got {state.shape[0]}"
    return state
