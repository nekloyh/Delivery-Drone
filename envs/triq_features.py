#
# FILE: triq_features.py
#
"""Feature extraction for 35-dimensional observation space.

Components: position(3) + velocity(3) + goal_vector(3) + battery(1) + occupancy(24) + min_distance(1)
"""

from typing import Tuple
import numpy as np


def occupancy_histogram(
    points_xyz: np.ndarray, 
    nbins: int = 24, 
    radius: float = 5.0
) -> np.ndarray:
    """Compute angular occupancy histogram from local map points (hoặc Lidar)."""
    if points_xyz.size == 0:
        return np.zeros(nbins, dtype=np.float32)

    # Lidar data đã ở local, nên points_xyz[:, :2] là xy
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
    occupancy_grid: np.ndarray,  # Đổi tên từ map_points_local
    min_distance: float,         # Đổi tên từ rpe
) -> np.ndarray:
    """Build 35-dimensional observation vector."""
    # occ = occupancy_histogram(map_points_local, nbins=24, radius=5.0) # Đã tính bên ngoài

    state = np.concatenate([
        np.asarray(position, dtype=np.float32),
        np.asarray(velocity, dtype=np.float32),
        np.asarray(goal_vector, dtype=np.float32),
        np.asarray([battery_frac], dtype=np.float32),
        occupancy_grid, # Sử dụng trực tiếp
        np.asarray([min_distance], dtype=np.float32), # <-- SỬA Ở ĐÂY
    ])
    
    assert state.shape[0] == 35, f"Expected 35 dims, got {state.shape[0]}"
    return state