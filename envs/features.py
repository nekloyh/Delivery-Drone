
from typing import Tuple
import numpy as np

def occupancy_histogram(points_xyz: np.ndarray, nbins: int = 24, radius: float = 5.0) -> np.ndarray:
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
