from typing import Iterable
import numpy as np

def compute_reward(
    dist_to_goal: float,
    dt: float,
    rotor_thrusts: Iterable[float],
    collided: bool,
    jerk_mag: float,
    reached_goal: bool,
) -> float:
    reward = 0.0

    if reached_goal:
        reward += 500.0

    reward += -5.0 * dist_to_goal

    reward += -0.1 * dt

    thrusts = np.asarray(list(rotor_thrusts), dtype=np.float32)
    energy = float(np.sum(thrusts ** 2))
    reward += -0.01 * energy

    if collided:
        reward += -1000.0

    reward += -10.0 * jerk_mag

    return reward
