"""Reward utilities for the indoor drone environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for the shaped reward function."""

    goal_bonus: float = 500.0
    dist_coeff: float = -5.0
    dt_coeff: float = -0.1
    act_quadratic: float = -0.01
    collision: float = -1000.0
    jerk: float = -10.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, float] | MutableMapping[str, float] | None) -> "RewardWeights":
        if not raw:
            return cls()
        defaults = cls()
        data = {key: float(value) for key, value in raw.items() if hasattr(defaults, key)}
        return cls(
            goal_bonus=data.get("goal_bonus", defaults.goal_bonus),
            dist_coeff=data.get("dist_coeff", defaults.dist_coeff),
            dt_coeff=data.get("dt_coeff", defaults.dt_coeff),
            act_quadratic=data.get("act_quadratic", defaults.act_quadratic),
            collision=data.get("collision", defaults.collision),
            jerk=data.get("jerk", defaults.jerk),
        )


def compute_reward(
    dist_to_goal: float,
    dt: float,
    rotor_thrusts: Iterable[float],
    collided: bool,
    jerk_mag: float,
    reached_goal: bool,
    weights: RewardWeights | None = None,
) -> float:
    """Compute the shaped reward value using configurable weights."""

    cfg = weights or RewardWeights()
    reward = 0.0

    if reached_goal:
        reward += cfg.goal_bonus

    reward += cfg.dist_coeff * dist_to_goal
    reward += cfg.dt_coeff * dt

    thrusts = np.asarray(list(rotor_thrusts), dtype=np.float32)
    energy = float(np.sum(thrusts**2))
    reward += cfg.act_quadratic * energy

    if collided:
        reward += cfg.collision

    reward += cfg.jerk * jerk_mag

    return float(reward)