"""
RRT* baseline planner (skeleton).

This module contains a placeholder implementation of a sampling‑based
Rapidly‑exploring Random Tree (RRT*) planner.  In the context of
indoor drone navigation RRT* can generate kinodynamically feasible
trajectories that avoid obstacles but, unlike the A* baseline, it
generally does not optimise for travel time or energy.  Implementing
RRT* is substantially more involved than A* because it must sample
states, connect them using a steering function and maintain a tree of
states while rewiring for optimality.

The functions below outline what is required to build an RRT*
baseline.  You may also choose to use an existing sampling planner
library (e.g. OMPL) and wrap it for the AirSim environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
import random

# Placeholder types
Point3 = Tuple[float, float, float]
Path = List[Point3]


@dataclass
class RRTStarConfig:
    """Configuration for RRT* baseline."""
    step_size: float = 0.5
    max_iterations: int = 5000
    neighbourhood_radius: float = 1.0
    z_levels: List[float] = None

    def __post_init__(self) -> None:
        if self.z_levels is None:
            self.z_levels = [-1.5, -3.0, -4.5]


class RRTStarBaseline:
    """Skeleton class for RRT* baseline planning."""

    def __init__(self, cfg: RRTStarConfig) -> None:
        self.cfg = cfg
        # Sampling space bounds and discrete z-levels
        self.bounds_min: Optional[np.ndarray] = None
        self.bounds_max: Optional[np.ndarray] = None
        self._z_levels = np.asarray(self.cfg.z_levels, dtype=np.float32)
        # Occupancy grid (optional) for collision checking, shared with A*
        self.grid: Optional[np.ndarray] = None  # bool array (nx, ny, nz)
        self.origin: Optional[np.ndarray] = None  # world min corner
        self.voxel_size: float = 0.25

    # ---------------------------------------------------------------
    # Utilities to (optionally) build an occupancy grid
    # ---------------------------------------------------------------
    def set_bounds(self, min_corner: Point3, max_corner: Point3) -> None:
        self.bounds_min = np.asarray(min_corner, dtype=np.float32)
        self.bounds_max = np.asarray(max_corner, dtype=np.float32)

    def set_occupancy_from_points(self, points_xyz: np.ndarray, voxel_size: float = 0.25, inflate_radius: int = 1) -> None:
        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.size == 0:
            return
        mn = pts.min(axis=0) - 1.0
        mx = pts.max(axis=0) + 1.0
        self.origin = mn
        self.voxel_size = float(voxel_size)
        nx = int(math.ceil((mx[0] - mn[0]) / self.voxel_size)) + 1
        ny = int(math.ceil((mx[1] - mn[1]) / self.voxel_size)) + 1
        nz = len(self._z_levels)
        self.grid = np.zeros((nx, ny, nz), dtype=bool)
        gx = np.floor((pts[:, 0] - self.origin[0]) / self.voxel_size).astype(int)
        gy = np.floor((pts[:, 1] - self.origin[1]) / self.voxel_size).astype(int)
        z = pts[:, 2:3]
        dz = np.abs(z - self._z_levels.reshape(1, -1))
        gz = np.argmin(dz, axis=1).astype(int)
        for ix, iy, iz in zip(gx, gy, gz):
            if 0 <= ix < self.grid.shape[0] and 0 <= iy < self.grid.shape[1] and 0 <= iz < self.grid.shape[2]:
                x0 = max(0, ix - inflate_radius)
                x1 = min(self.grid.shape[0], ix + inflate_radius + 1)
                y0 = max(0, iy - inflate_radius)
                y1 = min(self.grid.shape[1], iy + inflate_radius + 1)
                self.grid[x0:x1, y0:y1, iz] = True

    def _grid_index(self, p: np.ndarray) -> Tuple[int, int, int]:
        assert self.origin is not None
        gx = int(math.floor((p[0] - self.origin[0]) / self.voxel_size))
        gy = int(math.floor((p[1] - self.origin[1]) / self.voxel_size))
        gz = int(np.argmin(np.abs(self._z_levels - p[2])))
        return gx, gy, gz

    def _free_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check collision along segment a->b using voxel sampling if grid available."""
        if self.grid is None or self.origin is None:
            return True
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / max(1e-6, self.voxel_size))))
        for t in np.linspace(0.0, 1.0, n):
            p = a * (1 - t) + b * t
            ix, iy, iz = self._grid_index(p)
            if not (0 <= ix < self.grid.shape[0] and 0 <= iy < self.grid.shape[1] and 0 <= iz < self.grid.shape[2]):
                return False
            if self.grid[ix, iy, iz]:
                return False
        return True

    def plan(self, start: Point3, goal: Point3) -> Path:
        """Plan a path using RRT*.

        Steps to implement:
        1. Define the sampling space bounds based on the environment.
        2. Sample random points and build a tree by connecting each
           sample to its nearest neighbour if the connection is collision‑free.
        3. Rewire nearby nodes to ensure asymptotic optimality.
        4. Stop when a node is within a given distance of the goal and
           construct the path by backtracking parents.

        Returns
        -------
        A list of waypoints from `start` to `goal` in world coordinates.
        """
        start_p = np.asarray(start, dtype=np.float32)
        goal_p = np.asarray(goal, dtype=np.float32)
        # Bounds
        if self.bounds_min is None or self.bounds_max is None:
            mn = np.minimum(start_p, goal_p) - 2.0
            mx = np.maximum(start_p, goal_p) + 2.0
            self.set_bounds(tuple(mn.tolist()), tuple(mx.tolist()))
        assert self.bounds_min is not None and self.bounds_max is not None

        step = float(self.cfg.step_size)
        radius = float(self.cfg.neighbourhood_radius)

        class Node:
            __slots__ = ('p', 'parent', 'cost')
            def __init__(self, p: np.ndarray, parent: Optional[int], cost: float) -> None:
                self.p = p
                self.parent = parent
                self.cost = cost

        nodes: List[Node] = [Node(start_p, parent=None, cost=0.0)]
        goal_index: Optional[int] = None

        def sample_point() -> np.ndarray:
            x = random.uniform(self.bounds_min[0], self.bounds_max[0])
            y = random.uniform(self.bounds_min[1], self.bounds_max[1])
            # Snap z to a random discrete level to encourage floor-wise planning
            z = float(random.choice(list(self._z_levels)))
            return np.array([x, y, z], dtype=np.float32)

        def nearest(q: np.ndarray) -> int:
            d2 = [float(np.sum((n.p - q) ** 2)) for n in nodes]
            return int(np.argmin(d2))

        def steer(a: np.ndarray, b: np.ndarray, step_size: float) -> np.ndarray:
            d = b - a
            L = float(np.linalg.norm(d))
            if L <= step_size:
                return b
            return a + d / (L + 1e-8) * step_size

        def near(q: np.ndarray, rad: float) -> List[int]:
            out: List[int] = []
            r2 = rad * rad
            for i, n in enumerate(nodes):
                if float(np.sum((n.p - q) ** 2)) <= r2:
                    out.append(i)
            return out

        for _ in range(self.cfg.max_iterations):
            q_rand = sample_point()
            i_near = nearest(q_rand)
            q_new = steer(nodes[i_near].p, q_rand, step)
            if not self._free_segment(nodes[i_near].p, q_new):
                continue
            # Choose best parent among neighbours (RRT*)
            i_best = i_near
            c_best = nodes[i_near].cost + float(np.linalg.norm(q_new - nodes[i_near].p))
            for j in near(q_new, radius):
                if self._free_segment(nodes[j].p, q_new):
                    c = nodes[j].cost + float(np.linalg.norm(q_new - nodes[j].p))
                    if c < c_best:
                        c_best = c
                        i_best = j
            nodes.append(Node(q_new, parent=i_best, cost=c_best))
            i_new = len(nodes) - 1
            # Rewire neighbours through the new node if it shortens their path
            for j in near(q_new, radius):
                n = nodes[j]
                new_cost = nodes[i_new].cost + float(np.linalg.norm(n.p - q_new))
                if new_cost + 1e-6 < n.cost and self._free_segment(q_new, n.p):
                    n.parent = i_new
                    n.cost = new_cost

            # Check goal proximity
            if float(np.linalg.norm(q_new - goal_p)) < step * 2.0 and self._free_segment(q_new, goal_p):
                # Connect to goal
                nodes.append(Node(goal_p, parent=i_new, cost=nodes[i_new].cost + float(np.linalg.norm(goal_p - q_new))))
                goal_index = len(nodes) - 1
                break

        # If goal not connected, take nearest to goal
        if goal_index is None:
            i_goal = nearest(goal_p)
            goal_index = i_goal

        # Reconstruct path from goal_index back to root
        path: Path = []
        i = goal_index
        visited_guard = set()
        while i is not None and i not in visited_guard:
            visited_guard.add(i)
            path.append(tuple(map(float, nodes[i].p.tolist())))
            i = nodes[i].parent  # type: ignore
        path.reverse()
        # Ensure endpoints
        if len(path) == 0 or np.linalg.norm(np.array(path[0]) - start_p) > 1e-3:
            path.insert(0, tuple(map(float, start_p.tolist())))
        if np.linalg.norm(np.array(path[-1]) - goal_p) > 1e-3:
            path.append(tuple(map(float, goal_p.tolist())))
        return path

    def track(self, path: Path) -> Dict[str, float]:
        """Track the generated RRT* path.

        Use a controller similar to the one described in the A* baseline
        to follow the waypoints.  Log performance metrics for
        comparison with the RL agent.
        """
        try:
            import airsim  # type: ignore
        except Exception as e:
            raise ImportError('AirSim is required for path tracking') from e

        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()

        dt = 0.1
        time_s = 0.0
        energy_total = 0.0
        jerk_total = 0.0
        max_jerk = 0.0
        collisions = 0
        battery = 1.0
        prev_acc = np.zeros(3, dtype=np.float32)

        path_len_m = 0.0
        for i in range(1, len(path)):
            p0 = np.array(path[i - 1])
            p1 = np.array(path[i])
            path_len_m += float(np.linalg.norm(p1 - p0))

        for wx, wy, wz in path:
            while True:
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
                d = np.array([wx, wy, wz], dtype=np.float32) - p
                dist = float(np.linalg.norm(d))
                if dist < 0.25:
                    break
                speed = float(min(self.cfg.step_size, max(0.1, dist)))
                client.moveToPositionAsync(wx, wy, wz, speed, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                           yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)).join()
                state = client.getMultirotorState()
                acc = np.array([
                    state.kinematics_estimated.linear_acceleration.x_val,
                    state.kinematics_estimated.linear_acceleration.y_val,
                    state.kinematics_estimated.linear_acceleration.z_val,
                ], dtype=np.float32)
                jerk = float(np.linalg.norm((acc - prev_acc) / dt))
                prev_acc = acc
                jerk_total += jerk
                max_jerk = max(max_jerk, jerk)
                try:
                    rotors = client.getRotorStates().rotors  # type: ignore
                    thrusts = np.array([r.thrust for r in rotors], dtype=np.float32)
                    if thrusts.size == 0:
                        thrusts = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                except Exception:
                    thrusts = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                energy = float(np.sum(thrusts ** 2))
                energy_total += energy * dt
                battery = float(max(0.0, battery - 1e-4 * energy * dt))
                time_s += dt
                if client.simGetCollisionInfo().has_collided:
                    collisions += 1
                    break

        success = int(collisions == 0 and np.linalg.norm(np.array(path[-1]) - p) < 0.3)
        metrics: Dict[str, float] = {
            'success': success,
            'collisions': collisions,
            'time_s': time_s,
            'energy_j': energy_total,
            'path_len_m': path_len_m,
            'mean_jerk': jerk_total / max(1, int(time_s / dt)),
            'max_jerk': max_jerk,
            'battery_used_frac': float(1.0 - battery),
        }
        try:
            client.landAsync().join()
        except Exception:
            pass
        client.enableApiControl(False)
        client.armDisarm(False)
        return metrics
