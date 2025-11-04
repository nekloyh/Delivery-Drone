from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import airsim  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "The airsim package is required to run the classical baselines."
    ) from exc


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


@dataclass
class Node:
    x: float
    y: float
    parent: int
    cost: float


def ned_dist(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def world_to_grid(x: float, y: float, xy_min: Tuple[float, float], res: float) -> Tuple[int, int]:
    ix = int(math.floor((x - xy_min[0]) / res))
    iy = int(math.floor((y - xy_min[1]) / res))
    return ix, iy


def grid_to_world(ix: int, iy: int, xy_min: Tuple[float, float], res: float) -> Tuple[float, float]:
    wx = xy_min[0] + (ix + 0.5) * res
    wy = xy_min[1] + (iy + 0.5) * res
    return wx, wy


def is_in_bounds(occ: np.ndarray, ix: int, iy: int) -> bool:
    return 0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]


def is_free(occ: np.ndarray, ix: int, iy: int) -> bool:
    return is_in_bounds(occ, ix, iy) and occ[ix, iy] == 0


def collision_free_segment(
    occ: np.ndarray,
    xy_min: Tuple[float, float],
    res: float,
    start_xyz: Sequence[float],
    goal_xyz: Sequence[float],
    max_step: float | None = None,
) -> bool:
    sx, sy, _ = start_xyz
    gx, gy, _ = goal_xyz
    dist = math.hypot(gx - sx, gy - sy)
    if dist <= 1e-6:
        ix, iy = world_to_grid(sx, sy, xy_min, res)
        return is_free(occ, ix, iy)
    step = max_step or max(res * 0.5, 0.05)
    steps = max(int(math.ceil(dist / step)), 1)
    for i in range(steps + 1):
        t = i / steps
        x = sx + (gx - sx) * t
        y = sy + (gy - sy) * t
        ix, iy = world_to_grid(x, y, xy_min, res)
        if not is_free(occ, ix, iy):
            return False
    return True


def _iter_lidar_points(client: airsim.MultirotorClient, vehicle: str, lidar_name: str) -> Iterator[Tuple[float, float, float]]:
    data = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle)
    pts = np.asarray(data.point_cloud, dtype=np.float32)
    if pts.size == 0:
        return iter(())
    pts = pts.reshape(-1, 3)
    return iter(map(tuple, pts))


def build_occupancy_from_lidar(
    client: airsim.MultirotorClient,
    vehicle: str,
    xy_min: Tuple[float, float],
    xy_max: Tuple[float, float],
    res: float = 0.25,
    z_clip: Tuple[float, float] | None = None,
    lidar_name: str = "LidarSensor1",
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    xmin, ymin = xy_min
    xmax, ymax = xy_max
    width = max(int(math.ceil((xmax - xmin) / res)), 1)
    height = max(int(math.ceil((ymax - ymin) / res)), 1)
    occ = np.zeros((width, height), dtype=np.uint8)

    zmin, zmax = z_clip if z_clip is not None else (-10.0, 10.0)

    for x, y, z in _iter_lidar_points(client, vehicle, lidar_name):
        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            continue
        ix, iy = world_to_grid(x, y, xy_min, res)
        if is_in_bounds(occ, ix, iy):
            occ[ix, iy] = 1

    return occ, xy_min, res


# ---------------------------------------------------------------------------
# RRT* planner
# ---------------------------------------------------------------------------


def rrt_star(
    occ: np.ndarray,
    xy_min: Tuple[float, float],
    res: float,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    max_iters: int = 20000,
    step_size: float = 0.3,
    rewire_radius: float = 1.5,
    goal_tol: float = 0.4,
    goal_bias: float = 0.05,
) -> List[Tuple[float, float]]:
    sx, sy = start_xy
    gx, gy = goal_xy
    nodes: List[Node] = [Node(sx, sy, parent=-1, cost=0.0)]
    rng = random.Random(0)

    def nearest(x: float, y: float) -> int:
        best, bestd = 0, 1e18
        for i, n in enumerate(nodes):
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < bestd:
                bestd, best = d, i
        return best

    def near_indices(x: float, y: float, radius: float) -> List[int]:
        r2 = radius * radius
        return [i for i, n in enumerate(nodes) if (n.x - x) ** 2 + (n.y - y) ** 2 <= r2]

    for _ in range(max_iters):
        if rng.random() < goal_bias:
            rx, ry = gx, gy
        else:
            w, h = occ.shape
            rx, ry = grid_to_world(
                rng.randrange(0, w), rng.randrange(0, h), xy_min, res
            )
            if not is_free(occ, *world_to_grid(rx, ry, xy_min, res)):
                continue

        ni = nearest(rx, ry)
        nx, ny = nodes[ni].x, nodes[ni].y
        ang = math.atan2(ry - ny, rx - nx)
        tx = nx + step_size * math.cos(ang)
        ty = ny + step_size * math.sin(ang)

        if not collision_free_segment(occ, xy_min, res, (nx, ny, 0.0), (tx, ty, 0.0)):
            continue

        best_parent = ni
        best_cost = nodes[ni].cost + math.hypot(tx - nx, ty - ny)
        near = near_indices(tx, ty, rewire_radius)
        for j in near:
            nj = nodes[j]
            new_cost = nj.cost + math.hypot(tx - nj.x, ty - nj.y)
            if new_cost < best_cost and collision_free_segment(
                occ, xy_min, res, (nj.x, nj.y, 0.0), (tx, ty, 0.0)
            ):
                best_cost = new_cost
                best_parent = j

        nodes.append(Node(tx, ty, parent=best_parent, cost=best_cost))
        new_i = len(nodes) - 1

        for j in near:
            nj = nodes[j]
            alt_cost = nodes[new_i].cost + math.hypot(nj.x - tx, nj.y - ty)
            if alt_cost + 1e-9 < nj.cost and collision_free_segment(
                occ, xy_min, res, (tx, ty, 0.0), (nj.x, nj.y, 0.0)
            ):
                nodes[j] = Node(nj.x, nj.y, parent=new_i, cost=alt_cost)

        if math.hypot(tx - gx, ty - gy) <= goal_tol:
            nodes.append(
                Node(
                    gx,
                    gy,
                    parent=new_i,
                    cost=nodes[new_i].cost + math.hypot(tx - gx, ty - gy),
                )
            )
            gi = len(nodes) - 1
            path: List[Tuple[float, float]] = []
            i = gi
            while i != -1:
                n = nodes[i]
                path.append((n.x, n.y))
                i = nodes[i].parent
            path.reverse()
            return path

    best_i = min(
        range(len(nodes)), key=lambda i: (nodes[i].x - gx) ** 2 + (nodes[i].y - gy) ** 2
    )
    path = []
    i = best_i
    while i != -1:
        n = nodes[i]
        path.append((n.x, n.y))
        i = nodes[i].parent
    path.reverse()
    if len(path) == 0:
        path = [(sx, sy)]
    path.append((gx, gy))
    return path


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def follow_path(
    client: airsim.MultirotorClient,
    vehicle: str,
    path_xy: List[Tuple[float, float]],
    start_z: float,
    goal_z: float,
    dt: float = 0.1,
    vxy: float = 1.0,
    vz: float = 0.5,
    yawrate: float = 0.5,
) -> dict:
    if len(path_xy) == 0:
        return dict(
            success=False,
            collisions=0,
            time_s=0.0,
            energy=0.0,
            path_len=0.0,
            mean_jerk=np.nan,
            max_jerk=np.nan,
        )
    z_seq = np.linspace(start_z, goal_z, num=len(path_xy)).tolist()

    client.enableApiControl(True, vehicle)
    client.armDisarm(True, vehicle)
    x0, y0 = path_xy[0]
    pose = airsim.Pose(airsim.Vector3r(x0, y0, z_seq[0]), airsim.to_quaternion(0, 0, 0))
    client.simSetVehiclePose(pose, True, vehicle_name=vehicle)
    time.sleep(0.05)

    collisions = 0
    energy = 0.0
    path_len = 0.0
    prev_v = np.zeros(3, dtype=float)
    jerks: List[float] = []
    t0 = time.time()
    for i in range(1, len(path_xy)):
        x, y = path_xy[i]
        z = float(z_seq[i])
        state = client.getMultirotorState(vehicle_name=vehicle)
        p = state.kinematics_estimated.position
        cx, cy, cz = p.x_val, p.y_val, p.z_val

        dx, dy = x - cx, y - cy
        dxy = math.hypot(dx, dy)
        if dxy > 1e-3:
            vx = (dx / dxy) * vxy
            vy = (dy / dxy) * vxy
        else:
            vx = vy = 0.0
        dz = z - cz
        vz_cmd = max(-vz, min(vz, dz / max(dt, 1e-2)))

        client.moveByVelocityAsync(
            vx, vy, vz_cmd, duration=dt, vehicle_name=vehicle
        ).join()
        energy += (vx * vx + vy * vy + vz_cmd * vz_cmd) * dt
        path_len += dxy
        v_now = np.array([vx, vy, vz_cmd], dtype=float)
        jerk = np.linalg.norm((v_now - prev_v) / max(dt, 1e-2))
        jerks.append(jerk)
        prev_v = v_now
        if client.getCollisionInfo(vehicle_name=vehicle).has_collided:
            collisions += 1
    t1 = time.time()

    success = ned_dist((cx, cy, cz), (path_xy[-1][0], path_xy[-1][1], z_seq[-1])) <= 0.5
    return dict(
        success=bool(success),
        collisions=int(collisions),
        time_s=float(t1 - t0),
        energy=float(energy),
        path_len=float(path_len),
        mean_jerk=(float(np.mean(jerks)) if len(jerks) > 0 else np.nan),
        max_jerk=(float(np.max(jerks)) if len(jerks) > 0 else np.nan),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_rrt_star_episodes(
    airsim_ip: str,
    airsim_port: int,
    starts_xyz: Sequence[Sequence[float]],
    goals_xyz: Sequence[Sequence[float]],
    max_iters: int = 20000,
    rewire_radius: float = 1.5,
    step_size: float = 0.3,
    dt: float = 0.1,
    vxy: float = 1.0,
    vz: float = 0.5,
    yawrate: float = 0.5,
    vehicle: str = "Drone0",
    lidar_name: str = "LidarFront",
) -> pd.DataFrame:
    client = airsim.MultirotorClient(ip=airsim_ip, port=airsim_port)
    client.confirmConnection()

    rows = []
    for ep, (s, g) in enumerate(zip(starts_xyz, goals_xyz)):
        sx, sy, sz = s
        gx, gy, gz = g
        pose = airsim.Pose(airsim.Vector3r(sx, sy, sz), airsim.to_quaternion(0, 0, 0))
        client.enableApiControl(True, vehicle)
        client.armDisarm(True, vehicle)
        client.simSetVehiclePose(pose, True, vehicle)

        margin = 8.0
        xmin = min(sx, gx) - margin
        xmax = max(sx, gx) + margin
        ymin = min(sy, gy) - margin
        ymax = max(sy, gy) + margin
        occ, xy_min, res = build_occupancy_from_lidar(
            client,
            vehicle,
            (xmin, ymin),
            (xmax, ymax),
            res=0.25,
            z_clip=(min(sz, gz) - 2.0, max(sz, gz) + 2.0),
            lidar_name=lidar_name,
        )
        if collision_free_segment(occ, xy_min, res, (sx, sy, sz), (gx, gy, gz)):
            path_xy = [(sx, sy), (gx, gy)]
        else:
            path_xy = rrt_star(
                occ,
                xy_min,
                res,
                (sx, sy),
                (gx, gy),
                max_iters=max_iters,
                step_size=step_size,
                rewire_radius=rewire_radius,
                goal_tol=0.4,
                goal_bias=0.1,
            )
            if len(path_xy) == 0:
                path_xy = [(sx, sy), (gx, gy)]

        metrics = follow_path(
            client,
            vehicle,
            path_xy,
            start_z=sz,
            goal_z=gz,
            dt=dt,
            vxy=vxy,
            vz=vz,
            yawrate=yawrate,
        )
        rows.append(
            {
                "episode_id": ep,
                "success": metrics["success"],
                "collisions": metrics["collisions"],
                "time_s": metrics["time_s"],
                "energy_j": metrics["energy"],
                "path_len_m": metrics["path_len"],
                "mean_jerk": metrics["mean_jerk"],
                "max_jerk": metrics["max_jerk"],
            }
        )
        print(
            f"[RRT*] ep={ep} success={metrics['success']} time={metrics['time_s']:.2f}s "
            f"energy={metrics['energy']:.2f} collisions={metrics['collisions']}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--airsim-ip", default="host.docker.internal")
    ap.add_argument("--airsim-port", type=int, default=41451)
    ap.add_argument("--vehicle", default="Drone1")
    ap.add_argument("--start-goal-list", required=True)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--vxy-max", type=float, default=1.0)
    ap.add_argument("--vz-max", type=float, default=0.5)
    ap.add_argument("--yawrate-max", type=float, default=0.5)
    ap.add_argument("--max-iters", type=int, default=20000)
    ap.add_argument("--rewire-radius", type=float, default=1.5)
    ap.add_argument("--step-size", type=float, default=0.3)
    ap.add_argument("--lidar-name", default="LidarSensor1")
    ap.add_argument("--log", default="/workspace/out/rrt_star_eval.parquet")
    args = ap.parse_args()

    pairs = json.load(open(args.start_goal_list, "r"))
    if args.episodes is not None:
        pairs = pairs[: args.episodes]
    starts = [p["start_xyz"] for p in pairs]
    goals = [p["goal_xyz"] for p in pairs]

    df = run_rrt_star_episodes(
        args.airsim_ip,
        args.airsim_port,
        starts,
        goals,
        max_iters=args.max_iters,
        rewire_radius=args.rewire_radius,
        step_size=args.step_size,
        dt=args.dt,
        vxy=args.vxy_max,
        vz=args.vz_max,
        yawrate=args.yawrate_max,
        vehicle=args.vehicle,
        lidar_name=args.lidar_name,
    )
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    try:
        df.to_parquet(args.log)
        print("[RRT*] wrote", args.log)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        alt = os.path.splitext(args.log)[0] + ".csv"
        df.to_csv(alt, index=False)
        print("[RRT*] parquet failed, wrote CSV:", alt, "err=", exc)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
