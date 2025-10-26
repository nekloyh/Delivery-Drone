def rrt_star(
    occ,
    xy_min,
    res,
    start_xy,
    goal_xy,
    max_iters=20000,
    step_size=0.3,
    rewire_radius=1.5,
    goal_tol=0.4,
    goal_bias=0.05,
):
    sx, sy = start_xy
    gx, gy = goal_xy
    nodes: List[Node] = [Node(sx, sy, parent=-1, cost=0.0)]
    rng = random.Random(0)

    def nearest(x, y):
        best, bestd = 0, 1e18
        for i, n in enumerate(nodes):
            d = (n.x - x) ** 2 + (n.y - y) ** 2
            if d < bestd:
                bestd, best = d, i
        return best

    def near_indices(x, y, radius):
        r2 = radius * radius
        idx = []
        for i, n in enumerate(nodes):
            if (n.x - x) ** 2 + (n.y - y) ** 2 <= r2:
                idx.append(i)
        return idx

    for it in range(max_iters):
        if rng.random() < goal_bias:
            rx, ry = gx, gy
        else:
            # Sample within bounding box of occ
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

        # Collision check segment
        if not collision_free_segment(occ, xy_min, res, (nx, ny, 0.0), (tx, ty, 0.0)):
            continue

        # Choose parent with minimal cost (rewire)
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

        # Rewire neighbors through new node
        for j in near:
            nj = nodes[j]
            alt_cost = nodes[new_i].cost + math.hypot(nj.x - tx, nj.y - ty)
            if alt_cost + 1e-9 < nj.cost and collision_free_segment(
                occ, xy_min, res, (tx, ty, 0.0), (nj.x, nj.y, 0.0)
            ):
                nodes[j] = Node(nj.x, nj.y, parent=new_i, cost=alt_cost)

        # Goal reached?
        if math.hypot(tx - gx, ty - gy) <= goal_tol:
            # connect to goal cell center
            nodes.append(
                Node(
                    gx,
                    gy,
                    parent=new_i,
                    cost=nodes[new_i].cost + math.hypot(tx - gx, ty - gy),
                )
            )
            gi = len(nodes) - 1
            # Reconstruct
            path = []
            i = gi
            while i != -1:
                n = nodes[i]
                path.append((n.x, n.y))
                i = nodes[i].parent
            path.reverse()
            return path

    # Failed -> return best toward goal
    best_i = min(
        range(len(nodes)), key=lambda i: (nodes[i].x - gx) ** 2 + (nodes[i].y - gy) ** 2
    )
    # Reconstruct partial
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


# ---------- execution ----------
def follow_path(
    client,
    vehicle,
    path_xy: List[Tuple[float, float]],
    start_z: float,
    goal_z: float,
    dt=0.1,
    vxy=1.0,
    vz=0.5,
    yawrate=0.5,
):
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
    jerks = []
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


# ---------- public API ----------
def run_rrt_star_episodes(
    airsim_ip,
    airsim_port,
    starts_xyz,
    goals_xyz,
    max_iters=20000,
    rewire_radius=1.5,
    step_size=0.3,
    dt=0.1,
    vxy=1.0,
    vz=0.5,
    yawrate=0.5,
    vehicle="Drone1",
):
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
        )
        # If straight-line free: trivial plan
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
                path_xy = [(sx, sy), (gx, gy)]  # fallback

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


# ---------- CLI ----------
def _main():
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
    )
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    try:
        df.to_parquet(args.log)
        print("[RRT*] wrote", args.log)
    except Exception as e:
        alt = os.path.splitext(args.log)[0] + ".csv"
        df.to_csv(alt, index=False)
        print("[RRT*] parquet failed, wrote CSV:", alt, "err=", e)


if __name__ == "__main__":
    _main()
