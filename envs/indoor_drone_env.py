import os
import time
from typing import Any, Dict, Optional, List

import gymnasium as gym
import numpy as np
import zmq

try:
    import airsim  # type: ignore
except ImportError:
    raise ImportError(
        "The airsim module is required for IndoorDroneEnv but is not installed. "
        "Please install it with `pip install airsim` or run inside the provided Docker container."
    )

from .features import build_state
from .reward import compute_reward


class IndoorDroneEnv(gym.Env):
    metadata = {
        "render_modes": [],
        "description": "Indoor drone navigation environment with AirSim backend",
    }

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = cfg or {}

        # Goal / ZMQ / timing
        self.spawn_actor_name = cfg.get("spawn_actor_name", "DroneSpawn")
        self.target_actor_names = cfg.get("target_actor_names", ["TargetSpawn_1", "TargetSpawn_2", "TargetSpawn_3"])
        # Tọa độ mục tiêu (sẽ được cập nhật trong reset)
        self.goal = np.asarray(cfg.get("goal", [10.0, 5.0, -1.5]), dtype=np.float32)
        self.target_goal_name = None # Tên Actor mục tiêu được chọn ngẫu nhiên trong reset
        self.target_goal_pos = None # Vị trí tuyệt đối của mục tiêu được lấy từ AirSim trong reset
        self.dt = float(cfg.get("dt", 0.1))
        self.feature_addr = cfg.get("feature_addr", "tcp://127.0.0.1:5557")
        self.airsim_ip = cfg.get("airsim_ip", "127.0.0.1")
        self.airsim_port = int(cfg.get("airsim_port", 41451))

        # Spawn config (thêm mode tuyệt đối)
        self.spawn_use_abs = bool(cfg.get("spawn_use_abs", False))
        self.spawn_xyz_abs = None
        if cfg.get("spawn_xyz_abs") is not None:
            self.spawn_xyz_abs = np.asarray(cfg.get("spawn_xyz_abs"), dtype=np.float32)

        # Spawn theo sàn/tầng (khi spawn_use_abs=False)
        self.spawn_xy = np.asarray(cfg.get("spawn_xy", [0.0, 0.0]), dtype=np.float32)
        self.spawn_yaw = float(cfg.get("spawn_yaw", 0.0))
        self.floor_heights: List[float] = list(
            map(float, cfg.get("floor_heights", [-1.0, -4.5, -8.0]))
        )
        self.spawn_floor_idx = int(cfg.get("spawn_floor_idx", 0))
        self.spawn_on_ground = bool(cfg.get("spawn_on_ground", True))
        self.spawn_agl = float(cfg.get("spawn_agl", 1.0))
        self.debug_markers = bool(cfg.get("debug_markers", True))

        # Action/Obs spaces
        self.max_xy = 1.0
        self.max_vz = 0.5
        self.max_yaw_rate = 0.5
        low = np.array(
            [-self.max_xy, -self.max_xy, -self.max_vz, -self.max_yaw_rate],
            dtype=np.float32,
        )
        high = np.array(
            [self.max_xy, self.max_xy, self.max_vz, self.max_yaw_rate], dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(35,), dtype=np.float32
        )

        # AirSim client
        try:
            os.environ["AIRSIM_RPC_PORT"] = str(self.airsim_port)
        except Exception:
            pass
        self.client = airsim.MultirotorClient(ip=self.airsim_ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # ZMQ sub
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            self.sub.connect(self.feature_addr)
        except Exception:
            pass

        # Internal
        self.battery: float = 1.0
        self.prev_acc: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_position: Optional[np.ndarray] = None

    # -------------- helpers --------------
    def _read_bridge(self, timeout: float = 0.05):
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        socks = dict(poller.poll(int(timeout * 1000)))
        if self.sub not in socks:
            return None
        try:
            msg = self.sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        p = np.array(msg.get("p", [0.0, 0.0, 0.0]), dtype=np.float32)
        pts = msg.get("map_points", []) or []
        map_pts = (
            np.array(pts, dtype=np.float32)
            if len(pts) > 0
            else np.empty((0, 3), dtype=np.float32)
        )
        rpe = float(msg.get("rpe", 0.0))
        return p, map_pts, rpe

    def _get_ground_truth_state(self):
        st = self.client.getMultirotorState()
        pos = st.kinematics_estimated.position
        p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        return p, np.empty((0, 3), dtype=np.float32), 0.0

    def _compute_jerk(self, current_acc: np.ndarray) -> float:
        jerk = (current_acc - self.prev_acc) / max(self.dt, 1e-6)
        self.prev_acc = current_acc
        return float(np.linalg.norm(jerk))

    def _update_battery(self, energy: float) -> None:
        self.battery = float(max(0.0, self.battery - 1e-4 * energy * self.dt))

    def _resolve_floor_z(self) -> float:
        if len(self.floor_heights) == 0:
            return -1.0
        idx = min(max(self.spawn_floor_idx, 0), len(self.floor_heights) - 1)
        return float(self.floor_heights[idx])

    def _plot_spawn_marker(self, label: str = "SPAWN") -> None:
        try:
            self.client.simFlushPersistentMarkers()
        except Exception:
            pass
        try:
            st = self.client.getMultirotorState()
            p = st.kinematics_estimated.position
            pos = airsim.Vector3r(float(p.x_val), float(p.y_val), float(p.z_val))
            self.client.simPlotPoints([pos], [0.0, 1.0, 0.0, 1.0], 40.0, -1.0, True)
            txt = f"{label}  z={p.z_val:.2f}"
            self.client.simPlotStrings([txt], [pos], 1.2, [1, 1, 1, 1], 120.0)
        except Exception:
            pass

    # -------------- Gym API --------------
	    def reset(
	        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
	    ):
	        super().reset(seed=seed)
	
	        self.client.reset()
	        self.client.enableApiControl(True)
	        self.client.armDisarm(True)
	        
	        # --- 1. Xử lý World Origin Rebasing: Lấy tọa độ từ UE Actors ---
	        
	        # Lấy vị trí của DroneSpawn Actor
	        spawn_pose = self.client.simGetObjectPose(self.spawn_actor_name)
	        spawn_pos = spawn_pose.position
	        
	        # Chọn ngẫu nhiên một Target Actor và lấy vị trí của nó
	        self.target_goal_name = self.np_random.choice(self.target_actor_names)
	        target_pose = self.client.simGetObjectPose(self.target_goal_name)
	        target_pos = target_pose.position
	        
	        # Cập nhật vị trí mục tiêu tuyệt đối (cho logic step)
	        self.target_goal_pos = np.array([target_pos.x_val, target_pos.y_val, target_pos.z_val], dtype=np.float32)
	        self.goal = self.target_goal_pos # Cập nhật self.goal
	
	        # --- 2. Đặt lại vị trí Drone ---
	        
	        # Sử dụng vị trí của DroneSpawn để đặt lại drone
	        spawn_x, spawn_y, spawn_z = spawn_pos.x_val, spawn_pos.y_val, spawn_pos.z_val
	        
	        # Tùy chỉnh nhỏ: Nếu bạn muốn drone spawn ở trên mặt đất một chút (ví dụ: 1m)
	        # spawn_z -= 1.0 # Điều chỉnh Z (NED)
	        
	        pose = airsim.Pose(
	            airsim.Vector3r(spawn_x, spawn_y, spawn_z),
	            spawn_pose.orientation, # Giữ nguyên góc quay của Actor
	        )
	        # Đặt drone về vị trí của Actor Spawn
	        self.client.simSetVehiclePose(pose, ignore_collison=True)
	        
	        # --- 3. Khởi tạo trạng thái khác ---
	        
	        # Tắt các logic spawn cũ không cần thiết
	        # if not self.spawn_use_abs and not self.spawn_on_ground:
	        #     try:
	        #         self.client.takeoffAsync().join()
	        #     except Exception:
	        #         pass
	        #     self.client.moveToZAsync(spawn_z, 2.0).join()
	
	        self.battery = 1.0
	        self.prev_acc = np.zeros(3, dtype=np.float32)
	        time.sleep(0.1) # Đợi AirSim ổn định
	
	        if self.debug_markers:
	            self._plot_spawn_marker(f"SPAWN ({self.spawn_actor_name})")
	            self._plot_goal_marker(f"GOAL ({self.target_goal_name})") # Thêm marker cho mục tiêu
	
	        obs, _ = self._get_observation()
	        return obs, {}
	
	    def _plot_goal_marker(self, label: str = "GOAL") -> None:
	        try:
	            pos = airsim.Vector3r(float(self.goal[0]), float(self.goal[1]), float(self.goal[2]))
	            # Màu đỏ cho mục tiêu
	            self.client.simPlotPoints([pos], [1.0, 0.0, 0.0, 1.0], 40.0, -1.0, True) 
	            txt = f"{label}  z={self.goal[2]:.2f}"
	            self.client.simPlotStrings([txt], [pos], 1.2, [1, 1, 1, 1], 120.0)
	        except Exception:
	            pass

    def _get_observation(self):
        data = self._read_bridge()
        if data is None:
            p, map_pts, rpe = self._get_ground_truth_state()
            used_slam = False
        else:
            p, map_pts, rpe = data
            used_slam = True

        st = self.client.getMultirotorState()
        v_ = st.kinematics_estimated.linear_velocity
        v = np.array([v_.x_val, v_.y_val, v_.z_val], dtype=np.float32)

        goal_vec = self.goal - p
        local_pts = map_pts - p.reshape(1, 3) if map_pts.size > 0 else map_pts

        obs = build_state(p, v, goal_vec, self.battery, local_pts, rpe)
        return obs, used_slam

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float32
        )
        vx, vy, vz, yaw_rate = [float(x) for x in action]

        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, self.dt, airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()

        st = self.client.getMultirotorState()
        acc = np.array(
            [
                st.kinematics_estimated.linear_acceleration.x_val,
                st.kinematics_estimated.linear_acceleration.y_val,
                st.kinematics_estimated.linear_acceleration.z_val,
            ],
            dtype=np.float32,
        )
        jerk = self._compute_jerk(acc)

        try:
            rotors = self.client.getRotorStates().rotors  # type: ignore
            thrusts = np.array([r.thrust for r in rotors], dtype=np.float32)
            if thrusts.size == 0:
                thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
        except Exception:
            thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)

        collision_info = self.client.simGetCollisionInfo()
        collided = bool(collision_info.has_collided)

        if self.last_position is None:
            pos = st.kinematics_estimated.position
            position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        else:
            position = self.last_position

        dist = float(np.linalg.norm(self.goal - position))
        reached = dist < 0.3

        reward = compute_reward(
            dist_to_goal=dist,
            dt=self.dt,
            rotor_thrusts=thrusts,
            collided=collided,
            jerk_mag=jerk,
            reached_goal=reached,
        )

        energy = float(np.sum(thrusts**2))
        self._update_battery(energy)

        obs, _ = self._get_observation()
        self.last_position = obs[:3]
        terminated = reached or collided
        truncated = False
        info = {
            "energy": energy * self.dt,
            "distance_to_goal": dist,
            "jerk": jerk,
            "collision": collided,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.client.reset()
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        try:
            self.sub.close()
        except Exception:
            pass