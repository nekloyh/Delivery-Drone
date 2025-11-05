# """Gymnasium environment wrapping an AirSim indoor drone scenario."""
# from __future__ import annotations

# import itertools
# import logging
# import os
# import time
# from typing import Any, Dict, Optional, List, Sequence, Tuple, Iterable, Iterator

# import gymnasium as gym
# import numpy as np
# import zmq

# try:
#     import airsim  # type: ignore
# except ImportError as exc:  # pragma: no cover - handled at runtime
#     raise ImportError(
#         "The airsim module is required for IndoorDroneEnv but is not installed. "
#         "Please install it with `pip install airsim` or activate the conda environment 'drone-env'."
#     ) from exc

# from .triq_features import build_state, occupancy_histogram
# from .triq_reward import RewardWeights, compute_reward

# LOGGER = logging.getLogger(__name__)


# class IndoorDroneEnv(gym.Env):
#     """Indoor drone navigation with AirSim and ORB-SLAM3.
    
#     Observation: 35-dim continuous
#     Action: 4-dim continuous (vx, vy, vz, yaw_rate)
#     Reward: R = +500·goal - 5·d - 0.1·t - 0.01·E - 1000·c - 10·j
#     """

#     metadata = {
#         "render_modes": [],
#         "description": "Indoor drone navigation environment with AirSim backend",
#     }

#     def __init__(
#         self,
#         cfg: Optional[Dict[str, Any]] = None,
#         **overrides: Any,
#     ) -> None:
#         super().__init__()
#         cfg = dict(cfg or {})
#         if overrides:
#             cfg.update(overrides)

#         # Goal / ZMQ / timing
#         self.spawn_actor_name = str(cfg.get("spawn_actor_name", "DroneSpawn"))
#         target_names_cfg = cfg.get(
#             "target_actor_names", ["Landing_101", "Landing_102", "Landing_103"]
#         )
#         self.target_actor_names = self._coerce_name_list(target_names_cfg)
#         if not self.target_actor_names:
#             raise ValueError("IndoorDroneEnv requires at least one target actor name")

#         # Target goal is populated during reset via UE Actors
#         self.goal = np.asarray(cfg.get("goal", [10.0, 5.0, -1.5]), dtype=np.float32)
#         self.target_goal_name: Optional[str] = None
#         self.target_goal_pos: Optional[np.ndarray] = None

#         self.dt = float(cfg.get("dt", 0.1))
#         self.feature_addr = ""
#         self.airsim_ip = cfg.get("airsim_ip", "127.0.0.1")  # Default to localhost for Conda
#         self.airsim_port = int(cfg.get("airsim_port", 41451))

#         self.max_steps = int(cfg.get("max_steps", 300))
#         self.reach_dist = float(cfg.get("reach_dist", 0.3))
#         self.step_count = 0
        
#         self.reward_weights = RewardWeights.from_mapping(cfg.get("reward"))

#         limits_cfg = cfg.get("limits", {}) or {}
#         self.max_xy = float(limits_cfg.get("vxy", cfg.get("max_vxy", 1.0)))
#         self.max_vz = float(limits_cfg.get("vz", cfg.get("max_vz", 0.5)))
#         self.max_yaw_rate = float(limits_cfg.get("yaw_rate", cfg.get("max_yaw_rate", 0.5)))

#         # Spawn config (supports absolute spawn locations)
#         self.spawn_use_abs = bool(cfg.get("spawn_use_abs", False))
#         self.spawn_xyz_abs: Optional[np.ndarray] = None
#         if cfg.get("spawn_xyz_abs") is not None:
#             self.spawn_xyz_abs = np.asarray(cfg.get("spawn_xyz_abs"), dtype=np.float32)
#             if self.spawn_xyz_abs.shape != (3,):
#                 raise ValueError("spawn_xyz_abs must be an iterable of length 3")
#         spawn_cfg = cfg.get("spawn", {}) or {}
#         if self.spawn_xyz_abs is None and isinstance(spawn_cfg, dict):
#             xyz = spawn_cfg.get("xyz")
#             if xyz is None and {"x", "y", "z"} <= set(spawn_cfg):
#                 xyz = [spawn_cfg["x"], spawn_cfg["y"], spawn_cfg["z"]]
#             if xyz is not None:
#                 self.spawn_use_abs = True
#                 self.spawn_xyz_abs = np.asarray(xyz, dtype=np.float32)
#         if self.spawn_use_abs and self.spawn_xyz_abs is None:
#             raise ValueError("spawn_use_abs=True but no spawn_xyz_abs provided")

#         # Spawn based on floor heights (when spawn_use_abs=False)
#         self.spawn_xy = np.asarray(cfg.get("spawn_xy", [0.0, 0.0]), dtype=np.float32)
#         self.spawn_yaw = float(cfg.get("spawn_yaw", 0.0))
#         self.floor_heights: List[float] = list(map(float, cfg.get("floor_heights", [-1.0, -4.5, -8.0])))
#         self.spawn_floor_idx = int(cfg.get("spawn_floor_idx", 0))
#         self.spawn_on_ground = bool(cfg.get("spawn_on_ground", True))
#         self.spawn_agl = float(cfg.get("spawn_agl", 1.0))
#         self.debug_markers = bool(cfg.get("debug_markers", True))

#         fixed_targets_cfg: Iterable[Any] = cfg.get("fixed_target_positions") or cfg.get("fixed_targets") or []
#         self._fixed_target_cycle: Optional[Iterator[np.ndarray]] = None
#         fixed_targets: List[np.ndarray] = []
#         for entry in fixed_targets_cfg:
#             if isinstance(entry, dict):
#                 if "xyz" in entry and len(entry["xyz"]) == 3:
#                     target_xyz = np.asarray(entry["xyz"], dtype=np.float32)
#                 elif {"x", "y", "z"} <= set(entry):
#                     target_xyz = np.asarray([entry["x"], entry["y"], entry["z"]], dtype=np.float32)
#                 else:
#                     continue
#             else:
#                 arr = np.asarray(entry, dtype=np.float32)
#                 if arr.shape != (3,):
#                     continue
#                 target_xyz = arr
#             fixed_targets.append(target_xyz)
#         if fixed_targets:
#             self._fixed_target_cycle = itertools.cycle(fixed_targets)

#         low = np.array(
#             [-self.max_xy, -self.max_xy, -self.max_vz, -self.max_yaw_rate],
#             dtype=np.float32,
#         )
#         high = np.array(
#             [self.max_xy, self.max_xy, self.max_vz, self.max_yaw_rate], dtype=np.float32
#         )
#         self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
#         self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32)

#         # AirSim client with timeout
#         os.environ.setdefault("AIRSIM_RPC_PORT", str(self.airsim_port))
#         self.client = airsim.MultirotorClient(ip=self.airsim_ip, port=self.airsim_port)
        
#         # Add timeout for connection
#         try:
#             self.client.confirmConnection()
#             LOGGER.info("Connected to AirSim at %s:%d", self.airsim_ip, self.airsim_port)
#         except Exception as exc:
#             raise ConnectionError(
#                 f"Failed to connect to AirSim at {self.airsim_ip}:{self.airsim_port}. "
#                 "Ensure AirSim/Unreal Engine is running."
#             ) from exc
        
#         try:
#             self.lidar_name = "LidarFront"
#             self.client.getLidarData(lidar_name=self.lidar_name)
#             LOGGER.info("✓ Lidar sensor '%s' connected", self.lidar_name)
#         except Exception as exc:
#             raise ConnectionError(
#                 f"Failed to connect to LidarSensor. "
#                 f"Make sure you have a Lidar named '{self.lidar_name}' on your drone."
#             ) from exc
        
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)

#         # ZMQ subscriber for feature bridge
#         # ctx = zmq.Context.instance()
#         # self.sub = ctx.socket(zmq.SUB)
#         # self.sub.setsockopt(zmq.SUBSCRIBE, b"")
#         # try:
#         #     self.sub.connect(self.feature_addr)
#         # except Exception as exc:  # pragma: no cover - connection errors logged
#         #     LOGGER.warning("Failed to connect feature subscriber at %s: %s", self.feature_addr, exc)

#         # Internal state
#         self.battery: float = 1.0
#         self.prev_acc: np.ndarray = np.zeros(3, dtype=np.float32)
#         self.last_position: Optional[np.ndarray] = None

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _coerce_name_list(raw: Any) -> List[str]:
#         if isinstance(raw, str):
#             candidates = [seg.strip() for seg in raw.split(",")]
#         elif isinstance(raw, Sequence):
#             candidates = [str(seg).strip() for seg in raw]
#         else:
#             return []
#         return [name for name in candidates if name]

#     @staticmethod
#     def _vector3r_to_np(vec: airsim.Vector3r) -> np.ndarray:
#         arr = np.array([vec.x_val, vec.y_val, vec.z_val], dtype=np.float32)
#         if not np.all(np.isfinite(arr)):
#             raise RuntimeError(f"Received non-finite coordinates from AirSim: {arr!r}")
#         return arr

#     # def _read_bridge(self, timeout: float = 0.05):
#     #     poller = zmq.Poller()
#     #     poller.register(self.sub, zmq.POLLIN)
#     #     socks = dict(poller.poll(int(timeout * 1000)))
#     #     if self.sub not in socks:
#     #         return None
#     #     try:
#     #         msg = self.sub.recv_json(flags=zmq.NOBLOCK)
#     #     except zmq.Again:
#     #         return None
#     #     p = np.array(msg.get("p", [0.0, 0.0, 0.0]), dtype=np.float32)
#     #     pts = msg.get("map_points", []) or []
#     #     map_pts = (
#     #         np.array(pts, dtype=np.float32)
#     #         if len(pts) > 0
#     #         else np.empty((0, 3), dtype=np.float32)
#     #     )
#     #     rpe = float(msg.get("rpe", 0.0))
#     #     return p, map_pts, rpe

#     # def _get_ground_truth_state(self):
#     #     st = self.client.getMultirotorState()
#     #     pos = st.kinematics_estimated.position
#     #     p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
#     #     return p, np.empty((0, 3), dtype=np.float32), 0.0

#     def _get_lidar_features(self, p_drone: np.ndarray) -> Tuple[np.ndarray, float]:
#         try:
#             lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
#         except Exception:
#             occ = np.zeros(24, dtype=np.float32)
#             min_dist = 5.0 
#             return occ, min_dist

#         points = lidar_data.point_cloud
#         if len(points) < 3:
#             occ = np.zeros(24, dtype=np.float32)
#             min_dist = 5.0
#             return occ, min_dist

#         points_xyz = np.array(points, dtype=np.float32).reshape(-1, 3)
        
#         occ = occupancy_histogram(points_xyz, nbins=24, radius=5.0)

#         distances = np.linalg.norm(points_xyz, axis=1)
#         min_dist = float(np.min(distances)) if distances.size > 0 else 5.0
        
#         return occ, min_dist

#     def _compute_jerk(self, acc: np.ndarray) -> float:
#         """Compute jerk magnitude from acceleration."""
#         jerk_vec = (acc - self.prev_acc) / max(self.dt, 1e-6)
#         self.prev_acc = acc.copy()
#         return float(np.linalg.norm(jerk_vec))

#     def _update_battery(self, energy: float) -> None:
#         """Update battery based on squared thrust consumption."""
#         self.battery = float(max(0.0, self.battery - 1e-4 * energy * self.dt))

#     def _resolve_floor_z(self) -> float:
#         if len(self.floor_heights) == 0:
#             return -1.0
#         idx = min(max(self.spawn_floor_idx, 0), len(self.floor_heights) - 1)
#         return float(self.floor_heights[idx])

#     def _plot_spawn_marker(self, label: str = "SPAWN") -> None:
#         try:
#             self.client.simFlushPersistentMarkers()
#         except Exception:
#             pass
#         try:
#             st = self.client.getMultirotorState()
#             p = st.kinematics_estimated.position
#             pos = airsim.Vector3r(float(p.x_val), float(p.y_val), float(p.z_val))
#             self.client.simPlotPoints([pos], [0.0, 1.0, 0.0, 1.0], 40.0, -1.0, True)
#             txt = f"{label}  z={p.z_val:.2f}"
#             self.client.simPlotStrings([txt], [pos], 1.2, [1, 1, 1, 1], 120.0)
#         except Exception:
#             pass

#     def _plot_goal_marker(self, label: str = "GOAL") -> None:
#         if self.goal is None:
#             return
#         try:
#             pos = airsim.Vector3r(float(self.goal[0]), float(self.goal[1]), float(self.goal[2]))
#             self.client.simPlotPoints([pos], [1.0, 0.0, 0.0, 1.0], 40.0, -1.0, True)
#             txt = f"{label}  z={self.goal[2]:.2f}"
#             self.client.simPlotStrings([txt], [pos], 1.2, [1, 1, 1, 1], 120.0)
#         except Exception:
#             pass

#     # ------------------------------------------------------------------
#     # Gym API
#     # ------------------------------------------------------------------

#     def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
#         """
#         IMPROVED RESET với proper drone initialization để tránh rơi
#         """
#         super().reset(seed=seed)
#         self.step_count = 0
        
#         # Step 1: Reset AirSim
#         self.client.reset()
#         time.sleep(0.2)  # Chờ physics reset
        
#         # Step 2: Enable API control
#         self.client.enableApiControl(True)
#         self.client.armDisarm(True)
#         time.sleep(0.1)
        
#         # Step 3: Hover BEFORE setting pose (quan trọng!)
#         # Điều này đảm bảo PID controller được khởi tạo
#         self.client.hoverAsync().join()
#         time.sleep(0.3)
        
#         # Step 4: Xác định spawn position
#         if self.spawn_use_abs and self.spawn_xyz_abs is not None:
#             spawn_xyz = self.spawn_xyz_abs.astype(np.float32)
#             pose = airsim.Pose(
#                 airsim.Vector3r(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])),
#                 airsim.to_quaternion(0.0, 0.0, float(self.spawn_yaw)),
#             )
#         else:
#             try:
#                 spawn_pose = self.client.simGetObjectPose(self.spawn_actor_name)
#             except Exception as exc:
#                 raise RuntimeError(
#                     f"Failed to query spawn actor '{self.spawn_actor_name}' from AirSim"
#                 ) from exc
            
#             if spawn_pose is None:
#                 raise RuntimeError(
#                     f"Spawn actor '{self.spawn_actor_name}' does not exist in the level"
#                 )
#             spawn_xyz = self._vector3r_to_np(spawn_pose.position)
            
#             # IMPORTANT: Thêm offset z để spawn trên không (không chạm đất)
#             if not self.spawn_on_ground:
#                 spawn_xyz[2] -= self.spawn_agl  # AirSim dùng NED coordinate (z âm = lên cao)
            
#             pose = airsim.Pose(
#                 airsim.Vector3r(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])),
#                 spawn_pose.orientation,
#             )
        
#         # Step 5: Set vehicle pose
#         self.client.simSetVehiclePose(pose, ignore_collision=True)
#         time.sleep(0.2)  # Chờ teleport hoàn tất
        
#         # Step 6: CRITICAL - Hover lại sau khi set pose
#         # Điều này activate PID controller ở vị trí mới
#         self.client.hoverAsync().join()
#         time.sleep(0.5)  # Chờ stabilization
        
#         # Step 7: Verify drone is stable
#         state = self.client.getMultirotorState()
#         vel = state.kinematics_estimated.linear_velocity
#         speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        
#         if speed > 1.0:
#             print(f"[WARNING] Drone speed {speed:.2f} m/s after reset - may be unstable!")
#             # Force hover thêm lần nữa
#             self.client.hoverAsync().join()
#             time.sleep(0.5)
        
#         # Step 8: Select goal (giữ nguyên logic cũ)
#         if self._fixed_target_cycle is not None:
#             goal_xyz = np.asarray(next(self._fixed_target_cycle), dtype=np.float32)
#             self.goal = goal_xyz.copy()
#             self.target_goal_name = "manual"
#             self.target_goal_pos = goal_xyz.copy()
#         else:
#             self.target_goal_name = str(self.np_random.choice(self.target_actor_names))
#             try:
#                 target_pose = self.client.simGetObjectPose(self.target_goal_name)
#             except Exception as exc:
#                 raise RuntimeError(
#                     f"Failed to query target actor '{self.target_goal_name}' from AirSim"
#                 ) from exc
            
#             if target_pose is None:
#                 raise RuntimeError(
#                     f"Target actor '{self.target_goal_name}' does not exist"
#                 )
            
#             self.target_goal_pos = self._vector3r_to_np(target_pose.position)
#             self.goal = self.target_goal_pos.copy()
        
#         # Step 9: Reset internal state
#         self.battery = 1.0
#         self.prev_acc = np.zeros(3, dtype=np.float32)
        
#         # Step 10: Get initial observation
#         obs = self._get_observation()
#         self.last_position = obs[:3].copy()
        
#         # Step 11: Debug markers
#         if self.debug_markers:
#             self._plot_spawn_marker(f"SPAWN ({self.spawn_actor_name})")
#             self._plot_goal_marker(f"GOAL ({self.target_goal_name})")
        
#         return obs, {}

#     def _get_observation(self):
#         # THAY THẾ HOÀN TOÀN LOGIC CŨ
#         st = self.client.getMultirotorState()
#         pos = st.kinematics_estimated.position
#         vel = st.kinematics_estimated.linear_velocity
        
#         p = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
#         v = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float32)
        
#         # Lấy feature từ LIDAR
#         occ, min_dist = self._get_lidar_features(p)

#         goal_vec = self.goal - p

#         obs = build_state(
#             position=p, 
#             velocity=v, 
#             goal_vector=goal_vec, 
#             battery_frac=self.battery, 
#             occupancy_grid=occ,  # Đổi tên cho rõ
#             min_distance=min_dist  # Đổi rpe thành min_distance
#         )
#         return obs

#     def step(self, action: np.ndarray):
#         self.step_count += 1
#         action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
#         vx, vy, vz_cmd, yaw_rate = [float(x) for x in action]
#         vz = -vz_cmd

#         self.client.moveByVelocityBodyFrameAsync(
#             vx, vy, vz, self.dt, airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
#         ).join()

#         st = self.client.getMultirotorState()
#         acc = np.array(
#             [
#                 st.kinematics_estimated.linear_acceleration.x_val,
#                 st.kinematics_estimated.linear_acceleration.y_val,
#                 st.kinematics_estimated.linear_acceleration.z_val,
#             ],
#             dtype=np.float32,
#         )
#         jerk = self._compute_jerk(acc)

#         try:
#             rotors = self.client.getRotorStates().rotors
#             thrusts = np.array([r.thrust for r in rotors], dtype=np.float32)
#             if thrusts.size == 0:
#                 thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
#         except Exception:
#             thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)

#         collision_info = self.client.simGetCollisionInfo()
#         collided = bool(collision_info.has_collided)

#         # Lấy Observation MỚI
#         obs = self._get_observation()
#         self.last_position = obs[:3].copy()
        
#         # Trích xuất dữ liệu từ observation MỚI
#         position = obs[:3]
#         min_dist = obs[-1] # <--- Dữ liệu khoảng cách TỐI THIỂU MỚI

#         dist = float(np.linalg.norm(self.goal - position))
#         reached = dist < self.reach_dist

#         reward = compute_reward(
#             dist_to_goal=dist,
#             dt=self.dt,
#             rotor_thrusts=thrusts,
#             collided=collided,
#             jerk_mag=jerk,
#             reached_goal=reached,
#             weights=self.reward_weights,
#         )

#         energy = float(np.sum(thrusts**2))
#         self._update_battery(energy)
        
#         battery_depleted = self.battery <= 0.0

#         terminated = reached or collided or battery_depleted
#         truncated = (self.step_count >= self.max_steps) and not terminated
#         info = {
#             "energy": energy * self.dt,
#             "distance_to_goal": dist,
#             "min_distance_obstacle": min_dist, # <--- TRẢ VỀ INFO CHO SAFETY FILTER
#             "jerk": jerk,
#             "collision": collided,
#             "collisions": int(collided),
#             "success": bool(reached),
#             "battery_depleted": battery_depleted,
#             "battery": self.battery,
#         }
#         return obs, reward, terminated, truncated, info

#     def render(self):
#         pass

#     # def close(self):
#     #     try:
#     #         self.client.reset()
#     #         self.client.enableApiControl(False)
#     #         self.client.armDisarm(False)
#     #     except Exception:
#     #         pass
#     #     try:
#     #         self.sub.close()
#     #     except Exception:
#     #         pass






from __future__ import annotations
import itertools
import logging
import os
import time
from typing import Any, Dict, Optional, List, Sequence, Tuple, Iterable, Iterator
import gymnasium as gym
import numpy as np

try:
    import airsim
except ImportError as exc:
    raise ImportError(
        "The airsim module is required. Install with: pip install airsim"
    ) from exc

from .triq_features import build_state, occupancy_histogram
from .triq_reward import RewardWeights, compute_reward

LOGGER = logging.getLogger(__name__)

class IndoorDroneEnv(gym.Env):
    """Indoor drone navigation with AirSim - FIXED VERSION"""
    
    metadata = {
        "render_modes": [],
        "description": "Indoor drone navigation with AirSim - stable flight version",
    }
    
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        if overrides:
            cfg.update(overrides)
        
        # === SPAWN CONFIGURATION ===
        self.spawn_actor_name = str(cfg.get("spawn_actor_name", "DroneSpawn"))
        target_names_cfg = cfg.get(
            "target_actor_names", ["Landing_101", "Landing_102", "Landing_103"]
        )
        self.target_actor_names = self._coerce_name_list(target_names_cfg)
        if not self.target_actor_names:
            raise ValueError("IndoorDroneEnv requires at least one target actor name")
        
        # Goal
        self.goal = np.asarray(cfg.get("goal", [10.0, 5.0, -1.5]), dtype=np.float32)
        self.target_goal_name: Optional[str] = None
        self.target_goal_pos: Optional[np.ndarray] = None
        
        # Timing - CRITICAL FIX: Use longer duration
        self.dt = float(cfg.get("dt", 0.1))
        self.control_duration = 1.0  # Send 1 second commands (stable)
        self.airsim_ip = cfg.get("airsim_ip", "127.0.0.1")  # Default to localhost for Conda
        self.airsim_port = int(cfg.get("airsim_port", 41451))
        self.max_steps = int(cfg.get("max_steps", 300))
        self.reach_dist = float(cfg.get("reach_dist", 0.5))
        self.step_count = 0
        
        # Reward
        self.reward_weights = RewardWeights.from_mapping(cfg.get("reward"))
        
        # Action limits
        limits_cfg = cfg.get("limits", {}) or {}
        self.max_xy = float(limits_cfg.get("vxy", cfg.get("max_vxy", 1.0)))
        self.max_vz = float(limits_cfg.get("vz", cfg.get("max_vz", 0.5)))
        self.max_yaw_rate = float(limits_cfg.get("yaw_rate", cfg.get("max_yaw_rate", 0.5)))
        
        # Spawn settings
        self.spawn_use_abs = bool(cfg.get("spawn_use_abs", False))
        self.spawn_xyz_abs: Optional[np.ndarray] = None
        
        if cfg.get("spawn_xyz_abs") is not None:
            self.spawn_xyz_abs = np.asarray(cfg.get("spawn_xyz_abs"), dtype=np.float32)
            if self.spawn_xyz_abs.shape != (3,):
                raise ValueError("spawn_xyz_abs must be [x, y, z]")
        
        self.spawn_xy = np.asarray(cfg.get("spawn_xy", [0.0, 0.0]), dtype=np.float32)
        self.spawn_yaw = float(cfg.get("spawn_yaw", 0.0))
        self.floor_heights: List[float] = list(map(float, cfg.get("floor_heights", [-1.0, -4.5, -8.0])))
        self.spawn_floor_idx = int(cfg.get("spawn_floor_idx", 0))
        self.spawn_on_ground = bool(cfg.get("spawn_on_ground", False))
        self.spawn_agl = float(cfg.get("spawn_agl", 2.0))
        self.debug_markers = bool(cfg.get("debug_markers", True))
        
        # Fixed targets
        fixed_targets_cfg: Iterable[Any] = cfg.get("fixed_target_positions") or cfg.get("fixed_targets") or []
        self._fixed_target_cycle: Optional[Iterator[np.ndarray]] = None
        fixed_targets: List[np.ndarray] = []
        
        for entry in fixed_targets_cfg:
            if isinstance(entry, dict):
                if "xyz" in entry and len(entry["xyz"]) == 3:
                    target_xyz = np.asarray(entry["xyz"], dtype=np.float32)
                elif {"x", "y", "z"} <= set(entry):
                    target_xyz = np.asarray([entry["x"], entry["y"], entry["z"]], dtype=np.float32)
                else:
                    continue
            else:
                arr = np.asarray(entry, dtype=np.float32)
                if arr.shape != (3,):
                    continue
                target_xyz = arr
            fixed_targets.append(target_xyz)
        
        if fixed_targets:
            self._fixed_target_cycle = itertools.cycle(fixed_targets)
        
        # === SPACES ===
        low = np.array(
            [-self.max_xy, -self.max_xy, -self.max_vz, -self.max_yaw_rate],
            dtype=np.float32,
        )
        high = np.array(
            [self.max_xy, self.max_xy, self.max_vz, self.max_yaw_rate], 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32)
        
        # === AIRSIM CLIENT ===
        os.environ.setdefault("AIRSIM_RPC_PORT", str(self.airsim_port))
        self.client = airsim.MultirotorClient(ip=self.airsim_ip, port=self.airsim_port)
        
        try:
            self.client.confirmConnection()
            LOGGER.info("✓ Connected to AirSim at %s:%d", self.airsim_ip, self.airsim_port)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to AirSim at {self.airsim_ip}:{self.airsim_port}"
            ) from exc
        
        # Verify Lidar
        try:
            self.lidar_name = "LidarFront"
            self.client.getLidarData(lidar_name=self.lidar_name)
            LOGGER.info("✓ Lidar sensor '%s' connected", self.lidar_name)
        except Exception as exc:
            LOGGER.warning(f"Lidar not found: {exc}")
            self.lidar_name = None
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Internal state
        self.battery: float = 1.0
        self.prev_acc: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_position: Optional[np.ndarray] = None
    
    # ====================================================================
    # HELPER METHODS
    # ====================================================================
    
    @staticmethod
    def _coerce_name_list(raw: Any) -> List[str]:
        if isinstance(raw, str):
            candidates = [seg.strip() for seg in raw.split(",")]
        elif isinstance(raw, Sequence):
            candidates = [str(seg).strip() for seg in raw]
        else:
            return []
        return [name for name in candidates if name]
    
    @staticmethod
    def _vector3r_to_np(vec: airsim.Vector3r) -> np.ndarray:
        arr = np.array([vec.x_val, vec.y_val, vec.z_val], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Non-finite coordinates: {arr!r}")
        return arr
    
    def _get_spawn_position(self) -> Tuple[np.ndarray, float]:
        """Get spawn position."""
        if self.spawn_use_abs and self.spawn_xyz_abs is not None:
            return self.spawn_xyz_abs.copy(), self.spawn_yaw
        
        # Query UE4 actor
        try:
            obj_list = self.client.simListSceneObjects(name_regex=f".*{self.spawn_actor_name}.*")
            if obj_list:
                pose = self.client.simGetObjectPose(obj_list[0])
                spawn_pos = self._vector3r_to_np(pose.position)
                if not self.spawn_on_ground:
                    spawn_pos[2] -= self.spawn_agl
                return spawn_pos, self.spawn_yaw
        except Exception as e:
            LOGGER.warning(f"Cannot find spawn actor: {e}")
        
        # Fallback
        floor_z = self.floor_heights[self.spawn_floor_idx % len(self.floor_heights)]
        if not self.spawn_on_ground:
            floor_z -= self.spawn_agl
        
        spawn_xyz = np.array([
            self.spawn_xy[0],
            self.spawn_xy[1],
            floor_z
        ], dtype=np.float32)
        
        return spawn_xyz, self.spawn_yaw
    
    def _get_target_position(self) -> np.ndarray:
        """Get target position."""
        if self._fixed_target_cycle is not None:
            return next(self._fixed_target_cycle).copy()
        
        try:
            import random
            target_name = random.choice(self.target_actor_names)
            obj_list = self.client.simListSceneObjects(name_regex=f".*{target_name}.*")
            if obj_list:
                pose = self.client.simGetObjectPose(obj_list[0])
                self.target_goal_name = obj_list[0]
                self.target_goal_pos = self._vector3r_to_np(pose.position)
                return self.target_goal_pos.copy()
        except Exception as e:
            LOGGER.warning(f"Cannot find target: {e}")
        
        return self.goal.copy()
    
    def _get_lidar_features(self, drone_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get Lidar occupancy histogram."""
        if self.lidar_name is None:
            return np.zeros(24, dtype=np.float32), 5.0
        
        try:
            lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
            
            if len(lidar_data.point_cloud) < 3:
                return np.zeros(24, dtype=np.float32), 5.0
            
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)
            
            if points.shape[0] == 0:
                return np.zeros(24, dtype=np.float32), 5.0
            
            occ = occupancy_histogram(points, nbins=24, radius=5.0)
            distances = np.linalg.norm(points, axis=1)
            min_dist = float(np.min(distances)) if distances.size > 0 else 5.0
            
            return occ, min_dist
            
        except Exception as e:
            LOGGER.warning(f"Lidar error: {e}")
            return np.zeros(24, dtype=np.float32), 5.0
    
    def _compute_jerk(self, current_acc: np.ndarray) -> float:
        """Compute jerk."""
        jerk_vec = (current_acc - self.prev_acc) / max(self.dt, 1e-6)
        self.prev_acc = current_acc.copy()
        return float(np.linalg.norm(jerk_vec))
    
    def _update_battery(self, energy: float) -> None:
        """Update battery."""
        battery_drain = energy * 0.0001
        self.battery = max(0.0, self.battery - battery_drain)
    
    def _get_observation(self) -> np.ndarray:
        """Build observation."""
        st = self.client.getMultirotorState()
        
        p = self._vector3r_to_np(st.kinematics_estimated.position)
        v = self._vector3r_to_np(st.kinematics_estimated.linear_velocity)
        
        occ, min_dist = self._get_lidar_features(p)
        goal_vec = self.goal - p
        
        obs = build_state(
            position=p,
            velocity=v,
            goal_vector=goal_vec,
            battery_frac=self.battery,
            occupancy_grid=occ,
            min_distance=min_dist
        )
        
        return obs
    
    def _is_drone_stable(self) -> bool:
        """Check if drone is stable (not falling)."""
        try:
            st = self.client.getMultirotorState()
            
            # Check landed state
            if st.landed_state == airsim.LandedState.Landed:
                return False
            
            # Check velocity (not freefalling)
            vel = self._vector3r_to_np(st.kinematics_estimated.linear_velocity)
            speed = float(np.linalg.norm(vel))
            
            # Freefalling if falling too fast
            if vel[2] > 2.0:  # z > 0 = falling down in NED
                LOGGER.warning(f"Drone freefalling! vz={vel[2]:.2f}")
                return False
            
            return True
            
        except Exception as e:
            LOGGER.warning(f"Cannot check stability: {e}")
            return True
    
    # ====================================================================
    # GYMNASIUM INTERFACE
    # ====================================================================
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment - FIXED FOR STABLE HOVER."""
        super().reset(seed=seed)
        
        # Reset AirSim
        self.client.reset()
        time.sleep(0.3)
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(0.2)
        
        # Get spawn position
        spawn_xyz, spawn_yaw = self._get_spawn_position()
        LOGGER.info(f"Spawning at: {spawn_xyz}, yaw: {spawn_yaw}")
        
        # Teleport
        pose = airsim.Pose(
            airsim.Vector3r(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])),
            airsim.to_quaternion(0, 0, spawn_yaw)
        )
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.3)
        
        # CRITICAL: Move to altitude and enable hover
        self.client.moveToZAsync(float(spawn_xyz[2]), 1.0).join()
        time.sleep(0.2)
        
        # Enable hover mode
        self.client.hoverAsync().join()
        time.sleep(0.5)
        
        # Verify stable
        if not self._is_drone_stable():
            LOGGER.warning("Drone not stable after reset! Retrying hover...")
            self.client.hoverAsync().join()
            time.sleep(0.5)
        
        # Get target
        self.goal = self._get_target_position()
        LOGGER.info(f"Target: {self.goal}")
        
        # Reset state
        self.step_count = 0
        self.battery = 1.0
        self.prev_acc = np.zeros(3, dtype=np.float32)
        self.last_position = None
        
        obs = self._get_observation()
        self.last_position = obs[:3].copy()
        
        info = {
            "spawn_position": spawn_xyz,
            "goal_position": self.goal,
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step - FIXED FOR ACTUAL FLIGHT."""
        self.step_count += 1
        
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        vx, vy, vz, yaw_rate = [float(x) for x in action]
        
        # CRITICAL FIX: Check if drone is stable before sending command
        if not self._is_drone_stable():
            LOGGER.warning(f"Step {self.step_count}: Drone unstable! Emergency hover...")
            self.client.hoverAsync().join()
            time.sleep(0.2)
        
        # CRITICAL FIX: Send command with longer duration for stability
        # moveByVelocityBodyFrameAsync: body frame (forward, right, down)
        # vz negative = up, vz positive = down (NED convention already)
        try:
            self.client.moveByVelocityBodyFrameAsync(
                vx, vy, vz, 
                self.control_duration,  # 1 second duration (not dt=0.1!)
                airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            ).join()
        except Exception as e:
            LOGGER.warning(f"Control command failed: {e}")
            self.client.hoverAsync().join()
        
        # Wait for command to execute
        time.sleep(self.dt)
        
        # Get state
        st = self.client.getMultirotorState()
        acc = np.array([
            st.kinematics_estimated.linear_acceleration.x_val,
            st.kinematics_estimated.linear_acceleration.y_val,
            st.kinematics_estimated.linear_acceleration.z_val,
        ], dtype=np.float32)
        
        jerk = self._compute_jerk(acc)
        
        # Get thrusts
        try:
            rotors = self.client.getRotorStates().rotors
            thrusts = np.array([r.thrust for r in rotors], dtype=np.float32)
            if thrusts.size == 0:
                thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
        except Exception:
            thrusts = np.array([vx, vy, vz, yaw_rate], dtype=np.float32)
        
        # Check collision
        collision_info = self.client.simGetCollisionInfo()
        collided = bool(collision_info.has_collided)
        
        # Get observation
        obs = self._get_observation()
        self.last_position = obs[:3].copy()
        
        position = obs[:3]
        min_dist = obs[-1]
        dist = float(np.linalg.norm(self.goal - position))
        reached = dist < self.reach_dist
        
        # Compute reward
        reward = compute_reward(
            dist_to_goal=dist,
            dt=self.dt,
            rotor_thrusts=thrusts,
            collided=collided,
            jerk_mag=jerk,
            reached_goal=reached,
            weights=self.reward_weights,
        )
        
        # Update battery
        energy = float(np.sum(thrusts**2))
        self._update_battery(energy)
        battery_depleted = self.battery <= 0.0
        
        # Check if drone is falling (emergency termination)
        is_stable = self._is_drone_stable()
        if not is_stable and not collided:
            LOGGER.warning(f"Step {self.step_count}: Drone falling! Terminating episode.")
            collided = True
            reward -= 500.0  # Penalty for falling
        
        # Termination
        terminated = reached or collided or battery_depleted
        truncated = (self.step_count >= self.max_steps) and not terminated
        
        info = {
            "energy": energy * self.dt,
            "distance_to_goal": dist,
            "min_distance_obstacle": min_dist,
            "jerk": jerk,
            "collision": collided,
            "collisions": int(collided),
            "success": bool(reached),
            "battery_depleted": battery_depleted,
            "battery": self.battery,
            "drone_stable": is_stable,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render (not implemented)."""
        pass
    
    def close(self):
        """Clean up."""
        try:
            self.client.reset()
            self.client.enableApiControl(False)
            self.client.armDisarm(False)
        except Exception as e:
            LOGGER.warning(f"Close error: {e}")