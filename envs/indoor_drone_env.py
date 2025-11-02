"""Gymnasium environment wrapping an AirSim indoor drone scenario."""
from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Any, Dict, Optional, List, Sequence, Tuple, Iterable, Iterator

import gymnasium as gym
import numpy as np
import zmq

try:
    import airsim  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The airsim module is required for IndoorDroneEnv but is not installed. "
        "Please install it with `pip install airsim` or run inside the provided Docker container."
    ) from exc

from .features import build_state
from .reward import RewardWeights, compute_reward

LOGGER = logging.getLogger(__name__)


class IndoorDroneEnv(gym.Env):
    """Indoor drone navigation environment with AirSim backend.

    The constructor accepts either a full configuration mapping or individual
    keyword overrides.  This keeps the environment backwards compatible with
    older call-sites that instantiated it without arguments while also
    supporting the modern configuration-driven workflow.
    """

    metadata = {
        "render_modes": [],
        "description": "Indoor drone navigation environment with AirSim backend",
    }

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        if overrides:
            cfg.update(overrides)

        # Goal / ZMQ / timing
        self.spawn_actor_name = str(cfg.get("spawn_actor_name", "DroneSpawn"))
        target_names_cfg = cfg.get(
            "target_actor_names", ["TargetSpawn_1", "TargetSpawn_2", "TargetSpawn_3"]
        )
        self.target_actor_names = self._coerce_name_list(target_names_cfg)
        if not self.target_actor_names:
            raise ValueError("IndoorDroneEnv requires at least one target actor name")

        # Target goal is populated during reset via UE Actors
        self.goal = np.asarray(cfg.get("goal", [10.0, 5.0, -1.5]), dtype=np.float32)
        self.target_goal_name: Optional[str] = None
        self.target_goal_pos: Optional[np.ndarray] = None

        self.dt = float(cfg.get("dt", 0.1))
        self.feature_addr = cfg.get("feature_addr", "tcp://127.0.0.1:5557")
        self.airsim_ip = cfg.get("airsim_ip", "127.0.0.1")
        self.airsim_port = int(cfg.get("airsim_port", 41451))

        self.reward_weights = RewardWeights.from_mapping(cfg.get("reward"))

        limits_cfg = cfg.get("limits", {}) or {}
        self.max_xy = float(limits_cfg.get("vxy", cfg.get("max_vxy", 1.0)))
        self.max_vz = float(limits_cfg.get("vz", cfg.get("max_vz", 0.5)))
        self.max_yaw_rate = float(limits_cfg.get("yaw_rate", cfg.get("max_yaw_rate", 0.5)))

        # Spawn config (supports absolute spawn locations)
        self.spawn_use_abs = bool(cfg.get("spawn_use_abs", False))
        self.spawn_xyz_abs: Optional[np.ndarray] = None
        if cfg.get("spawn_xyz_abs") is not None:
            self.spawn_xyz_abs = np.asarray(cfg.get("spawn_xyz_abs"), dtype=np.float32)
            if self.spawn_xyz_abs.shape != (3,):
                raise ValueError("spawn_xyz_abs must be an iterable of length 3")
        spawn_cfg = cfg.get("spawn", {}) or {}
        if self.spawn_xyz_abs is None and isinstance(spawn_cfg, dict):
            xyz = spawn_cfg.get("xyz")
            if xyz is None and {"x", "y", "z"} <= set(spawn_cfg):
                xyz = [spawn_cfg["x"], spawn_cfg["y"], spawn_cfg["z"]]
            if xyz is not None:
                self.spawn_use_abs = True
                self.spawn_xyz_abs = np.asarray(xyz, dtype=np.float32)
        if self.spawn_use_abs and self.spawn_xyz_abs is None:
            raise ValueError("spawn_use_abs=True but no spawn_xyz_abs provided")

        # Spawn based on floor heights (when spawn_use_abs=False)
        self.spawn_xy = np.asarray(cfg.get("spawn_xy", [0.0, 0.0]), dtype=np.float32)
        self.spawn_yaw = float(cfg.get("spawn_yaw", 0.0))
        self.floor_heights: List[float] = list(map(float, cfg.get("floor_heights", [-1.0, -4.5, -8.0])))
        self.spawn_floor_idx = int(cfg.get("spawn_floor_idx", 0))
        self.spawn_on_ground = bool(cfg.get("spawn_on_ground", True))
        self.spawn_agl = float(cfg.get("spawn_agl", 1.0))
        self.debug_markers = bool(cfg.get("debug_markers", True))

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

        # Action/Obs spaces
        low = np.array(
            [-self.max_xy, -self.max_xy, -self.max_vz, -self.max_yaw_rate],
            dtype=np.float32,
        )
        high = np.array(
            [self.max_xy, self.max_xy, self.max_vz, self.max_yaw_rate], dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32)

        # AirSim client with timeout
        os.environ.setdefault("AIRSIM_RPC_PORT", str(self.airsim_port))
        self.client = airsim.MultirotorClient(ip=self.airsim_ip, port=self.airsim_port)
        
        # Add timeout for connection
        try:
            self.client.confirmConnection()
            LOGGER.info(f"✅ Connected to AirSim at {self.airsim_ip}:{self.airsim_port}")
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to AirSim at {self.airsim_ip}:{self.airsim_port}. "
                "Ensure AirSim/Unreal Engine is running."
            ) from exc
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # ZMQ subscriber for feature bridge
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            self.sub.connect(self.feature_addr)
        except Exception as exc:  # pragma: no cover - connection errors logged
            LOGGER.warning("Failed to connect feature subscriber at %s: %s", self.feature_addr, exc)

        # Internal state
        self.battery: float = 1.0
        self.prev_acc: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_position: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
            raise RuntimeError(f"Received non-finite coordinates from AirSim: {arr!r}")
        return arr

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

    def _compute_jerk(self, acc: np.ndarray) -> float:
        """Compute jerk magnitude from acceleration using finite difference."""
        jerk_vec = (acc - self.prev_acc) / max(self.dt, 1e-6)
        self.prev_acc = acc.copy()
        return float(np.linalg.norm(jerk_vec))

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

    def _plot_goal_marker(self, label: str = "GOAL") -> None:
        if self.goal is None:
            return
        try:
            pos = airsim.Vector3r(float(self.goal[0]), float(self.goal[1]), float(self.goal[2]))
            self.client.simPlotPoints([pos], [1.0, 0.0, 0.0, 1.0], 40.0, -1.0, True)
            txt = f"{label}  z={self.goal[2]:.2f}"
            self.client.simPlotStrings([txt], [pos], 1.2, [1, 1, 1, 1], 120.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        if self.spawn_use_abs and self.spawn_xyz_abs is not None:
            spawn_xyz = self.spawn_xyz_abs.astype(np.float32)
            pose = airsim.Pose(
                airsim.Vector3r(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])),
                airsim.to_quaternion(0.0, 0.0, float(self.spawn_yaw)),
            )
        else:
            try:
                spawn_pose = self.client.simGetObjectPose(self.spawn_actor_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to query spawn actor '{self.spawn_actor_name}' from AirSim"
                ) from exc

            if spawn_pose is None:
                raise RuntimeError(
                    f"Spawn actor '{self.spawn_actor_name}' does not exist in the level"
                )
            spawn_xyz = self._vector3r_to_np(spawn_pose.position)
            pose = airsim.Pose(
                airsim.Vector3r(float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])),
                spawn_pose.orientation,
            )

        if self._fixed_target_cycle is not None:
            goal_xyz = np.asarray(next(self._fixed_target_cycle), dtype=np.float32)
            self.goal = goal_xyz.copy()
            self.target_goal_name = "manual"
            self.target_goal_pos = goal_xyz.copy()
        else:
            # Validate target actors exist before selecting
            self.target_goal_name = str(self.np_random.choice(self.target_actor_names))
            try:
                target_pose = self.client.simGetObjectPose(self.target_goal_name)
            except Exception as exc:
                LOGGER.error(f"Failed to query target actor '{self.target_goal_name}': {exc}")
                raise RuntimeError(
                    f"Failed to query target actor '{self.target_goal_name}' from AirSim. "
                    f"Available actors: {self.target_actor_names}"
                ) from exc
            
            if target_pose is None or not np.all(np.isfinite([
                target_pose.position.x_val, 
                target_pose.position.y_val, 
                target_pose.position.z_val
            ])):
                LOGGER.error(f"Target actor '{self.target_goal_name}' does not exist or has invalid position")
                raise RuntimeError(
                    f"Target actor '{self.target_goal_name}' does not exist in the level or has invalid position. "
                    f"Check that actor names in config match actors in Unreal Engine level."
                )
            
            self.target_goal_pos = self._vector3r_to_np(target_pose.position)
            self.goal = self.target_goal_pos.copy()
            LOGGER.debug(f"Selected target: {self.target_goal_name} at {self.goal}")

        self.client.simSetVehiclePose(pose, ignore_collision=True)

        self.battery = 1.0
        self.prev_acc = np.zeros(3, dtype=np.float32)
        self.last_position = spawn_xyz.copy()
        time.sleep(0.1)  # Allow AirSim physics to stabilise

        if self.debug_markers:
            self._plot_spawn_marker(f"SPAWN ({self.spawn_actor_name})")
            self._plot_goal_marker(f"GOAL ({self.target_goal_name})")

        obs, _ = self._get_observation()
        self.last_position = obs[:3].copy()
        return obs, {}

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
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
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
            weights=self.reward_weights,
        )

        energy = float(np.sum(thrusts**2))
        self._update_battery(energy)
        
        # Check battery depletion
        battery_depleted = self.battery <= 0.0

        obs, _ = self._get_observation()
        self.last_position = obs[:3].copy()
        terminated = reached or collided or battery_depleted
        truncated = False
        info = {
            "energy": energy * self.dt,
            "distance_to_goal": dist,
            "jerk": jerk,
            "collision": collided,
            "collisions": int(collided),
            "success": bool(reached),
            "battery_depleted": battery_depleted,
            "battery": self.battery,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        try:
            self.client.reset()
            self.client.enableApiControl(False)
            self.client.armDisarm(False)
        except Exception:
            pass
        try:
            self.sub.close()
        except Exception:
            pass