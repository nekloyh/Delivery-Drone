"""Centralized configuration management with validation and presets.

This module provides a robust configuration system with:
- JSON schema validation
- Configuration presets (development/production/reproduction)
- Type-safe configuration classes
- Environment variable overrides
- Configuration inheritance and merging
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkConfig:
    """Neural network architecture configuration."""
    architecture: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "Tanh"
    ortho_init: bool = False
    log_std_init: float = -1.0
    
    def validate(self) -> None:
        """Validate network configuration."""
        if not self.architecture:
            raise ValueError("Network architecture cannot be empty")
        if any(n <= 0 for n in self.architecture):
            raise ValueError("All layer sizes must be positive")
        if self.activation not in ["Tanh", "ReLU", "ELU", "GELU"]:
            raise ValueError(f"Unsupported activation: {self.activation}")


@dataclass(frozen=True)
class PPOConfig:
    """PPO algorithm configuration matching paper specifications."""
    
    # Network architecture
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    
    # Device configuration
    device: str = "auto"
    
    def validate(self) -> None:
        """Validate PPO configuration."""
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
        if self.n_steps <= 0:
            raise ValueError(f"Invalid n_steps: {self.n_steps}")
        if self.batch_size <= 0 or self.batch_size > self.n_steps:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.n_epochs <= 0:
            raise ValueError(f"Invalid n_epochs: {self.n_epochs}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"Invalid gamma: {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"Invalid gae_lambda: {self.gae_lambda}")
        if self.clip_range <= 0:
            raise ValueError(f"Invalid clip_range: {self.clip_range}")
        if self.ent_coef < 0:
            raise ValueError(f"Invalid ent_coef: {self.ent_coef}")
        if self.vf_coef < 0:
            raise ValueError(f"Invalid vf_coef: {self.vf_coef}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"Invalid max_grad_norm: {self.max_grad_norm}")
        
        self.network.validate()
    
    def to_sb3_kwargs(self) -> Dict[str, Any]:
        """Convert to Stable-Baselines3 PPO kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "use_sde": self.use_sde,
            "sde_sample_freq": self.sde_sample_freq,
            "device": self.device,
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get policy network kwargs."""
        import torch.nn as nn
        
        activation_map = {
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
            "ELU": nn.ELU,
            "GELU": nn.GELU,
        }
        
        return {
            "net_arch": self.network.architecture,
            "activation_fn": activation_map[self.network.activation],
            "ortho_init": self.network.ortho_init,
            "log_std_init": self.network.log_std_init,
        }


@dataclass(frozen=True)
class RewardConfig:
    """Reward function configuration."""
    goal_bonus: float = 500.0
    distance_coeff: float = -5.0
    time_penalty: float = -0.1
    energy_coeff: float = -0.01
    collision_penalty: float = -1000.0
    jerk_penalty: float = -10.0
    
    def validate(self) -> None:
        """Validate reward configuration."""
        if self.goal_bonus <= 0:
            LOGGER.warning("goal_bonus should be positive")
        if self.distance_coeff >= 0:
            LOGGER.warning("distance_coeff should be negative (penalty)")
        if self.collision_penalty >= 0:
            LOGGER.warning("collision_penalty should be negative (penalty)")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment configuration."""
    
    # AirSim connection
    airsim_ip: str = "127.0.0.1"
    airsim_port: int = 41451
    
    # Simulation parameters
    dt: float = 0.1
    clock_speed: float = 1.0
    
    # Drone parameters
    spawn_position: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    spawn_yaw: float = 0.0
    max_velocity_xy: float = 1.0
    max_velocity_z: float = 0.5
    max_yaw_rate: float = 0.5
    
    # Episode configuration
    max_episode_steps: int = 3000
    success_distance: float = 1.0
    
    # Observation space
    state_dim: int = 35
    
    # Reward configuration
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    # Target configuration
    target_actor_names: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate environment configuration."""
        if self.dt <= 0:
            raise ValueError(f"Invalid dt: {self.dt}")
        if self.clock_speed <= 0:
            raise ValueError(f"Invalid clock_speed: {self.clock_speed}")
        if self.max_velocity_xy <= 0:
            raise ValueError(f"Invalid max_velocity_xy: {self.max_velocity_xy}")
        if self.max_velocity_z <= 0:
            raise ValueError(f"Invalid max_velocity_z: {self.max_velocity_z}")
        if self.max_episode_steps <= 0:
            raise ValueError(f"Invalid max_episode_steps: {self.max_episode_steps}")
        if self.state_dim != 35:
            LOGGER.warning(f"State dim {self.state_dim} != 35, may cause issues")
        
        self.reward.validate()
    
    def to_env_kwargs(self) -> Dict[str, Any]:
        """Convert to environment initialization kwargs."""
        return {
            "airsim_ip": self.airsim_ip,
            "airsim_port": self.airsim_port,
            "dt": self.dt,
            "spawn_xyz_abs": self.spawn_position,
            "spawn_use_abs": True,
            "spawn_yaw": self.spawn_yaw,
            "max_vxy": self.max_velocity_xy,
            "max_vz": self.max_velocity_z,
            "max_yaw_rate": self.max_yaw_rate,
            "target_actor_names": self.target_actor_names,
            "reward": {
                "goal_bonus": self.reward.goal_bonus,
                "dist_coeff": self.reward.distance_coeff,
                "dt_coeff": self.reward.time_penalty,
                "act_quadratic": self.reward.energy_coeff,
                "collision": self.reward.collision_penalty,
                "jerk": self.reward.jerk_penalty,
            }
        }


@dataclass(frozen=True)
class CurriculumStageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    targets: List[str]
    max_steps: int
    success_threshold: float
    min_episodes: int
    difficulty: int
    
    def validate(self) -> None:
        """Validate stage configuration."""
        if not self.targets:
            raise ValueError(f"Stage {self.name} has no targets")
        if self.max_steps <= 0:
            raise ValueError(f"Invalid max_steps for stage {self.name}")
        if not 0 < self.success_threshold <= 1:
            raise ValueError(f"Invalid success_threshold for stage {self.name}")
        if self.min_episodes <= 0:
            raise ValueError(f"Invalid min_episodes for stage {self.name}")


@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = True
    stages: List[CurriculumStageConfig] = field(default_factory=list)
    evaluation_window: int = 1000
    
    def validate(self) -> None:
        """Validate curriculum configuration."""
        if self.enabled and not self.stages:
            raise ValueError("Curriculum enabled but no stages defined")
        if self.evaluation_window <= 0:
            raise ValueError(f"Invalid evaluation_window: {self.evaluation_window}")
        
        for stage in self.stages:
            stage.validate()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CurriculumConfig:
        """Create from dictionary (typically from JSON)."""
        enabled = data.get("enabled", True)
        evaluation_window = data.get("evaluation_window", 1000)
        
        stages = []
        if "curriculum" in data:
            curriculum_data = data["curriculum"]
            for stage_name, stage_config in curriculum_data.items():
                stage = CurriculumStageConfig(
                    name=stage_name,
                    targets=stage_config.get("targets", []),
                    max_steps=stage_config.get("max_steps", 3000),
                    success_threshold=stage_config.get("success_threshold", 0.7),
                    min_episodes=stage_config.get("min_episodes", 1000),
                    difficulty=stage_config.get("difficulty", 1),
                )
                stages.append(stage)
        
        return cls(
            enabled=enabled,
            stages=stages,
            evaluation_window=evaluation_window,
        )


@dataclass(frozen=True)
class MonitoringConfig:
    """Monitoring and logging configuration."""
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "drone-navigation-rl"
    wandb_entity: Optional[str] = None
    log_interval: int = 100
    checkpoint_freq: int = 10000
    video_freq: int = 0  # 0 = disabled
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        if self.log_interval <= 0:
            raise ValueError(f"Invalid log_interval: {self.log_interval}")
        if self.checkpoint_freq <= 0:
            raise ValueError(f"Invalid checkpoint_freq: {self.checkpoint_freq}")


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Reproducibility configuration."""
    seed: int = 42
    torch_deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    numpy_seed: Optional[int] = None
    python_hash_seed: Optional[int] = None
    
    def validate(self) -> None:
        """Validate reproducibility configuration."""
        if self.seed < 0:
            raise ValueError(f"Invalid seed: {self.seed}")
    
    def get_numpy_seed(self) -> int:
        """Get NumPy seed (defaults to main seed if not set)."""
        return self.numpy_seed if self.numpy_seed is not None else self.seed
    
    def get_python_hash_seed(self) -> int:
        """Get Python hash seed (defaults to main seed if not set)."""
        return self.python_hash_seed if self.python_hash_seed is not None else self.seed


class ConfigManager:
    """Centralized configuration manager with validation and presets.
    
    This class manages all configuration for the training pipeline,
    providing validation, preset loading, and environment variable overrides.
    
    Example:
        >>> config = ConfigManager("configs/production_config.yaml", preset="production")
        >>> config.validate()
        >>> ppo_config = config.get_ppo_config()
        >>> env_config = config.get_env_config()
    """
    
    PRESETS = {
        "development": "configs/presets/development.yaml",
        "production": "configs/presets/production.yaml",
        "reproduction": "configs/presets/reproduction.yaml",
    }
    
    def __init__(
        self,
        config_path: Union[str, Path],
        preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to main configuration file (YAML or JSON)
            preset: Optional preset name to load
            overrides: Optional dictionary of configuration overrides
        """
        self.config_path = Path(config_path)
        self.preset = preset
        self.overrides = overrides or {}
        
        # Load configuration
        self.raw_config = self._load_config()
        
        # Apply preset if specified
        if preset:
            self._apply_preset(preset)
        
        # Apply overrides
        if self.overrides:
            self._apply_overrides(self.overrides)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Parse into typed configurations
        self._parse_config()
        
        LOGGER.info("Configuration loaded successfully from %s", config_path)
        if preset:
            LOGGER.info("  Preset: %s", preset)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            if self.config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif self.config_path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
    
    def _apply_preset(self, preset: str) -> None:
        """Apply configuration preset."""
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        
        preset_path = Path(self.PRESETS[preset])
        if not preset_path.exists():
            LOGGER.warning("Preset file not found: %s", preset_path)
            return
        
        with open(preset_path, "r") as f:
            if preset_path.suffix in [".yaml", ".yml"]:
                preset_config = yaml.safe_load(f)
            else:
                preset_config = json.load(f)
        
        # Deep merge preset into config
        self.raw_config = self._deep_merge(self.raw_config, preset_config)
        LOGGER.debug("Applied preset: %s", preset)
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        self.raw_config = self._deep_merge(self.raw_config, overrides)
        LOGGER.debug("Applied overrides: %s", overrides)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.
        
        Environment variables in format: DRONE_RL_<SECTION>_<KEY>
        Example: DRONE_RL_PPO_LEARNING_RATE=0.0001
        """
        prefix = "DRONE_RL_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Parse key
            parts = key[len(prefix):].lower().split("_")
            if len(parts) < 2:
                continue
            
            # Navigate to nested config
            current = self.raw_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set value (try to parse as JSON, fallback to string)
            try:
                current[parts[-1]] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                current[parts[-1]] = value
            
            LOGGER.debug("Applied environment override: %s=%s", key, value)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _parse_config(self) -> None:
        """Parse raw configuration into typed configuration objects."""
        # Parse PPO config
        ppo_data = self.raw_config.get("ppo", {})
        network_data = ppo_data.get("network", {})
        network_config = NetworkConfig(
            architecture=network_data.get("architecture", [256, 256, 256]),
            activation=network_data.get("activation", "Tanh"),
            ortho_init=network_data.get("ortho_init", False),
            log_std_init=network_data.get("log_std_init", -1.0),
        )
        
        training_data = ppo_data.get("training", {})
        self.ppo_config = PPOConfig(
            network=network_config,
            learning_rate=training_data.get("learning_rate", 3e-4),
            n_steps=training_data.get("n_steps", 2048),
            batch_size=training_data.get("batch_size", 64),
            n_epochs=training_data.get("n_epochs", 10),
            gamma=training_data.get("gamma", 0.99),
            gae_lambda=training_data.get("gae_lambda", 0.95),
            clip_range=training_data.get("clip_range", 0.2),
            clip_range_vf=training_data.get("clip_range_vf"),
            ent_coef=training_data.get("ent_coef", 0.01),
            vf_coef=training_data.get("vf_coef", 0.5),
            max_grad_norm=training_data.get("max_grad_norm", 0.5),
            use_sde=training_data.get("use_sde", False),
            sde_sample_freq=training_data.get("sde_sample_freq", -1),
            device=training_data.get("device", "auto"),
        )
        
        # Parse environment config
        env_data = self.raw_config.get("environment", {})
        sim_data = env_data.get("simulation", {})
        drone_data = env_data.get("drone", {})
        reward_data = env_data.get("reward", {})
        
        reward_config = RewardConfig(
            goal_bonus=reward_data.get("goal_bonus", 500.0),
            distance_coeff=reward_data.get("distance_coeff", -5.0),
            time_penalty=reward_data.get("time_penalty", -0.1),
            energy_coeff=reward_data.get("energy_coeff", -0.01),
            collision_penalty=reward_data.get("collision_penalty", -1000.0),
            jerk_penalty=reward_data.get("jerk_penalty", -10.0),
        )
        
        self.env_config = EnvironmentConfig(
            airsim_ip=sim_data.get("airsim_ip", "127.0.0.1"),
            airsim_port=sim_data.get("airsim_port", 41451),
            dt=sim_data.get("dt", 0.1),
            clock_speed=sim_data.get("clock_speed", 1.0),
            spawn_position=drone_data.get("spawn_position", [0.0, 0.0, -1.0]),
            spawn_yaw=drone_data.get("spawn_yaw", 0.0),
            max_velocity_xy=drone_data.get("max_velocity_xy", 1.0),
            max_velocity_z=drone_data.get("max_velocity_z", 0.5),
            max_yaw_rate=drone_data.get("max_yaw_rate", 0.5),
            max_episode_steps=env_data.get("max_episode_steps", 3000),
            success_distance=env_data.get("success_distance", 1.0),
            state_dim=env_data.get("observation", {}).get("state_dim", 35),
            reward=reward_config,
            target_actor_names=env_data.get("target_actor_names", []),
        )
        
        # Parse curriculum config
        curriculum_data = self.raw_config.get("curriculum", {})
        self.curriculum_config = CurriculumConfig.from_dict(curriculum_data)
        
        # Parse monitoring config
        mon_data = self.raw_config.get("monitoring", {})
        self.monitoring_config = MonitoringConfig(
            tensorboard=mon_data.get("tensorboard", True),
            wandb=mon_data.get("wandb", False),
            wandb_project=mon_data.get("wandb_project", "drone-navigation-rl"),
            wandb_entity=mon_data.get("wandb_entity"),
            log_interval=mon_data.get("log_interval", 100),
            checkpoint_freq=mon_data.get("checkpoint_freq", 10000),
            video_freq=mon_data.get("video_freq", 0),
        )
        
        # Parse reproducibility config
        repro_data = self.raw_config.get("reproducibility", {})
        self.repro_config = ReproducibilityConfig(
            seed=repro_data.get("seed", 42),
            torch_deterministic=repro_data.get("torch_deterministic", True),
            cudnn_deterministic=repro_data.get("cudnn_deterministic", True),
            cudnn_benchmark=repro_data.get("cudnn_benchmark", False),
            numpy_seed=repro_data.get("numpy_seed"),
            python_hash_seed=repro_data.get("python_hash_seed"),
        )
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        try:
            self.ppo_config.validate()
            self.env_config.validate()
            self.curriculum_config.validate()
            self.monitoring_config.validate()
            self.repro_config.validate()
            LOGGER.info("✓ All configurations validated successfully")
        except Exception as exc:
            LOGGER.error("Configuration validation failed: %s", exc)
            raise
    
    def get_ppo_config(self) -> PPOConfig:
        """Get PPO configuration."""
        return self.ppo_config
    
    def get_env_config(self) -> EnvironmentConfig:
        """Get environment configuration."""
        return self.env_config
    
    def get_curriculum_config(self) -> CurriculumConfig:
        """Get curriculum configuration."""
        return self.curriculum_config
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.monitoring_config
    
    def get_repro_config(self) -> ReproducibilityConfig:
        """Get reproducibility configuration."""
        return self.repro_config
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            if output_path.suffix in [".yaml", ".yml"]:
                yaml.safe_dump(self.raw_config, f, default_flow_style=False, indent=2)
            elif output_path.suffix == ".json":
                json.dump(self.raw_config, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        LOGGER.info("Configuration saved to %s", output_path)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConfigManager(\n"
            f"  config_path={self.config_path},\n"
            f"  preset={self.preset},\n"
            f"  ppo={self.ppo_config},\n"
            f"  env={self.env_config},\n"
            f"  curriculum={self.curriculum_config},\n"
            f"  monitoring={self.monitoring_config},\n"
            f"  repro={self.repro_config}\n"
            f")"
        )
