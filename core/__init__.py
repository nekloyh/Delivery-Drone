"""Core package for production-ready RL training pipeline.

This package provides the essential components for a robust, scalable,
and reproducible reinforcement learning training system.

Modules:
    config_manager: Centralized configuration with validation
    training_orchestrator: Main training loop orchestration
    checkpoint_manager: Intelligent checkpoint management
    monitoring: Real-time monitoring and observability
"""

__version__ = "3.0.0"
__author__ = "Senior AI Engineering Team"

from .config_manager import ConfigManager, PPOConfig, EnvironmentConfig, CurriculumConfig
from .training_orchestrator import TrainingOrchestrator
from .checkpoint_manager import CheckpointManager
from .monitoring import MonitoringSystem

__all__ = [
    "ConfigManager",
    "PPOConfig",
    "EnvironmentConfig",
    "CurriculumConfig",
    "TrainingOrchestrator",
    "CheckpointManager",
    "MonitoringSystem",
]
