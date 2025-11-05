"""Training orchestrator - main training loop coordinator.

Coordinates all training components: environment, model, curriculum,
checkpointing, and monitoring.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager
from core.checkpoint_manager import CheckpointManager
from core.monitoring import MonitoringSystem
from training.curriculum_manager import CurriculumManager
from envs.indoor_drone_env import IndoorDroneEnv
from utils.reproducibility import setup_reproducibility

LOGGER = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates the complete training pipeline.
    
    Responsibilities:
    - Environment setup and configuration
    - Model creation and loading
    - Training loop execution
    - Checkpoint management
    - Curriculum advancement
    - Monitoring and logging
    - Error handling and recovery
    """
    
    def __init__(
        self,
        config: ConfigManager,
        log_dir: Optional[str] = None,
        resume: bool = False,
    ):
        """Initialize training orchestrator.
        
        Args:
            config: Configuration manager
            log_dir: Directory for logs and checkpoints
            resume: Whether to resume from checkpoint
        """
        self.config = config
        self.resume = resume
        
        # Setup log directory
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"logs_production/{timestamp}"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("=" * 70)
        LOGGER.info("TRAINING ORCHESTRATOR INITIALIZED")
        LOGGER.info("  Log directory: %s", self.log_dir)
        LOGGER.info("  Resume: %s", self.resume)
        LOGGER.info("=" * 70)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.log_dir / "checkpoints"),
            keep_n_checkpoints=5,
            save_best=True,
        )
        
        monitoring_config = config.get_monitoring_config()
        self.monitoring = MonitoringSystem(
            log_dir=str(self.log_dir),
            use_tensorboard=monitoring_config.tensorboard,
            use_wandb=monitoring_config.wandb,
            wandb_project=monitoring_config.wandb_project,
            wandb_entity=monitoring_config.wandb_entity,
            wandb_run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Initialize curriculum manager
        self.curriculum_manager: Optional[CurriculumManager] = None
        curriculum_config = config.get_curriculum_config()
        if curriculum_config.enabled:
            # Load curriculum from separate config file
            curriculum_json_path = "configs/curriculum_config.json"
            if Path(curriculum_json_path).exists():
                self.curriculum_manager = CurriculumManager(curriculum_json_path)
                LOGGER.info("Curriculum learning enabled: %d stages", len(self.curriculum_manager.stage_names))
        
        # Will be initialized in setup
        self.env = None
        self.model = None
        self.vec_normalize = None
    
    def setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro_config = self.config.get_repro_config()
        
        setup_reproducibility(
            seed=repro_config.seed,
            torch_deterministic=repro_config.torch_deterministic,
            cudnn_deterministic=repro_config.cudnn_deterministic,
            cudnn_benchmark=repro_config.cudnn_benchmark,
        )
        
        LOGGER.info("✓ Reproducibility configured (seed=%d)", repro_config.seed)
    
    def setup_environment(self) -> gym.Env:
        """Create and configure training environment."""
        LOGGER.info("Setting up environment...")
        
        env_config = self.config.get_env_config()
        
        # Get target actors from curriculum if enabled
        target_actors = []
        if self.curriculum_manager:
            stage = self.curriculum_manager.get_current_stage()
            target_actors = stage.get("targets", [])
            LOGGER.info("  Curriculum Stage: %s", self.curriculum_manager.current_stage_name)
            LOGGER.info("  Targets: %d", len(target_actors))
        
        # Create environment
        def make_env():
            env_kwargs = env_config.to_env_kwargs()
            env_kwargs["target_actor_names"] = target_actors
            return IndoorDroneEnv(**env_kwargs)
        
        env = DummyVecEnv([make_env])
        
        # Wrap with VecNormalize
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        
        self.env = env
        LOGGER.info("✓ Environment created")
        
        return env
    
    def setup_model(self) -> PPO:
        """Create or load PPO model."""
        LOGGER.info("Setting up model...")
        
        ppo_config = self.config.get_ppo_config()
        
        # Try to load checkpoint if resuming
        if self.resume:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                LOGGER.info("  Loading checkpoint: %s", latest_checkpoint)
                model, vec_normalize, metadata = self.checkpoint_manager.load_checkpoint(
                    latest_checkpoint,
                    env=self.env
                )
                
                if vec_normalize:
                    self.vec_normalize = vec_normalize
                    self.env = vec_normalize
                    self.env.training = True
                    self.env.norm_reward = True
                
                self.model = model
                LOGGER.info("✓ Model loaded from checkpoint (step %d)", metadata.get("timestep", 0))
                return model
            else:
                LOGGER.warning("  No checkpoint found, creating new model")
        
        # Create new model
        LOGGER.info("  Creating new PPO model...")
        
        model = PPO(
            policy="MlpPolicy",
            env=self.env,
            policy_kwargs=ppo_config.get_policy_kwargs(),
            **ppo_config.to_sb3_kwargs(),
            verbose=1,
        )
        
        self.model = model
        LOGGER.info("✓ New model created")
        LOGGER.info("  Network: %s", ppo_config.network.architecture)
        LOGGER.info("  Activation: %s", ppo_config.network.activation)
        LOGGER.info("  Learning rate: %s", ppo_config.learning_rate)
        
        return model
    
    def train(self, total_timesteps: int):
        """Execute training loop.
        
        Args:
            total_timesteps: Total number of training timesteps
        """
        LOGGER.info("=" * 70)
        LOGGER.info("STARTING TRAINING")
        LOGGER.info("  Total timesteps: %s", f"{total_timesteps:,}")
        LOGGER.info("=" * 70)
        
        # Setup reproducibility
        self.setup_reproducibility()
        
        # Setup environment
        if self.env is None:
            self.setup_environment()
        
        # Setup model
        if self.model is None:
            self.setup_model()
        
        # Configure logger
        sb3_logger = configure(str(self.log_dir), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(sb3_logger)
        
        # Setup callbacks
        callbacks = self._create_callbacks()
        
        # Training loop with error handling
        try:
            LOGGER.info("Training started...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
            LOGGER.info("✓ Training completed successfully!")
            
        except KeyboardInterrupt:
            LOGGER.warning("Training interrupted by user (Ctrl+C)")
            
        except Exception as e:
            LOGGER.error("Training failed with error: %s", e, exc_info=True)
            raise
            
        finally:
            # Save final model
            self._save_final()
            
            # Cleanup
            self.cleanup()
    
    def _create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        callbacks = []
        
        # Checkpoint callback
        monitoring_config = self.config.get_monitoring_config()
        checkpoint_callback = CheckpointCallback(
            save_freq=monitoring_config.checkpoint_freq,
            save_path=str(self.log_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Curriculum callback if enabled
        if self.curriculum_manager:
            from training.train_ppo_curriculum import CurriculumCallback
            curriculum_callback = CurriculumCallback(
                curriculum_manager=self.curriculum_manager,
                check_freq=1000,
                verbose=1,
            )
            callbacks.append(curriculum_callback)
        
        # Monitoring callback
        monitoring_callback = self._create_monitoring_callback()
        callbacks.append(monitoring_callback)
        
        return CallbackList(callbacks)
    
    def _create_monitoring_callback(self) -> BaseCallback:
        """Create monitoring callback."""
        
        class MonitoringCallback(BaseCallback):
            def __init__(self, monitoring_system: MonitoringSystem, log_interval: int = 100):
                super().__init__()
                self.monitoring = monitoring_system
                self.log_interval = log_interval
                self.episode_count = 0
            
            def _on_step(self) -> bool:
                # Log metrics periodically
                if self.num_timesteps % self.log_interval == 0:
                    # Get metrics from logger
                    metrics = {}
                    if hasattr(self.model, "logger") and self.model.logger:
                        for key in self.model.logger.name_to_value.keys():
                            if key in ["rollout/ep_rew_mean", "rollout/ep_len_mean", "train/loss"]:
                                metrics[key] = self.model.logger.name_to_value[key]
                    
                    if metrics:
                        self.monitoring.log_metrics(metrics, self.num_timesteps)
                        
                        # Health check
                        warnings = self.monitoring.check_training_health(metrics)
                        for warning in warnings:
                            LOGGER.warning("Health check: %s", warning)
                
                return True
        
        monitoring_config = self.config.get_monitoring_config()
        return MonitoringCallback(self.monitoring, log_interval=monitoring_config.log_interval)
    
    def _save_final(self):
        """Save final model and stats."""
        if self.model is None:
            return
        
        LOGGER.info("Saving final model...")
        
        # Get final metrics
        metrics = {}
        if hasattr(self.model, "logger") and self.model.logger:
            for key, value in self.model.logger.name_to_value.items():
                metrics[key] = value
        
        # Save via checkpoint manager
        self.checkpoint_manager.save_final(
            model=self.model,
            vec_normalize=self.vec_normalize if isinstance(self.env, VecNormalize) else None,
            metrics=metrics,
        )
        
        # Save curriculum state if enabled
        if self.curriculum_manager:
            curriculum_state_path = self.log_dir / "curriculum_state_final.json"
            self.curriculum_manager.save_state(str(curriculum_state_path))
            LOGGER.info("Saved curriculum state: %s", curriculum_state_path)
        
        LOGGER.info("✓ Final model saved: %s/final_model", self.log_dir)
    
    def cleanup(self):
        """Cleanup resources."""
        LOGGER.info("Cleaning up...")
        
        # Close environment
        if self.env:
            self.env.close()
        
        # Finish monitoring
        self.monitoring.finish()
        
        LOGGER.info("✓ Cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
