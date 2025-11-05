"""Monitoring system for training metrics.

Simple monitoring with TensorBoard and optional WandB support.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class MonitoringSystem:
    """Training monitoring with TensorBoard and optional WandB.
    
    Features:
    - TensorBoard logging
    - Optional WandB integration
    - Training health checks
    - Curriculum tracking
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """Initialize monitoring system.
        
        Args:
            log_dir: Directory for logs
            use_tensorboard: Enable TensorBoard
            use_wandb: Enable Weights & Biases
            wandb_project: WandB project name
            wandb_entity: WandB entity name
            wandb_run_name: WandB run name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        self.wandb_run = None
        
        # Initialize WandB if enabled
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    dir=str(self.log_dir),
                    sync_tensorboard=True,  # Sync TB logs
                )
                LOGGER.info("WandB initialized: %s/%s", wandb_entity or "personal", wandb_project)
            except ImportError:
                LOGGER.warning("WandB not installed, disabling WandB logging")
                self.use_wandb = False
            except Exception as e:
                LOGGER.warning("Failed to initialize WandB: %s", e)
                self.use_wandb = False
        
        if self.use_tensorboard:
            LOGGER.info("TensorBoard logs: %s", self.log_dir)
            LOGGER.info("  View with: tensorboard --logdir %s", self.log_dir)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to monitoring backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/timestep
        """
        # WandB logging
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                LOGGER.warning("Failed to log to WandB: %s", e)
    
    def log_curriculum_transition(self, from_stage: int, to_stage: int, metrics: Dict[str, Any]):
        """Log curriculum stage transition.
        
        Args:
            from_stage: Previous stage index
            to_stage: New stage index
            metrics: Stage completion metrics
        """
        LOGGER.info("=" * 70)
        LOGGER.info("CURRICULUM STAGE TRANSITION")
        LOGGER.info("  From Stage: %d", from_stage)
        LOGGER.info("  To Stage: %d", to_stage)
        for key, value in metrics.items():
            if isinstance(value, float):
                LOGGER.info("  %s: %.4f", key, value)
            else:
                LOGGER.info("  %s: %s", key, value)
        LOGGER.info("=" * 70)
        
        # Log to WandB
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.log({
                    "curriculum/from_stage": from_stage,
                    "curriculum/to_stage": to_stage,
                    **{f"curriculum_transition/{k}": v for k, v in metrics.items()}
                })
            except Exception as e:
                LOGGER.warning("Failed to log curriculum transition to WandB: %s", e)
    
    def check_training_health(self, metrics: Dict[str, Any]) -> list:
        """Check for training anomalies.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            List of warning messages (empty if all healthy)
        """
        warnings = []
        
        # Check for NaN
        for key, value in metrics.items():
            if isinstance(value, float) and (value != value):  # NaN check
                warnings.append(f"NaN detected in {key}")
        
        # Check for exploding values
        if "loss" in metrics:
            loss = metrics["loss"]
            if isinstance(loss, float) and loss > 1000:
                warnings.append(f"Loss exploding: {loss:.2f}")
        
        # Check for very low success rate
        if "success_rate" in metrics:
            success_rate = metrics["success_rate"]
            if isinstance(success_rate, float) and success_rate < 0.01:
                warnings.append(f"Very low success rate: {success_rate:.2%}")
        
        return warnings
    
    def finish(self):
        """Finish monitoring and cleanup."""
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.finish()
                LOGGER.info("WandB run finished")
            except Exception as e:
                LOGGER.warning("Failed to finish WandB run: %s", e)
