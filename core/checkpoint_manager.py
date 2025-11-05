"""Checkpoint management for training pipeline.

Manages model checkpoints with intelligent saving, loading, and cleanup.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

LOGGER = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and versioning.
    
    Features:
    - Automatic checkpoint saving at intervals
    - Stage-based checkpoints for curriculum learning
    - Best model tracking
    - Automatic cleanup of old checkpoints
    - Checkpoint metadata (metrics, timestamp, etc.)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_n_checkpoints: int = 5,
        save_best: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_checkpoints: Number of regular checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_best = save_best
        
        self.best_reward = -float('inf')
        self.checkpoint_history: List[Dict[str, Any]] = []
        
        LOGGER.info("CheckpointManager initialized: %s", self.checkpoint_dir)
    
    def save_checkpoint(
        self,
        model: PPO,
        vec_normalize: Optional[VecNormalize],
        timestep: int,
        stage: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint with metadata.
        
        Args:
            model: PPO model to save
            vec_normalize: VecNormalize wrapper (optional)
            timestep: Current training timestep
            stage: Current curriculum stage (optional)
            metrics: Training metrics (optional)
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        
        # Create checkpoint name
        if stage is not None:
            name = f"stage_{stage}_step_{timestep}"
        else:
            name = f"step_{timestep}"
        
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = checkpoint_path / "model.zip"
        model.save(str(model_file))
        
        # Save VecNormalize stats
        if vec_normalize is not None:
            vecnorm_file = checkpoint_path / "vecnormalize.pkl"
            vec_normalize.save(str(vecnorm_file))
        
        # Save metadata
        metadata = {
            "timestep": timestep,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "is_best": is_best,
        }
        
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Track checkpoint
        self.checkpoint_history.append({
            "path": str(checkpoint_path),
            "timestep": timestep,
            "stage": stage,
            "metrics": metrics,
        })
        
        LOGGER.info("Saved checkpoint: %s", checkpoint_path.name)
        
        # Save as best if applicable
        if self.save_best:
            avg_reward = metrics.get("episode_reward_mean", -float('inf'))
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self._save_as_best(checkpoint_path)
        
        # Cleanup old checkpoints
        if stage is None:  # Only cleanup regular checkpoints, not stage checkpoints
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _save_as_best(self, source_path: Path):
        """Save checkpoint as best model."""
        best_path = self.checkpoint_dir / "best_model"
        
        # Remove old best if exists
        if best_path.exists():
            shutil.rmtree(best_path)
        
        # Copy checkpoint as best
        shutil.copytree(source_path, best_path)
        
        LOGGER.info("✓ New best model! Reward: %.2f", self.best_reward)
    
    def _cleanup_old_checkpoints(self):
        """Remove old regular checkpoints, keeping only the most recent N."""
        # Get all regular checkpoints (not stage or best)
        regular_checkpoints = []
        for cp in self.checkpoint_history:
            if cp["stage"] is None:
                regular_checkpoints.append(cp)
        
        # Sort by timestep
        regular_checkpoints.sort(key=lambda x: x["timestep"])
        
        # Remove old ones
        while len(regular_checkpoints) > self.keep_n_checkpoints:
            old_checkpoint = regular_checkpoints.pop(0)
            old_path = Path(old_checkpoint["path"])
            
            if old_path.exists():
                shutil.rmtree(old_path)
                LOGGER.debug("Removed old checkpoint: %s", old_path.name)
            
            # Remove from history
            self.checkpoint_history.remove(old_checkpoint)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        env = None,
    ) -> tuple:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            env: Environment for model (optional)
            
        Returns:
            Tuple of (model, vec_normalize, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        model_file = checkpoint_path / "model.zip"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = PPO.load(str(model_file), env=env)
        LOGGER.info("Loaded model from: %s", checkpoint_path.name)
        
        # Load VecNormalize
        vecnorm_file = checkpoint_path / "vecnormalize.pkl"
        vec_normalize = None
        if vecnorm_file.exists():
            vec_normalize = VecNormalize.load(str(vecnorm_file), env)
            LOGGER.info("Loaded VecNormalize stats")
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        return model, vec_normalize, metadata
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        if not self.checkpoint_history:
            return None
        
        latest = max(self.checkpoint_history, key=lambda x: x["timestep"])
        return latest["path"]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_model"
        if best_path.exists():
            return str(best_path)
        return None
    
    def get_stage_checkpoint(self, stage: int) -> Optional[str]:
        """Get checkpoint for specific curriculum stage."""
        for cp in self.checkpoint_history:
            if cp["stage"] == stage:
                return cp["path"]
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        return self.checkpoint_history.copy()
    
    def save_final(
        self,
        model: PPO,
        vec_normalize: Optional[VecNormalize],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Save final model after training completion."""
        final_path = self.checkpoint_dir / "final_model"
        final_path.mkdir(exist_ok=True)
        
        # Save model
        model.save(str(final_path / "model.zip"))
        
        # Save VecNormalize
        if vec_normalize is not None:
            vec_normalize.save(str(final_path / "vecnormalize.pkl"))
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "is_final": True,
        }
        
        with open(final_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        LOGGER.info("✓ Final model saved: %s", final_path)
