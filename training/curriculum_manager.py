"""Curriculum Manager for Progressive RL Training.

This module manages curriculum learning stages for drone navigation training,
automatically advancing through difficulty levels based on performance metrics.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

LOGGER = logging.getLogger(__name__)


class CurriculumManager:
    """Manages curriculum learning stages for drone training.
    
    The curriculum manager tracks training progress across different stages,
    monitors success rates, and determines when to advance to more difficult tasks.
    
    Attributes:
        config: Full curriculum configuration dictionary
        stages: Dictionary of stage configurations
        stage_names: Ordered list of stage names
        current_stage_idx: Index of current stage
        current_stage_name: Name of current stage
        episode_rewards: List of recent episode rewards
        episode_success: List of recent episode success flags
        episodes_in_stage: Number of episodes completed in current stage
    """
    
    def __init__(self, config_path: str = "configs/curriculum_config.json"):
        """Initialize curriculum manager from config file.
        
        Args:
            config_path: Path to curriculum configuration JSON file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is malformed
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Curriculum config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        if "curriculum" not in self.config:
            raise ValueError("Config must contain 'curriculum' key")
        
        self.stages = self.config["curriculum"]
        self.stage_names = list(self.stages.keys())
        
        if len(self.stage_names) == 0:
            raise ValueError("Curriculum must have at least one stage")
        
        self.current_stage_idx = 0
        self.current_stage_name = self.stage_names[0]
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_success: List[float] = []
        self.episodes_in_stage = 0
        
        LOGGER.info("Curriculum initialized with %d stages", len(self.stage_names))
        LOGGER.info("   Starting stage: %s", self.current_stage_name)
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current stage configuration.
        
        Returns:
            Dictionary containing current stage configuration
        """
        return self.stages[self.current_stage_name]
    
    def get_target_actors(self, all_actors: List[str]) -> List[str]:
        """Get target actors for current stage.
        
        Args:
            all_actors: List of all available actor names in the environment
            
        Returns:
            Filtered list of target actors for current stage
        """
        stage = self.get_current_stage()
        targets = stage["targets"]
        
        if targets == "ALL":
            # Filter only landing pads
            landing_actors = [a for a in all_actors if a.startswith("Landing_")]
            LOGGER.debug("Using ALL landing targets: %d actors", len(landing_actors))
            return landing_actors
        elif isinstance(targets, list):
            # Validate that requested targets exist
            available_set = set(all_actors)
            valid_targets = [t for t in targets if t in available_set]
            
            if len(valid_targets) < len(targets):
                missing = set(targets) - set(valid_targets)
                LOGGER.warning("Some target actors not found in environment: %s", missing)
            
            return valid_targets
        else:
            LOGGER.error("Invalid targets type in stage config: %s", type(targets))
            return all_actors
    
    def record_episode(self, reward: float, success: bool):
        """Record episode metrics.
        
        Args:
            reward: Total reward for the episode
            success: Whether the episode was successful (goal reached)
        """
        self.episode_rewards.append(float(reward))
        self.episode_success.append(float(success))
        self.episodes_in_stage += 1
    
    def should_advance(self) -> bool:
        """Check if should advance to next stage.
        
        Advancement criteria:
        1. Minimum number of episodes completed
        2. Success rate exceeds threshold over recent window
        
        Returns:
            True if should advance to next stage, False otherwise
        """
        stage = self.get_current_stage()
        
        # Check minimum episodes requirement
        if self.episodes_in_stage < stage["min_episodes"]:
            return False
        
        # Calculate success rate over recent window
        recent_window = min(1000, len(self.episode_success))
        if recent_window < 100:
            # Need at least 100 episodes to make reliable decision
            return False
        
        recent_success = self.episode_success[-recent_window:]
        success_rate = np.mean(recent_success)
        
        threshold = stage["success_threshold"]
        
        if success_rate >= threshold:
            LOGGER.info("Stage '%s' completed", self.current_stage_name)
            LOGGER.info("   Episodes: %d", self.episodes_in_stage)
            LOGGER.info("   Success rate: %.2f%% (threshold: %.2f%%)", 
                       success_rate * 100, threshold * 100)
            LOGGER.info("   Avg reward (last %d): %.2f", 
                       recent_window, np.mean(self.episode_rewards[-recent_window:]))
            return True
        
        return False
    
    def advance_stage(self) -> bool:
        """Advance to next stage.
        
        Returns:
            True if successfully advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stage_names) - 1:
            LOGGER.info("Already at final stage")
            return False
        
        # Reset metrics for new stage
        self.episode_rewards = []
        self.episode_success = []
        self.episodes_in_stage = 0
        
        # Advance
        self.current_stage_idx += 1
        self.current_stage_name = self.stage_names[self.current_stage_idx]
        
        stage = self.get_current_stage()
        LOGGER.info("="*70)
        LOGGER.info("Advanced to stage %d/%d: %s", 
                   self.current_stage_idx + 1, len(self.stage_names), self.current_stage_name)
        LOGGER.info("   Description: %s", stage['description'])
        LOGGER.info("   Targets: %s", 
                   len(stage['targets']) if stage['targets'] != 'ALL' else 'ALL')
        LOGGER.info("   Max steps: %d", stage['max_steps'])
        LOGGER.info("   Success threshold: %.2f%%", stage['success_threshold'] * 100)
        LOGGER.info("   Min episodes: %d", stage['min_episodes'])
        LOGGER.info("="*70)
        
        return True
    
    def get_stage_info(self) -> Dict[str, Any]:
        """Get current stage info for logging.
        
        Returns:
            Dictionary with current stage metrics and progress
        """
        stage = self.get_current_stage()
        recent_window = min(100, len(self.episode_success))
        
        info = {
            "stage_name": self.current_stage_name,
            "stage_idx": self.current_stage_idx + 1,
            "total_stages": len(self.stage_names),
            "episodes_in_stage": self.episodes_in_stage,
            "min_episodes": stage["min_episodes"],
            "progress_pct": (self.episodes_in_stage / stage["min_episodes"]) * 100,
            "success_rate_recent": np.mean(self.episode_success[-recent_window:]) if recent_window > 0 else 0.0,
            "threshold": stage["success_threshold"],
            "avg_reward_recent": np.mean(self.episode_rewards[-recent_window:]) if recent_window > 0 else 0.0,
            "max_steps": stage["max_steps"],
        }
        
        return info
    
    def save_state(self, path: str):
        """Save curriculum state to file.
        
        Args:
            path: Path to save state JSON file
        """
        state = {
            "current_stage_idx": self.current_stage_idx,
            "current_stage_name": self.current_stage_name,
            "episodes_in_stage": self.episodes_in_stage,
            "episode_rewards": self.episode_rewards[-1000:],  # Keep last 1000
            "episode_success": self.episode_success[-1000:],
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        LOGGER.info("Curriculum state saved to %s", path)
    
    def load_state(self, path: str) -> bool:
        """Load curriculum state from file.
        
        Args:
            path: Path to state JSON file
            
        Returns:
            True if loaded successfully, False if file not found
        """
        if not Path(path).exists():
            LOGGER.warning("Curriculum state file not found: %s", path)
            return False
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.current_stage_idx = state["current_stage_idx"]
            self.current_stage_name = state["current_stage_name"]
            self.episodes_in_stage = state["episodes_in_stage"]
            self.episode_rewards = state["episode_rewards"]
            self.episode_success = state["episode_success"]
            
            LOGGER.info("Curriculum state loaded from %s", path)
            LOGGER.info("   Resuming at stage: %s", self.current_stage_name)
            LOGGER.info("   Episodes in stage: %d", self.episodes_in_stage)
            return True
        except Exception as exc:
            LOGGER.error("Failed to load curriculum state: %s", exc)
            return False
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary.
        
        Returns:
            Formatted string with curriculum progress information
        """
        info = self.get_stage_info()
        
        summary = []
        summary.append("="*70)
        summary.append("CURRICULUM PROGRESS")
        summary.append("="*70)
        summary.append("Stage: %d/%d - %s" % (info['stage_idx'], info['total_stages'], info['stage_name']))
        summary.append("Episodes: %d/%d (%.1f%%)" % (info['episodes_in_stage'], info['min_episodes'], info['progress_pct']))
        summary.append("Recent Success Rate: %.2f%% (target: %.2f%%)" % (info['success_rate_recent'] * 100, info['threshold'] * 100))
        summary.append("Recent Avg Reward: %.2f" % info['avg_reward_recent'])
        summary.append("="*70)
        
        return "\n".join(summary)
