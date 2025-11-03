"""PPO training with curriculum learning.

10-stage curriculum: single target → all 36 targets
Auto-advances when success threshold met

Usage:
    python training/train_ppo_curriculum.py
    python training/train_ppo_curriculum.py --resume
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.indoor_drone_env import IndoorDroneEnv
from training.curriculum_manager import CurriculumManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """Callback to handle curriculum stage advancement during training.
    
    This callback:
    - Tracks episode outcomes (rewards, success)
    - Checks if current stage is mastered
    - Automatically advances to next stage when criteria met
    - Updates environment with new target set
    - Saves checkpoints at stage transitions
    """
    
    def __init__(
        self, 
        curriculum_manager: CurriculumManager, 
        check_freq: int = 1000,
        verbose: int = 0
    ):
        """Initialize curriculum callback.
        
        Args:
            curriculum_manager: CurriculumManager instance
            check_freq: Check for stage advancement every N episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.check_freq = check_freq
        self.episode_count = 0
        self.last_log_step = 0
    
    def _on_step(self) -> bool:
        """Called at each environment step.
        
        Returns:
            True to continue training, False to stop
        """
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            # Get episode info
            infos = self.locals.get("infos", [{}])
            if len(infos) > 0:
                info = infos[0]
                
                # Extract episode metrics
                episode_info = info.get("episode", {})
                episode_reward = episode_info.get("r", 0.0)
                episode_success = info.get("success", False)
                
                # Record episode
                self.curriculum.record_episode(episode_reward, episode_success)
                self.episode_count += 1
                
                # Log episode info
                if self.verbose > 0:
                    LOGGER.info(
                        f"Episode {self.episode_count}: "
                        f"reward={episode_reward:.2f}, "
                        f"success={episode_success}, "
                        f"dist={info.get('distance_to_goal', 0):.2f}m"
                    )
                
                # Check for stage advancement periodically
                if self.episode_count % self.check_freq == 0:
                    if self.curriculum.should_advance():
                        advanced = self.curriculum.advance_stage()
                        if advanced:
                            # Update env with new targets and max_steps
                            self._update_env_config()
                            
                            # Save curriculum state
                            state_path = f"{self.model.logger.dir}/curriculum_state.json"
                            self.curriculum.save_state(state_path)
                            
                            # Save model checkpoint at stage transition
                            model_path = f"{self.model.logger.dir}/stage_{self.curriculum.current_stage_idx}_model"
                            self.model.save(model_path)
                            LOGGER.info("Saved stage checkpoint: %s", model_path)
                            
                            # Save VecNormalize stats
                            if isinstance(self.training_env, VecNormalize):
                                norm_path = f"{self.model.logger.dir}/stage_{self.curriculum.current_stage_idx}_vecnorm.pkl"
                                self.training_env.save(norm_path)
                    
                    # Log curriculum progress
                    stage_info = self.curriculum.get_stage_info()
                    for key, value in stage_info.items():
                        self.logger.record(f"curriculum/{key}", value)
                    
                    if self.verbose > 0:
                        LOGGER.info(self.curriculum.get_progress_summary())
        
        # Periodic logging (every 10k steps)
        if self.num_timesteps - self.last_log_step >= 10000:
            stage_info = self.curriculum.get_stage_info()
            LOGGER.info(
                f"Step {self.num_timesteps}: "
                f"Stage {stage_info['stage_idx']}/{stage_info['total_stages']} "
                f"({stage_info['stage_name']}), "
                f"Success: {stage_info['success_rate_recent']:.2%}"
            )
            self.last_log_step = self.num_timesteps
        
        return True
    
    def _update_env_config(self):
        """Update environment configuration for new curriculum stage."""
        env = self.training_env.envs[0]
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        
        # Get all available actors from environment
        all_actors = unwrapped_env.target_actor_names
        
        # Get filtered targets for current stage
        new_targets = self.curriculum.get_target_actors(all_actors)
        
        # Update environment targets
        unwrapped_env.target_actor_names = new_targets
        
        # Update max_steps from curriculum
        stage = self.curriculum.get_current_stage()
        unwrapped_env.max_steps = stage["max_steps"]
        
        LOGGER.info("Configured env: %d targets, max_steps=%d", len(new_targets), stage["max_steps"])


def make_env(env_config: dict, curriculum: CurriculumManager):
    """Create environment factory with curriculum-filtered targets.
    
    Args:
        env_config: Environment configuration dictionary
        curriculum: CurriculumManager instance
        
    Returns:
        Callable that creates a new environment instance
    """
    def _init():
        env = IndoorDroneEnv(**env_config)
        
        # Store original target list
        original_targets = env.target_actor_names.copy()
        
        # Filter targets based on current curriculum stage
        filtered_targets = curriculum.get_target_actors(original_targets)
        env.target_actor_names = filtered_targets
        
        # Update max_steps from curriculum
        stage = curriculum.get_current_stage()
        env.max_steps = stage.get("max_steps", 1000)
        
        LOGGER.info(
            "Created env for stage %s: %d targets, max_steps=%d",
            curriculum.current_stage_name, len(filtered_targets), env.max_steps
        )
        
        return env
    
    return _init


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO with curriculum learning")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/fixed_config.json",
        help="Path to environment config"
    )
    parser.add_argument(
        "--ppo-config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to PPO config"
    )
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default="configs/curriculum_config.json",
        help="Path to curriculum config"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps"
    )
    args = parser.parse_args()
    
    # Load configs
    LOGGER.info("Loading configurations...")
    
    with open(args.ppo_config, "r") as f:
        ppo_config = yaml.safe_load(f)
    
    with open(args.env_config, "r") as f:
        env_config = json.load(f)
    
    # Initialize curriculum
    curriculum = CurriculumManager(args.curriculum_config)
    
    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_curriculum/{timestamp}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Try to load previous curriculum state if resuming
    if args.resume:
        curriculum_state_path = "logs_curriculum/latest/curriculum_state.json"
        if os.path.exists(curriculum_state_path):
            curriculum.load_state(curriculum_state_path)
            LOGGER.info("Resumed curriculum from: %s", curriculum_state_path)
        else:
            LOGGER.warning("Resume flag set but no previous state found, starting fresh")
    
    # Create environment
    LOGGER.info("Creating training environment...")
    env = DummyVecEnv([make_env(env_config, curriculum)])
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Create or load model
    model_path = "logs_curriculum/latest/final_model.zip"
    vecnorm_path = "logs_curriculum/latest/vecnorm_final.pkl"
    
    if args.resume and os.path.exists(model_path):
        LOGGER.info("Loading existing model from %s", model_path)
        model = PPO.load(model_path, env=env)
        
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True
            LOGGER.info("Loaded VecNormalize stats")
    else:
        LOGGER.info("Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            **ppo_config.get("ppo_params", {}),
            verbose=1,
            tensorboard_log=log_dir
        )
    
    # Setup logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    LOGGER.info("Training started")
    LOGGER.info("Timesteps: %d", args.timesteps)
    LOGGER.info("Stage: %s (%d/%d)", curriculum.current_stage_name, 1, len(curriculum.stage_names))
    LOGGER.info("Log dir: %s", log_dir)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="ppo_curriculum",
        save_vecnormalize=True
    )
    
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum,
        check_freq=1000,
        verbose=1
    )
    
    # Training
    try:
        LOGGER.info("Starting training...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user")
    finally:
        # Save final model
        LOGGER.info("Saving final model...")
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vecnorm_final.pkl")
        curriculum.save_state(f"{log_dir}/curriculum_state_final.json")
        
        # Create symlink to latest
        latest_dir = Path("logs_curriculum/latest")
        if latest_dir.exists() or latest_dir.is_symlink():
            latest_dir.unlink()
        try:
            latest_dir.symlink_to(Path(log_dir).absolute(), target_is_directory=True)
            LOGGER.info("Created symlink: logs_curriculum/latest -> %s", log_dir)
        except Exception as exc:
            LOGGER.warning("Failed to create symlink: %s", exc)
        
        LOGGER.info("Training completed. Model saved to %s", log_dir)
        LOGGER.info(curriculum.get_progress_summary())
    
    env.close()


if __name__ == "__main__":
    main()
