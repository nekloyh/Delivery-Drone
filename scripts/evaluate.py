"""Model evaluation script.

Evaluates trained models and generates comprehensive metrics.

Usage:
    # Evaluate final model
    python scripts/evaluate.py --checkpoint logs_production/20251106_120000/final_model
    
    # Evaluate with specific number of episodes
    python scripts/evaluate.py --checkpoint path/to/model --episodes 100
    
    # Evaluate all stages
    python scripts/evaluate.py --checkpoint path/to/model --all-stages
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.indoor_drone_env import IndoorDroneEnv
from core.checkpoint_manager import CheckpointManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained RL models."""
    
    def __init__(self, checkpoint_path: str):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        LOGGER.info("Loading checkpoint: %s", self.checkpoint_path.name)
        checkpoint_manager = CheckpointManager(checkpoint_dir=str(self.checkpoint_path.parent))
        
        # Create dummy env for loading
        dummy_env = DummyVecEnv([lambda: IndoorDroneEnv()])
        
        self.model, self.vec_normalize, self.metadata = checkpoint_manager.load_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            env=dummy_env
        )
        
        dummy_env.close()
        
        LOGGER.info("✓ Model loaded successfully")
        if self.metadata:
            LOGGER.info("  Checkpoint info:")
            LOGGER.info("    Timestep: %s", self.metadata.get("timestep", "unknown"))
            LOGGER.info("    Stage: %s", self.metadata.get("stage", "unknown"))
    
    def evaluate(
        self,
        n_episodes: int = 100,
        deterministic: bool = True,
        target_actors: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            n_episodes: Number of episodes to run
            deterministic: Use deterministic policy
            target_actors: List of target actor names (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        LOGGER.info("=" * 70)
        LOGGER.info("EVALUATION")
        LOGGER.info("  Episodes: %d", n_episodes)
        LOGGER.info("  Deterministic: %s", deterministic)
        if target_actors:
            LOGGER.info("  Targets: %d", len(target_actors))
        LOGGER.info("=" * 70)
        
        # Create evaluation environment
        def make_eval_env():
            env_kwargs = {}
            if target_actors:
                env_kwargs["target_actor_names"] = target_actors
            return IndoorDroneEnv(**env_kwargs)
        
        eval_env = DummyVecEnv([make_eval_env])
        
        # Apply normalization if available
        if self.vec_normalize:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,  # Don't normalize rewards during evaluation
                clip_obs=10.0,
                training=False,  # Set to evaluation mode
            )
            # Load normalization stats
            if hasattr(self.vec_normalize, "obs_rms"):
                eval_env.obs_rms = self.vec_normalize.obs_rms
            if hasattr(self.vec_normalize, "ret_rms"):
                eval_env.ret_rms = self.vec_normalize.ret_rms
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        collision_count = 0
        distances_to_goal = []
        
        # Run episodes
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if done:
                    # Extract episode info
                    if len(info) > 0:
                        ep_info = info[0]
                        
                        if ep_info.get("success", False):
                            success_count += 1
                        
                        if ep_info.get("collision", False):
                            collision_count += 1
                        
                        dist = ep_info.get("distance_to_goal", 0.0)
                        distances_to_goal.append(dist)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                LOGGER.info("  Episode %d/%d: reward=%.2f, length=%d", 
                           episode + 1, n_episodes, episode_reward, episode_length)
        
        eval_env.close()
        
        # Compute statistics
        results = {
            "n_episodes": n_episodes,
            "deterministic": deterministic,
            
            # Rewards
            "reward_mean": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "reward_min": float(np.min(episode_rewards)),
            "reward_max": float(np.max(episode_rewards)),
            
            # Episode lengths
            "length_mean": float(np.mean(episode_lengths)),
            "length_std": float(np.std(episode_lengths)),
            
            # Success metrics
            "success_rate": success_count / n_episodes,
            "success_count": success_count,
            
            # Failure metrics
            "collision_rate": collision_count / n_episodes,
            "collision_count": collision_count,
            
            # Distance to goal
            "distance_mean": float(np.mean(distances_to_goal)) if distances_to_goal else 0.0,
            "distance_std": float(np.std(distances_to_goal)) if distances_to_goal else 0.0,
        }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print evaluation results."""
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("EVALUATION RESULTS")
        LOGGER.info("=" * 70)
        LOGGER.info("")
        
        LOGGER.info("Performance Metrics:")
        LOGGER.info("  Success Rate:     %.2f%% (%d/%d episodes)", 
                   results["success_rate"] * 100,
                   results["success_count"],
                   results["n_episodes"])
        LOGGER.info("  Collision Rate:   %.2f%% (%d/%d episodes)", 
                   results["collision_rate"] * 100,
                   results["collision_count"],
                   results["n_episodes"])
        LOGGER.info("")
        
        LOGGER.info("Reward Statistics:")
        LOGGER.info("  Mean:   %.2f ± %.2f", results["reward_mean"], results["reward_std"])
        LOGGER.info("  Min:    %.2f", results["reward_min"])
        LOGGER.info("  Max:    %.2f", results["reward_max"])
        LOGGER.info("")
        
        LOGGER.info("Episode Length:")
        LOGGER.info("  Mean:   %.1f ± %.1f steps", results["length_mean"], results["length_std"])
        LOGGER.info("")
        
        LOGGER.info("Distance to Goal:")
        LOGGER.info("  Mean:   %.2f ± %.2f meters", results["distance_mean"], results["distance_std"])
        LOGGER.info("")
        LOGGER.info("=" * 70)
    
    def evaluate_all_stages(self, n_episodes: int = 50):
        """Evaluate on all curriculum stages.
        
        Args:
            n_episodes: Episodes per stage
            
        Returns:
            Dictionary of stage -> results
        """
        # Load curriculum config
        import json
        curriculum_path = Path("configs/curriculum_config.json")
        
        if not curriculum_path.exists():
            LOGGER.error("Curriculum config not found: %s", curriculum_path)
            return {}
        
        with open(curriculum_path, "r") as f:
            curriculum_data = json.load(f)
        
        stages = curriculum_data.get("curriculum", {})
        
        LOGGER.info("Evaluating on %d curriculum stages...", len(stages))
        
        all_results = {}
        
        for stage_name, stage_config in stages.items():
            targets = stage_config.get("targets", [])
            
            LOGGER.info("")
            LOGGER.info("Evaluating stage: %s (%d targets)", stage_name, len(targets))
            
            results = self.evaluate(
                n_episodes=n_episodes,
                deterministic=True,
                target_actors=targets
            )
            
            results["stage_name"] = stage_name
            results["n_targets"] = len(targets)
            all_results[stage_name] = results
        
        # Print summary
        self._print_stage_summary(all_results)
        
        return all_results
    
    def _print_stage_summary(self, all_results: Dict[str, Dict]):
        """Print summary of all stages."""
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("STAGE-WISE SUMMARY")
        LOGGER.info("=" * 70)
        LOGGER.info("")
        LOGGER.info("%-30s %15s %15s", "Stage", "Success Rate", "Avg Reward")
        LOGGER.info("-" * 70)
        
        for stage_name, results in all_results.items():
            LOGGER.info("%-30s %14.1f%% %15.2f",
                       stage_name,
                       results["success_rate"] * 100,
                       results["reward_mean"])
        
        LOGGER.info("=" * 70)
        
        # Overall statistics
        all_success_rates = [r["success_rate"] for r in all_results.values()]
        overall_success = np.mean(all_success_rates)
        
        LOGGER.info("")
        LOGGER.info("Overall Success Rate: %.2f%%", overall_success * 100)
        LOGGER.info("=" * 70)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained drone navigation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Evaluate on all curriculum stages"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Print header
    LOGGER.info("=" * 70)
    LOGGER.info("MODEL EVALUATION")
    LOGGER.info("=" * 70)
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(checkpoint_path=args.checkpoint)
        
        # Run evaluation
        if args.all_stages:
            results = evaluator.evaluate_all_stages(n_episodes=args.episodes)
        else:
            results = evaluator.evaluate(
                n_episodes=args.episodes,
                deterministic=args.deterministic
            )
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            LOGGER.info("")
            LOGGER.info("✓ Results saved to: %s", output_path)
        
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("✓ EVALUATION COMPLETED SUCCESSFULLY!")
        LOGGER.info("=" * 70)
        
    except Exception as e:
        LOGGER.error("Evaluation failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
