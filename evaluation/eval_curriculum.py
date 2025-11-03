"""Evaluate trained model across curriculum stages.

Usage:
    python evaluation/eval_curriculum.py \
        --model logs_curriculum/latest/final_model.zip \
        --vecnorm logs_curriculum/latest/vecnorm_final.pkl
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


def evaluate_stage(
    model,
    env,
    curriculum: CurriculumManager,
    stage_name: str,
    n_episodes: int = 50,
    deterministic: bool = True
) -> Dict:
    """Evaluate model on specific curriculum stage.
    
    Args:
        model: Trained PPO model
        env: Vectorized environment
        curriculum: CurriculumManager instance
        stage_name: Name of stage to evaluate
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Update curriculum to target stage
    curriculum.current_stage_name = stage_name
    curriculum.current_stage_idx = curriculum.stage_names.index(stage_name)
    stage = curriculum.get_current_stage()
    
    # Update env configuration
    unwrapped_env = env.envs[0].unwrapped if hasattr(env.envs[0], 'unwrapped') else env.envs[0]
    
    # Get all available actors
    all_actors = unwrapped_env.target_actor_names_original
    
    # Filter targets for this stage
    filtered_targets = curriculum.get_target_actors(all_actors)
    unwrapped_env.target_actor_names = filtered_targets
    unwrapped_env.max_steps = stage["max_steps"]
    
    LOGGER.info("="*70)
    LOGGER.info("Evaluating stage: %s", stage_name)
    LOGGER.info("   Description: %s", stage['description'])
    LOGGER.info("   Targets: %d", len(filtered_targets))
    LOGGER.info("   Max steps: %d", stage['max_steps'])
    LOGGER.info("="*70)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episode_distances = []
    collision_count = 0
    timeout_count = 0
    battery_depleted_count = 0
    total_energy = 0.0
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        ep_energy = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            ep_energy += info[0].get("energy", 0)
        
        info = info[0]
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_successes.append(float(info.get("success", False)))
        episode_distances.append(info.get("distance_to_goal", float('inf')))
        total_energy += ep_energy
        
        if info.get("collision", False):
            collision_count += 1
        if info.get("timeout", False):
            timeout_count += 1
        if info.get("battery_depleted", False):
            battery_depleted_count += 1
        
        if (ep + 1) % 10 == 0:
            recent_success = np.mean(episode_successes[-10:])
            LOGGER.info(
                f"  Episode {ep+1}/{n_episodes} - "
                f"Reward: {ep_reward:.2f}, "
                f"Success: {info.get('success', False)}, "
                f"Dist: {info.get('distance_to_goal', 0):.2f}m, "
                f"Recent success rate: {recent_success:.2%}"
            )
    
    # Calculate statistics
    results = {
        "stage_name": stage_name,
        "description": stage['description'],
        "n_episodes": n_episodes,
        "success_rate": np.mean(episode_successes),
        "success_std": np.std(episode_successes),
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "median_reward": np.median(episode_rewards),
        "avg_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "avg_final_distance": np.mean(episode_distances),
        "collision_rate": collision_count / n_episodes,
        "timeout_rate": timeout_count / n_episodes,
        "battery_depleted_rate": battery_depleted_count / n_episodes,
        "avg_energy_per_episode": total_energy / n_episodes,
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "success_count": int(np.sum(episode_successes)),
    }
    
    # Print results
    LOGGER.info("Results for %s:", stage_name)
    LOGGER.info("   Success Rate: %.2f%% ± %.2f%%", 
                results['success_rate'] * 100, results['success_std'] * 100)
    LOGGER.info("   Avg Reward: %.2f ± %.2f", 
                results['avg_reward'], results['std_reward'])
    LOGGER.info("   Median Reward: %.2f", results['median_reward'])
    LOGGER.info("   Avg Episode Length: %d steps", int(results['avg_length']))
    LOGGER.info("   Avg Final Distance: %.2fm", results['avg_final_distance'])
    LOGGER.info("   Collision Rate: %.2f%%", results['collision_rate'] * 100)
    LOGGER.info("   Timeout Rate: %.2f%%", results['timeout_rate'] * 100)
    LOGGER.info("   Battery Depleted Rate: %.2f%%", results['battery_depleted_rate'] * 100)
    LOGGER.info("   Avg Energy: %.4f", results['avg_energy_per_episode'])
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate PPO model on curriculum stages")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        required=True,
        help="Path to VecNormalize stats (.pkl)"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/fixed_config.json",
        help="Path to environment config"
    )
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default="configs/curriculum_config.json",
        help="Path to curriculum config"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per stage"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/curriculum_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=None,
        help="Specific stages to evaluate (default: all)"
    )
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        LOGGER.error("Model file not found: %s", args.model)
        return 1
    
    if not Path(args.vecnorm).exists():
        LOGGER.error("VecNormalize file not found: %s", args.vecnorm)
        return 1
    
    # Load env config
    LOGGER.info("Loading configurations...")
    with open(args.env_config, 'r') as f:
        env_config = json.load(f)
    
    # Load curriculum
    curriculum = CurriculumManager(args.curriculum_config)
    
    # Determine which stages to evaluate
    stages_to_eval = args.stages if args.stages else curriculum.stage_names
    
    # Create env
    LOGGER.info("Creating evaluation environment...")
    
    def make_env():
        env = IndoorDroneEnv(**env_config)
        # Store original target list for stage switching
        env.target_actor_names_original = env.target_actor_names.copy()
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False
    
    # Load model
    LOGGER.info("Loading model from %s", args.model)
    model = PPO.load(args.model)
    
    LOGGER.info("="*70)
    LOGGER.info("CURRICULUM EVALUATION")
    LOGGER.info("="*70)
    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Stages to evaluate: %d", len(stages_to_eval))
    LOGGER.info("Episodes per stage: %d", args.episodes)
    LOGGER.info("="*70)
    
    # Evaluate all stages
    all_results = {}
    for stage_name in stages_to_eval:
        if stage_name not in curriculum.stage_names:
            LOGGER.warning("Stage '%s' not found in curriculum, skipping", stage_name)
            continue
        
        results = evaluate_stage(
            model, 
            env, 
            curriculum, 
            stage_name, 
            n_episodes=args.episodes
        )
        all_results[stage_name] = results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    LOGGER.info("Evaluation complete. Results saved to %s", output_path)
    
    # Print overall summary
    LOGGER.info("="*70)
    LOGGER.info("OVERALL SUMMARY")
    LOGGER.info("="*70)
    LOGGER.info("%-30s %-15s %-15s", "Stage", "Success Rate", "Avg Reward")
    LOGGER.info("-"*70)
    
    for stage_name, results in all_results.items():
        LOGGER.info(
            f"{stage_name:<30} "
            f"{results['success_rate']:>6.2%} ± {results['success_std']:>5.2%}   "
            f"{results['avg_reward']:>7.2f} ± {results['std_reward']:>6.2f}"
        )
    
    LOGGER.info("="*70)
    
    # Calculate overall metrics
    all_success_rates = [r['success_rate'] for r in all_results.values()]
    overall_success = np.mean(all_success_rates)
    
    LOGGER.info("Overall Success Rate: %.2f%%", overall_success * 100)
    LOGGER.info("   Best Stage: %s", 
                max(all_results.items(), key=lambda x: x[1]['success_rate'])[0])
    LOGGER.info("   Worst Stage: %s", 
                min(all_results.items(), key=lambda x: x[1]['success_rate'])[0])
    
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
