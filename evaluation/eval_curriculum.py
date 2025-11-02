"""Evaluate trained model across all curriculum stages.

This script evaluates a trained PPO model on all curriculum stages to assess
generalization and performance across different difficulty levels.

Usage:
    python evaluation/eval_curriculum.py --model logs_curriculum/latest/final_model.zip --vecnorm logs_curriculum/latest/vecnorm_final.pkl
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
    
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info(f"📊 Evaluating stage: {stage_name}")
    LOGGER.info(f"   Description: {stage['description']}")
    LOGGER.info(f"   Targets: {len(filtered_targets)}")
    LOGGER.info(f"   Max steps: {stage['max_steps']}")
    LOGGER.info(f"{'='*70}")
    
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
    LOGGER.info(f"\n📈 Results for {stage_name}:")
    LOGGER.info(f"   Success Rate: {results['success_rate']:.2%} ± {results['success_std']:.2%}")
    LOGGER.info(f"   Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    LOGGER.info(f"   Median Reward: {results['median_reward']:.2f}")
    LOGGER.info(f"   Avg Episode Length: {results['avg_length']:.0f} steps")
    LOGGER.info(f"   Avg Final Distance: {results['avg_final_distance']:.2f}m")
    LOGGER.info(f"   Collision Rate: {results['collision_rate']:.2%}")
    LOGGER.info(f"   Timeout Rate: {results['timeout_rate']:.2%}")
    LOGGER.info(f"   Battery Depleted Rate: {results['battery_depleted_rate']:.2%}")
    LOGGER.info(f"   Avg Energy: {results['avg_energy_per_episode']:.4f}")
    
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
        LOGGER.error(f"Model file not found: {args.model}")
        return 1
    
    if not Path(args.vecnorm).exists():
        LOGGER.error(f"VecNormalize file not found: {args.vecnorm}")
        return 1
    
    # Load env config
    LOGGER.info("📖 Loading configurations...")
    with open(args.env_config, 'r') as f:
        env_config = json.load(f)
    
    # Load curriculum
    curriculum = CurriculumManager(args.curriculum_config)
    
    # Determine which stages to evaluate
    stages_to_eval = args.stages if args.stages else curriculum.stage_names
    
    # Create env
    LOGGER.info("🚁 Creating evaluation environment...")
    
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
    LOGGER.info(f"📂 Loading model from {args.model}")
    model = PPO.load(args.model)
    
    LOGGER.info("\n" + "="*70)
    LOGGER.info("🎯 CURRICULUM EVALUATION")
    LOGGER.info("="*70)
    LOGGER.info(f"Model: {args.model}")
    LOGGER.info(f"Stages to evaluate: {len(stages_to_eval)}")
    LOGGER.info(f"Episodes per stage: {args.episodes}")
    LOGGER.info("="*70 + "\n")
    
    # Evaluate all stages
    all_results = {}
    for stage_name in stages_to_eval:
        if stage_name not in curriculum.stage_names:
            LOGGER.warning(f"Stage '{stage_name}' not found in curriculum, skipping")
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
    
    LOGGER.info(f"\n✅ Evaluation complete. Results saved to {output_path}")
    
    # Print overall summary
    LOGGER.info(f"\n{'='*70}")
    LOGGER.info("📊 OVERALL SUMMARY")
    LOGGER.info(f"{'='*70}")
    LOGGER.info(f"{'Stage':<30} {'Success Rate':<15} {'Avg Reward':<15}")
    LOGGER.info("-"*70)
    
    for stage_name, results in all_results.items():
        LOGGER.info(
            f"{stage_name:<30} "
            f"{results['success_rate']:>6.2%} ± {results['success_std']:>5.2%}   "
            f"{results['avg_reward']:>7.2f} ± {results['std_reward']:>6.2f}"
        )
    
    LOGGER.info("="*70 + "\n")
    
    # Calculate overall metrics
    all_success_rates = [r['success_rate'] for r in all_results.values()]
    overall_success = np.mean(all_success_rates)
    
    LOGGER.info(f"📈 Overall Success Rate: {overall_success:.2%}")
    LOGGER.info(f"   Best Stage: {max(all_results.items(), key=lambda x: x[1]['success_rate'])[0]}")
    LOGGER.info(f"   Worst Stage: {min(all_results.items(), key=lambda x: x[1]['success_rate'])[0]}")
    
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
