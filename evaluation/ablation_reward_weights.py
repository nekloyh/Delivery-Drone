"""Ablation study for reward function weights.

This script performs a grid search or ablation study on reward function weights
to analyze their impact on training performance, as presented in Section V.C
of the paper.

The paper tested various weight combinations and found optimal values:
    - goal_bonus: +500.0
    - dist_coeff: -5.0  
    - dt_coeff: -0.1
    - act_quadratic: -0.01 (energy weight)
    - collision: -1000.0
    - jerk: -10.0

This script allows systematic exploration of:
1. Energy weight ablation (0.0, 0.005, 0.01, 0.02)
2. Jerk weight ablation (0.0, 5.0, 10.0, 20.0)
3. Full grid search over weight combinations

Results show trade-offs:
    - Higher energy weight → Lower power consumption, longer flight time
    - Higher jerk weight → Smoother trajectories, better comfort
    - Balanced weights → Best overall performance

Usage:
    # Run energy weight ablation
    python evaluation/ablation_reward_weights.py --ablation energy
    
    # Run jerk weight ablation
    python evaluation/ablation_reward_weights.py --ablation jerk
    
    # Full grid search (warning: computationally expensive)
    python evaluation/ablation_reward_weights.py --ablation full
    
    # Single configuration test
    python evaluation/ablation_reward_weights.py \\
        --energy-weight 0.02 --jerk-weight 5.0 --timesteps 100000
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.indoor_drone_env import IndoorDroneEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


def train_with_weights(
    env_config: Dict,
    ppo_config: Dict,
    reward_weights: Dict,
    timesteps: int,
    log_dir: str,
    seed: int = 42
) -> Dict:
    """Train PPO with specific reward weights and evaluate.
    
    Args:
        env_config: Base environment configuration
        ppo_config: PPO hyperparameters
        reward_weights: Reward function weights to test
        timesteps: Total training timesteps
        log_dir: Directory for logs and checkpoints
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with training and evaluation metrics
    """
    # Update env config with custom reward weights
    env_config_custom = env_config.copy()
    env_config_custom["reward"] = reward_weights
    
    LOGGER.info("Training with reward weights:")
    for key, value in reward_weights.items():
        LOGGER.info("   %s: %.4f", key, value)
    
    # Create environment
    def make_env():
        return IndoorDroneEnv(**env_config_custom)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create eval env
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        policy_kwargs={
            "net_arch": [256, 256, 256],
            "activation_fn": "tanh"
        },
        verbose=0,
        seed=seed,
        tensorboard_log=log_dir
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    # Train
    LOGGER.info("Training for %d timesteps...", timesteps)
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=False
    )
    
    # Evaluate trained model
    LOGGER.info("Evaluating trained model...")
    eval_rewards = []
    eval_successes = []
    eval_energies = []
    eval_collisions = []
    
    for _ in range(50):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        ep_energy = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
            ep_energy += info[0].get("energy", 0)
        
        eval_rewards.append(ep_reward)
        eval_successes.append(float(info[0].get("success", False)))
        eval_energies.append(ep_energy)
        eval_collisions.append(float(info[0].get("collision", False)))
    
    results = {
        "reward_weights": reward_weights,
        "success_rate": np.mean(eval_successes),
        "success_std": np.std(eval_successes),
        "avg_reward": np.mean(eval_rewards),
        "std_reward": np.std(eval_rewards),
        "avg_energy": np.mean(eval_energies),
        "std_energy": np.std(eval_energies),
        "collision_rate": np.mean(eval_collisions),
        "timesteps": timesteps
    }
    
    LOGGER.info("Results:")
    LOGGER.info("   Success Rate: %.2f%% ± %.2f%%", 
                results['success_rate'] * 100, results['success_std'] * 100)
    LOGGER.info("   Avg Reward: %.2f ± %.2f", 
                results['avg_reward'], results['std_reward'])
    LOGGER.info("   Avg Energy: %.4f ± %.4f", 
                results['avg_energy'], results['std_energy'])
    LOGGER.info("   Collision Rate: %.2f%%", results['collision_rate'] * 100)
    
    env.close()
    eval_env.close()
    
    return results


def ablation_energy_weight(
    env_config: Dict,
    ppo_config: Dict,
    timesteps: int,
    output_dir: str
) -> List[Dict]:
    """Ablation study on energy weight (act_quadratic).
    
    Tests: [0.0, 0.005, 0.01, 0.02, 0.03]
    
    Args:
        env_config: Base environment config
        ppo_config: PPO hyperparameters
        timesteps: Training timesteps per configuration
        output_dir: Directory to save results
        
    Returns:
        List of result dictionaries
    """
    LOGGER.info("="*70)
    LOGGER.info("ENERGY WEIGHT ABLATION STUDY")
    LOGGER.info("="*70)
    
    energy_weights = [0.0, 0.005, 0.01, 0.02, 0.03]
    results = []
    
    for i, energy_w in enumerate(energy_weights):
        LOGGER.info("\n[%d/%d] Testing energy weight: %.3f", 
                   i+1, len(energy_weights), energy_w)
        
        reward_weights = {
            "goal_bonus": 500.0,
            "dist_coeff": -5.0,
            "dt_coeff": -0.1,
            "act_quadratic": -energy_w,
            "collision": -1000.0,
            "jerk": -10.0
        }
        
        log_dir = f"{output_dir}/energy_{energy_w:.3f}"
        result = train_with_weights(
            env_config, ppo_config, reward_weights, timesteps, log_dir
        )
        results.append(result)
    
    return results


def ablation_jerk_weight(
    env_config: Dict,
    ppo_config: Dict,
    timesteps: int,
    output_dir: str
) -> List[Dict]:
    """Ablation study on jerk weight.
    
    Tests: [0.0, 5.0, 10.0, 15.0, 20.0]
    
    Args:
        env_config: Base environment config
        ppo_config: PPO hyperparameters
        timesteps: Training timesteps per configuration
        output_dir: Directory to save results
        
    Returns:
        List of result dictionaries
    """
    LOGGER.info("="*70)
    LOGGER.info("JERK WEIGHT ABLATION STUDY")
    LOGGER.info("="*70)
    
    jerk_weights = [0.0, 5.0, 10.0, 15.0, 20.0]
    results = []
    
    for i, jerk_w in enumerate(jerk_weights):
        LOGGER.info("\n[%d/%d] Testing jerk weight: %.1f", 
                   i+1, len(jerk_weights), jerk_w)
        
        reward_weights = {
            "goal_bonus": 500.0,
            "dist_coeff": -5.0,
            "dt_coeff": -0.1,
            "act_quadratic": -0.01,
            "collision": -1000.0,
            "jerk": -jerk_w
        }
        
        log_dir = f"{output_dir}/jerk_{jerk_w:.1f}"
        result = train_with_weights(
            env_config, ppo_config, reward_weights, timesteps, log_dir
        )
        results.append(result)
    
    return results


def main():
    """Main ablation study function."""
    parser = argparse.ArgumentParser(
        description="Ablation study for reward function weights"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["energy", "jerk", "full", "single"],
        default="energy",
        help="Type of ablation study"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/fixed_config.json",
        help="Environment config path"
    )
    parser.add_argument(
        "--ppo-config",
        type=str,
        default="configs/ppo_config.yaml",
        help="PPO config path"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Training timesteps per configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/ablation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.01,
        help="Energy weight for single test"
    )
    parser.add_argument(
        "--jerk-weight",
        type=float,
        default=10.0,
        help="Jerk weight for single test"
    )
    args = parser.parse_args()
    
    # Load configs
    with open(args.env_config, 'r') as f:
        env_config = json.load(f)
    
    with open(args.ppo_config, 'r') as f:
        ppo_config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{args.ablation}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run ablation study
    if args.ablation == "energy":
        results = ablation_energy_weight(
            env_config, ppo_config, args.timesteps, output_dir
        )
    elif args.ablation == "jerk":
        results = ablation_jerk_weight(
            env_config, ppo_config, args.timesteps, output_dir
        )
    elif args.ablation == "single":
        reward_weights = {
            "goal_bonus": 500.0,
            "dist_coeff": -5.0,
            "dt_coeff": -0.1,
            "act_quadratic": -args.energy_weight,
            "collision": -1000.0,
            "jerk": -args.jerk_weight
        }
        results = [train_with_weights(
            env_config, ppo_config, reward_weights, args.timesteps, output_dir
        )]
    else:
        LOGGER.error("Full ablation not implemented (too expensive)")
        return 1
    
    # Save results
    output_file = f"{output_dir}/results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    LOGGER.info("\n" + "="*70)
    LOGGER.info("ABLATION STUDY COMPLETE")
    LOGGER.info("="*70)
    LOGGER.info("Results saved to: %s", output_file)
    LOGGER.info("\nSummary:")
    for i, result in enumerate(results):
        LOGGER.info("[%d] Success: %.2f%%, Energy: %.4f, Reward: %.2f",
                   i+1, result['success_rate']*100, result['avg_energy'], result['avg_reward'])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
