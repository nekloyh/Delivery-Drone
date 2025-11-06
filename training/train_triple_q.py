# import os, argparse, yaml, numpy as np, torch
# import time
# from envs.triq_indoor_drone_env import IndoorDroneEnv
# from algos.deep_triple_q import DeepTripleQ
# import numpy as np

# def load_cfg(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

# def curriculum_noise_schedule(episode, total_episodes, start_noise=0.3, end_noise=0.1):
#     """
#     Giảm dần exploration noise theo curriculum (Giữ nguyên)
#     """
#     progress = episode / total_episodes
#     noise = start_noise * (end_noise / start_noise) ** progress
#     return noise


# def safe_training_loop(agent, env, cfg):
#     """
#     Training loop đã được dọn dẹp
#     """
#     episodes = int(cfg["algo"]["episodes"])
#     save_interval = int(cfg["algo"]["save_interval_episodes"])
    
#     start_noise = float(cfg["algo"].get("start_noise_std", 0.5))
#     end_noise = float(cfg["algo"].get("noise_std", 0.1))
    
#     frame_returns, frame_utils = [], []
#     frame_success, frame_viol = 0, 0
#     episodes_per_frame = int(cfg["algo"]["episodes_per_frame"])
    
#     print("\n" + "="*60)
#     print("STARTING TRAINING")
#     print("="*60)
    
#     for ep in range(1, episodes + 1):
#         noise_std = curriculum_noise_schedule(ep, episodes, start_noise, end_noise)
        
#         # Reset (HÀM RESET MỚI ĐÃ TỰ ỔN ĐỊNH)
#         obs, _ = env.reset()
#         # XÓA stabilize_drone()
        
#         ep_ret, ep_len, ep_util = 0.0, 0, 0.0
#         reached, viol = False, False
        
#         while True:
#             action = agent.select_action(obs, noise_std)
            
#             # Safety filter đã được gọi bên trong agent.train() (deep_triple_q.py)
            
#             next_obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
            
#             # Lấy utility từ tín hiệu LIDAR dày đặc (đã sửa trong deep_triple_q.py)
#             min_dist_obs = info.get("min_distance_obstacle", 5.0)
#             utility_margin = 1.5
#             x = (min_dist_obs - 0.5) / utility_margin
#             utility = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
#             # utility = min(1.0, max(0.0, (min_dist_obs - 0.5) / utility_margin))
            
#             agent.buf.add(obs, action, reward, utility, next_obs, done)
            
#             obs = next_obs
#             ep_ret += reward
#             ep_len += 1
#             ep_util += utility
            
#             reached = reached or bool(info.get("success", False))
#             viol = viol or bool(info.get("collision", False))
            
#             if agent.global_step > agent.warmup:
#                 _ = agent.update()
            
#             agent.global_step += 1
            
#             if done:
#                 break
        
#         # Frame tracking
#         frame_returns.append(ep_ret)
#         frame_utils.append(ep_util)
#         frame_success += int(reached)
#         frame_viol += int(viol)
        
#         # Logging mỗi episode
#         if ep % 10 == 0:
#             recent_ret = np.mean(frame_returns[-10:])
#             recent_util = np.mean(frame_utils[-10:])
#             print(f"[Ep {ep:4d}] Ret: {recent_ret:7.1f} | Util: {recent_util:5.2f} | "
#                   f"Noise: {noise_std:.3f} | Z: {agent.Z:.3f} | Success: {frame_success} | Viol: {frame_viol}")
        
#         # Frame boundary - update Z
#         if ep % episodes_per_frame == 0:
#             mean_utility = float(np.mean(frame_utils[-episodes_per_frame:]))
#             agent.Z = max(0.0, agent.Z + agent.rho + agent.epsilon - mean_utility)
            
#             # Write metrics
#             with open(agent.csv_path, "a", encoding="utf-8") as f:
#                 f.write(f"{ep},{ep_ret:.3f},{ep_len},{ep_util:.3f},{frame_viol},"
#                        f"{frame_success},{agent.Z:.4f},{agent.rho:.3f}\n")
            
#             # Reset frame counters
#             frame_success, frame_viol = 0, 0
        
#         # Save checkpoint
#         if ep % save_interval == 0 or ep == episodes:
#             agent.save(os.path.join(agent.run_dir, f"agent_ep{ep}.pt"))
#             print(f"[SAVE] Checkpoint saved at episode {ep}")
    
#     print("\n" + "="*60)
#     print("TRAINING COMPLETED")
#     print(f"Final model saved in: {agent.run_dir}")
#     print("="*60)

# def main(args):
#     cfg = load_cfg(args.config)
#     device = torch.device(cfg.get("device", "cpu"))
    
#     print("\n" + "="*60)
#     print("INITIALIZING ENVIRONMENT")
#     print("="*60)
    
#     env_cfg = cfg.get("env", {})
#     env_cfg["seed"] = cfg.get("algo", {}).get("seed")
    
#     # CRITICAL: Đảm bảo spawn trên không (Code của bạn đã đúng)
#     env_cfg["spawn_on_ground"] = False
#     env_cfg["spawn_agl"] = 1.0
    
#     env = IndoorDroneEnv(cfg=env_cfg)
#     print(f"✓ Environment created: {env.__class__.__name__}")
#     print(f"  Observation space: {env.observation_space.shape}")
#     print(f"  Action space: {env.action_space.shape}")
    
#     print("\n" + "="*60)
#     print("INITIALIZING AGENT")
#     print("="*60)
    
#     agent = DeepTripleQ(env, cfg, device=device)
#     print(f"✓ Agent created: DeepTripleQ")
#     print(f"  Device: {device}")
#     print(f"  Warmup steps: {agent.warmup}")

#     # ----------------------------------------------------------
#     # XÓA PHASE 1 (Stabilization) và PHASE 2 (Warmup)
#     # Chúng không cần thiết và làm chậm quá trình.
#     # ----------------------------------------------------------
    
#     # Phase 3: Main training loop
#     print("\n" + "="*60)
#     print("STARTING MAIN TRAINING")
#     print("="*60)
#     safe_training_loop(agent, env, cfg)
    
#     env.close()
#     print("\n✓ Training pipeline completed successfully!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train Triple-Q với proper drone initialization")
#     parser.add_argument("--config", type=str, default="configs/indoor.yaml",
#                        help="Path to config file")
#     parser.add_argument("--skip-warmup", action="store_true",
#                        help="Skip warmup phase (not recommended)")
#     args = parser.parse_args()
#     main(args)




import os
import argparse
import yaml
import numpy as np
import torch
import time
import logging
from datetime import datetime

from envs.triq_indoor_drone_env import IndoorDroneEnv
from algos.deep_triple_q import DeepTripleQ

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def curriculum_noise_schedule(episode, total_episodes, start_noise=0.3, end_noise=0.1):
    """Giảm dần exploration noise theo curriculum."""
    progress = episode / total_episodes
    noise = start_noise * (end_noise / start_noise) ** progress
    return noise

def safe_training_loop(agent, env, cfg):
    """Enhanced training loop with detailed logging."""
    
    episodes = int(cfg["algo"]["episodes"])
    save_interval = int(cfg["algo"]["save_interval_episodes"])
    start_noise = float(cfg["algo"].get("start_noise_std", 0.5))
    end_noise = float(cfg["algo"].get("noise_std", 0.1))
    
    frame_returns, frame_utils = [], []
    frame_success, frame_viol = 0, 0
    episodes_per_frame = int(cfg["algo"]["episodes_per_frame"])
    
    print("\n" + "="*80)
    print("STARTING TRAINING WITH DETAILED LOGGING")
    print("="*80)
    LOGGER.info(f"Total Episodes: {episodes}")
    LOGGER.info(f"Episodes per Frame: {episodes_per_frame}")
    LOGGER.info(f"Warmup Steps: {cfg['algo'].get('warmup_steps', 0)}")
    LOGGER.info(f"Start Noise: {start_noise}, End Noise: {end_noise}")
    print("="*80 + "\n")
    
    episode_count = 0
    global_step = 0
    
    # Statistics tracking
    best_reward = -np.inf
    recent_rewards = []
    recent_distances = []
    recent_actions = []
    
    while episode_count < episodes:
        # Curriculum noise
        noise_std = curriculum_noise_schedule(episode_count, episodes, start_noise, end_noise)
        
        # Reset
        obs, info = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        # Episode statistics
        episode_reward = 0.0
        episode_collisions = 0
        episode_actions = []
        episode_distances = []
        episode_positions = []
        
        spawn_pos = info['spawn_position']
        goal_pos = info['goal_position']
        initial_dist = float(np.linalg.norm(goal_pos - spawn_pos))
        
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Episode {episode_count + 1}/{episodes} | Noise: {noise_std:.4f}")
        LOGGER.info(f"Spawn: [{spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f}]")
        LOGGER.info(f"Goal:  [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f}]")
        LOGGER.info(f"Initial Distance: {initial_dist:.2f}m")
        LOGGER.info(f"{'='*80}")
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(obs, noise_std=noise_std)
            
            # Log action (every 10 steps)
            if step_count % 10 == 0:
                LOGGER.info(
                    f"  Step {step_count:3d} | Action: "
                    f"vx={action[0]:+.3f}, vy={action[1]:+.3f}, vz={action[2]:+.3f}, yaw={action[3]:+.3f}"
                )
            
            # Execute
            next_obs, reward, done, truncated, step_info = env.step(action)
            
            # Store transition
            # agent.store_transition(obs, action, reward, next_obs, done)
            
            # Update agent
            if global_step >= cfg["algo"].get("warmup_steps", 0):
                agent.update()
            
            # Track statistics
            episode_reward += reward
            episode_actions.append(action)
            episode_distances.append(step_info['distance_to_goal'])
            episode_positions.append(obs[:3].copy())
            
            if step_info.get('collision', False):
                episode_collisions += 1
            
            # Log detailed step info (every 10 steps)
            if step_count % 10 == 0:
                pos = obs[:3]
                vel = obs[3:6]
                LOGGER.info(
                    f"  Step {step_count:3d} | Pos: [{pos[0]:+7.2f}, {pos[1]:+7.2f}, {pos[2]:+7.2f}] | "
                    f"Vel: [{vel[0]:+.2f}, {vel[1]:+.2f}, {vel[2]:+.2f}] | "
                    f"Dist: {step_info['distance_to_goal']:.2f}m | "
                    f"Reward: {reward:+.2f}"
                )
            
            obs = next_obs
            step_count += 1
            global_step += 1
        
        # Episode summary
        episode_count += 1
        final_dist = episode_distances[-1] if episode_distances else initial_dist
        success = step_info.get('success', False)
        
        # Calculate statistics
        actions_array = np.array(episode_actions)
        action_mean = actions_array.mean(axis=0)
        action_std = actions_array.std(axis=0)
        action_magnitude = float(np.linalg.norm(actions_array, axis=1).mean())
        
        distance_traveled = 0.0
        if len(episode_positions) > 1:
            for i in range(1, len(episode_positions)):
                distance_traveled += float(np.linalg.norm(
                    np.array(episode_positions[i]) - np.array(episode_positions[i-1])
                ))
        
        LOGGER.info(f"\n{'─'*80}")
        LOGGER.info(f"EPISODE {episode_count} SUMMARY:")
        LOGGER.info(f"{'─'*80}")
        LOGGER.info(f"  Total Reward:        {episode_reward:+10.2f}")
        LOGGER.info(f"  Steps:               {step_count:10d}")
        LOGGER.info(f"  Success:             {str(success):>10}")
        LOGGER.info(f"  Collisions:          {episode_collisions:10d}")
        LOGGER.info(f"  Initial Distance:    {initial_dist:10.2f}m")
        LOGGER.info(f"  Final Distance:      {final_dist:10.2f}m")
        LOGGER.info(f"  Distance Traveled:   {distance_traveled:10.2f}m")
        LOGGER.info(f"  Progress:            {(1 - final_dist/max(initial_dist, 0.1))*100:9.1f}%")
        LOGGER.info(f"  Action Magnitude:    {action_magnitude:10.4f}")
        LOGGER.info(f"  Action Mean:         [{action_mean[0]:+.3f}, {action_mean[1]:+.3f}, {action_mean[2]:+.3f}, {action_mean[3]:+.3f}]")
        LOGGER.info(f"  Action Std:          [{action_std[0]:.3f}, {action_std[1]:.3f}, {action_std[2]:.3f}, {action_std[3]:.3f}]")
        LOGGER.info(f"{'─'*80}\n")
        
        # Track for frame statistics
        recent_rewards.append(episode_reward)
        recent_distances.append(final_dist)
        recent_actions.append(action_magnitude)
        
        if len(recent_rewards) > episodes_per_frame:
            recent_rewards.pop(0)
            recent_distances.pop(0)
            recent_actions.pop(0)
        
        # Update best
        if episode_reward > best_reward:
            best_reward = episode_reward
            LOGGER.info(f"🎯 NEW BEST REWARD: {best_reward:.2f}")
        
        # Frame update (every N episodes)
        if episode_count % episodes_per_frame == 0:
            frame_returns.append(float(np.mean(recent_rewards)))
            frame_utils.append(float(episode_collisions / max(step_count, 1)))
            frame_success += int(success)
            frame_viol += episode_collisions
            
            LOGGER.info(f"\n{'═'*80}")
            LOGGER.info(f"FRAME {episode_count // episodes_per_frame} SUMMARY:")
            LOGGER.info(f"{'═'*80}")
            LOGGER.info(f"  Average Reward:      {np.mean(recent_rewards):+10.2f}")
            LOGGER.info(f"  Average Distance:    {np.mean(recent_distances):10.2f}m")
            LOGGER.info(f"  Average Action Mag:  {np.mean(recent_actions):10.4f}")
            LOGGER.info(f"  Success Rate:        {frame_success/episodes_per_frame*100:9.1f}%")
            LOGGER.info(f"  Collision Rate:      {frame_viol/episodes_per_frame:10.2f}")
            LOGGER.info(f"  Virtual Queue Z:     {agent.Z:10.4f}")
            LOGGER.info(f"{'═'*80}\n")
            
            frame_success = 0
            frame_viol = 0
        
        # Save checkpoint
        if episode_count % save_interval == 0:
            save_path = f"checkpoints/ep_{episode_count}.pt"
            agent.save(save_path)
            LOGGER.info(f"💾 Checkpoint saved: {save_path}")
    
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("TRAINING COMPLETED!")
    LOGGER.info(f"Best Reward: {best_reward:.2f}")
    LOGGER.info(f"{'='*80}\n")
    
    return frame_returns, frame_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    cfg = load_cfg(args.config)
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Create environment
    LOGGER.info("Creating environment...")
    env = IndoorDroneEnv(cfg["env"])
    
    # Create agent
    LOGGER.info("Creating agent...")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    agent = DeepTripleQ(env, cfg, device=device)
    
    # Train
    LOGGER.info("Starting training...")
    frame_returns, frame_utils = safe_training_loop(agent, env, cfg)
    
    # Save final model
    agent.save("checkpoints/final_model.pt")
    LOGGER.info("Final model saved to checkpoints/final_model.pt")
    
    env.close()

if __name__ == "__main__":
    main()