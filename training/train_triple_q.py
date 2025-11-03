import os, argparse, yaml, numpy as np, torch
import time
from envs.triq_indoor_drone_env import IndoorDroneEnv
from algos.deep_triple_q import DeepTripleQ

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def curriculum_noise_schedule(episode, total_episodes, start_noise=0.5, end_noise=0.1):
    """
    Giảm dần exploration noise theo curriculum (Giữ nguyên)
    """
    progress = episode / total_episodes
    noise = start_noise * (end_noise / start_noise) ** progress
    return noise


def safe_training_loop(agent, env, cfg):
    """
    Training loop đã được dọn dẹp
    """
    episodes = int(cfg["algo"]["episodes"])
    save_interval = int(cfg["algo"]["save_interval_episodes"])
    
    start_noise = float(cfg["algo"].get("start_noise_std", 0.5))
    end_noise = float(cfg["algo"].get("noise_std", 0.1))
    
    frame_returns, frame_utils = [], []
    frame_success, frame_viol = 0, 0
    episodes_per_frame = int(cfg["algo"]["episodes_per_frame"])
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for ep in range(1, episodes + 1):
        noise_std = curriculum_noise_schedule(ep, episodes, start_noise, end_noise)
        
        # Reset (HÀM RESET MỚI ĐÃ TỰ ỔN ĐỊNH)
        obs, _ = env.reset()
        # XÓA stabilize_drone()
        
        ep_ret, ep_len, ep_util = 0.0, 0, 0.0
        reached, viol = False, False
        
        while True:
            action = agent.select_action(obs, noise_std)
            
            # Safety filter đã được gọi bên trong agent.train() (deep_triple_q.py)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Lấy utility từ tín hiệu LIDAR dày đặc (đã sửa trong deep_triple_q.py)
            min_dist_obs = info.get("min_distance_obstacle", 5.0)
            utility_margin = 1.5
            utility = min(1.0, max(0.0, (min_dist_obs - 0.5) / utility_margin))
            
            agent.buf.add(obs, action, reward, utility, next_obs, done)
            
            obs = next_obs
            ep_ret += reward
            ep_len += 1
            ep_util += utility
            
            reached = reached or bool(info.get("success", False))
            viol = viol or bool(info.get("collision", False))
            
            if agent.global_step > agent.warmup:
                _ = agent.update()
            
            agent.global_step += 1
            
            if done:
                break
        
        # Frame tracking
        frame_returns.append(ep_ret)
        frame_utils.append(ep_util)
        frame_success += int(reached)
        frame_viol += int(viol)
        
        # Logging mỗi episode
        if ep % 10 == 0:
            recent_ret = np.mean(frame_returns[-10:])
            recent_util = np.mean(frame_utils[-10:])
            print(f"[Ep {ep:4d}] Ret: {recent_ret:7.1f} | Util: {recent_util:5.2f} | "
                  f"Noise: {noise_std:.3f} | Z: {agent.Z:.3f} | Success: {frame_success} | Viol: {frame_viol}")
        
        # Frame boundary - update Z
        if ep % episodes_per_frame == 0:
            mean_utility = float(np.mean(frame_utils[-episodes_per_frame:]))
            agent.Z = max(0.0, agent.Z + agent.rho + agent.epsilon - mean_utility)
            
            # Write metrics
            with open(agent.csv_path, "a", encoding="utf-8") as f:
                f.write(f"{ep},{ep_ret:.3f},{ep_len},{ep_util:.3f},{frame_viol},"
                       f"{frame_success},{agent.Z:.4f},{agent.rho:.3f}\n")
            
            # Reset frame counters
            frame_success, frame_viol = 0, 0
        
        # Save checkpoint
        if ep % save_interval == 0 or ep == episodes:
            agent.save(os.path.join(agent.run_dir, f"agent_ep{ep}.pt"))
            print(f"[SAVE] Checkpoint saved at episode {ep}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Final model saved in: {agent.run_dir}")
    print("="*60)

def main(args):
    cfg = load_cfg(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    
    print("\n" + "="*60)
    print("INITIALIZING ENVIRONMENT")
    print("="*60)
    
    env_cfg = cfg.get("env", {})
    env_cfg["seed"] = cfg.get("algo", {}).get("seed")
    
    # CRITICAL: Đảm bảo spawn trên không (Code của bạn đã đúng)
    env_cfg["spawn_on_ground"] = False
    env_cfg["spawn_agl"] = 1.0
    
    env = IndoorDroneEnv(cfg=env_cfg)
    print(f"✓ Environment created: {env.__class__.__name__}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    print("\n" + "="*60)
    print("INITIALIZING AGENT")
    print("="*60)
    
    agent = DeepTripleQ(env, cfg, device=device)
    print(f"✓ Agent created: DeepTripleQ")
    print(f"  Device: {device}")
    print(f"  Warmup steps: {agent.warmup}")

    # ----------------------------------------------------------
    # XÓA PHASE 1 (Stabilization) và PHASE 2 (Warmup)
    # Chúng không cần thiết và làm chậm quá trình.
    # ----------------------------------------------------------
    
    # Phase 3: Main training loop
    print("\n" + "="*60)
    print("STARTING MAIN TRAINING")
    print("="*60)
    safe_training_loop(agent, env, cfg)
    
    env.close()
    print("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Triple-Q với proper drone initialization")
    parser.add_argument("--config", type=str, default="configs/indoor.yaml",
                       help="Path to config file")
    parser.add_argument("--skip-warmup", action="store_true",
                       help="Skip warmup phase (not recommended)")
    args = parser.parse_args()
    main(args)