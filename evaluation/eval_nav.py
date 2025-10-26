
import argparse
import csv
import math
import os
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.indoor_drone_env import IndoorDroneEnv

def run_episode(
    model: PPO,
    env: IndoorDroneEnv,
    max_steps: int,
    episode_id: int,
    agent: str,
    seed: int,
    map_id: str,
    per_step_writer: Optional[csv.DictWriter] = None,
) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_energy = 0.0
    total_jerk = 0.0
    max_jerk = 0.0
    collisions = 0
    steps = 0
    success = 0

    prev_pos = obs[:3].copy()
    start_xyz = tuple(map(float, prev_pos.tolist()))
    path_len_m = 0.0
    rpe_vals: List[float] = []
    goal_xyz = tuple(map(float, env.goal.tolist()))
    while not done and steps < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        total_energy += info.get('energy', 0.0)
        j = info.get('jerk', 0.0)
        total_jerk += j
        max_jerk = max(max_jerk, j)

        pos = obs[:3]
        try:
            path_len_m += float(math.dist(prev_pos.tolist(), pos.tolist()))
        except Exception:

            path_len_m += float(np.linalg.norm(prev_pos - pos))
        prev_pos = pos.copy()

        try:
            rpe_vals.append(float(obs[-1]))
        except Exception:
            pass
        if info.get('collision', False):
            collisions += 1
        if done and not info.get('collision', False):
            success = 1
        steps += 1

        if per_step_writer is not None:
            per_step_writer.writerow({
                'episode_id': episode_id,
                'step': steps,
                'jerk': j,
                'energy': info.get('energy', 0.0),
                'distance_to_goal': info.get('distance_to_goal', float('nan')),
                'pos_x': float(pos[0]),
                'pos_y': float(pos[1]),
                'pos_z': float(pos[2]),
            })

    battery_used = float(1.0 - getattr(env, 'battery', 1.0))
    rpe_mean = float(np.mean(rpe_vals)) if len(rpe_vals) > 0 else float('nan')
    metrics = {
        'episode_id': episode_id,
        'agent': agent,
        'seed': seed,
        'map_id': map_id,
        'start_xyz': start_xyz,
        'goal_xyz': goal_xyz,
        'success': bool(success),
        'collisions': collisions,
        'time_s': steps * env.dt,
        'energy_j': total_energy,
        'path_len_m': path_len_m,
        'mean_jerk': total_jerk / max(1, steps),
        'max_jerk': max_jerk,
        'ATE_m': float('nan'),
        'RPE_mean': rpe_mean,
        'battery_used_frac': max(0.0, min(1.0, battery_used)),
        'total_reward': total_reward,
    }
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO policy')
    parser.add_argument('--model', required=True, help='Path to model zip file')
    parser.add_argument('--vecnorm', default=None, help='Path to VecNormalize stats (pkl)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--feature-addr', default='tcp://127.0.0.1:5557', help='ZMQ address of feature bridge')
    parser.add_argument('--dt', type=float, default=0.1, help='Environment time step')
    parser.add_argument('--goal', nargs=3, type=float, metavar=('X', 'Y', 'Z'), default=[10.0, 5.0, -1.5], help='Goal position for evaluation')
    parser.add_argument('--output', default='eval_results.csv', help='CSV file to write episode metrics to')
    parser.add_argument('--agent', default='ppo', help='Agent name for logs (e.g., ppo, a_star, rrt_star)')
    parser.add_argument('--seed', type=int, default=0, help='Evaluation seed')
    parser.add_argument('--map-id', default='map-unknown', help='Map identifier for logs')
    parser.add_argument('--episode-id-start', type=int, default=0, help='Starting episode id in logs')
    parser.add_argument('--per-step-log', default=None, help='Optional CSV file to append per-step metrics')
    args = parser.parse_args()

    def _env_init() -> IndoorDroneEnv:
        cfg = {
            'goal': args.goal,
            'feature_addr': args.feature_addr,
            'dt': args.dt,
        }
        return IndoorDroneEnv(cfg)
    venv = DummyVecEnv([_env_init])

    if args.vecnorm:
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, env=venv)

    results: List[Dict[str, float]] = []
    per_step_fp = None
    per_step_writer: Optional[csv.DictWriter] = None
    if args.per_step_log:

        exists = os.path.exists(args.per_step_log)
        per_step_fp = open(args.per_step_log, 'a', newline='')
        per_step_writer = csv.DictWriter(per_step_fp, fieldnames=['episode_id', 'step', 'jerk', 'energy', 'distance_to_goal', 'pos_x', 'pos_y', 'pos_z'])
        if not exists:
            per_step_writer.writeheader()
    try:
        for i in range(args.episodes):
            ep_id = args.episode_id_start + i
            m = run_episode(
                model=model,
                env=venv.envs[0],
                max_steps=500,
                episode_id=ep_id,
                agent=args.agent,
                seed=args.seed + i,
                map_id=args.map_id,
                per_step_writer=per_step_writer,
            )
            results.append(m)
            print(f"Episode {i+1}/{args.episodes}: agent={m['agent']}, success={int(m['success'])}, time={m['time_s']:.1f}s, energy={m['energy_j']:.2f}, path={m['path_len_m']:.2f}m")
    finally:
        if per_step_fp is not None:
            per_step_fp.close()

    fieldnames = ['episode_id', 'agent', 'seed', 'map_id', 'start_xyz', 'goal_xyz', 'success', 'collisions', 'time_s', 'energy_j', 'path_len_m', 'mean_jerk', 'max_jerk', 'ATE_m', 'RPE_mean', 'battery_used_frac', 'total_reward']
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f'Results written to {args.output}')

if __name__ == '__main__':
    main()
