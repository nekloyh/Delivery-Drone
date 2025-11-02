# DroneDelivery-RL: Autonomous Drone Navigation with Curriculum Learning

## Project Summary

This project implements a complete Reinforcement Learning system for training autonomous drones to navigate indoor environments using AirSim simulator. The system uses PPO (Proximal Policy Optimization) with a sophisticated curriculum learning approach to progressively train the drone from simple to complex navigation tasks.

## What You're Building

### Core Problem
Training a drone to autonomously navigate from a spawn point (PlayerStart) to various landing pads (Landing_101 through Landing_606) in an indoor environment, while avoiding obstacles and optimizing for energy efficiency.

### Key Innovation
Instead of random training, you're using **curriculum learning** - starting with easy targets and progressively increasing difficulty across 10 stages, mimicking how humans learn.

## System Architecture

### 1. Environment (AirSim Integration)
- **File**: [`envs/indoor_drone_env.py`](envs/indoor_drone_env.py)
- **Purpose**: Gymnasium-compatible environment wrapping AirSim
- **Key Features**:
  - Continuous action space (velocity commands: vx, vy, vz, yaw_rate)
  - 35-dimensional observation space (position, velocity, goal direction, occupancy, battery, SLAM error)
  - Reward shaping with distance, energy, collision, and timeout penalties
  - Battery simulation with energy consumption model

### 2. Curriculum Learning System
- **File**: [`training/curriculum_manager.py`](training/curriculum_manager.py)
- **Purpose**: Manages progressive difficulty stages
- **10 Stages**:
  ```
  Stage 1: Single target (Landing_101) - 70% success threshold
  Stages 2a-2f: Floor-by-floor (6 targets each) - 50-75% thresholds
  Stages 3a-3b: Mixed floors (18 targets) - 55-65% thresholds
  Stage 4: All 36 targets randomized - 55% threshold
  ```
- **Auto-advancement**: Moves to next stage when success rate exceeds threshold over 1000 episodes

### 3. Training Pipeline
- **File**: [`training/train_ppo_curriculum.py`](training/train_ppo_curriculum.py)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Features**:
  - Automatic stage progression
  - Checkpointing at each stage transition
  - TensorBoard logging with curriculum metrics
  - VecNormalize for observation/reward normalization
  - Resume capability from any checkpoint

### 4. Evaluation System
- **File**: [`evaluation/eval_curriculum.py`](evaluation/eval_curriculum.py)
- **Purpose**: Comprehensive testing across all stages
- **Metrics**:
  - Success rate
  - Average reward
  - Episode length
  - Collision rate
  - Timeout rate
  - Energy consumption

### 5. SLAM Integration (Optional)
- **File**: [`bridges/feature_bridge.py`](bridges/feature_bridge.py)
- **Purpose**: Connects ORB-SLAM3 pose estimates via ZMQ
- **Benefit**: Tests drone's ability to navigate with real localization noise

## Project Structure

```
workspace/
├── envs/
│   ├── indoor_drone_env.py      # Main environment
│   ├── features.py               # State feature extraction
│   └── reward.py                 # Reward computation
├── training/
│   ├── curriculum_manager.py    # Curriculum logic
│   └── train_ppo_curriculum.py  # Training script
├── evaluation/
│   ├── eval_curriculum.py       # Evaluation script
│   └── energy_metrics.py        # Energy analysis
├── baselines/
│   ├── rrt_star_complete.py     # RRT* baseline
│   └── rrt_star_fixed.py        # Fixed RRT* variant
├── bridges/
│   └── feature_bridge.py        # SLAM integration
├── configs/
│   ├── curriculum_config.json   # Curriculum stages
│   ├── fixed_config.json        # Environment config
│   └── ppo_config.yaml          # PPO hyperparameters
├── scripts/
│   ├── train_curriculum.sh      # Training launcher
│   └── eval_curriculum.sh       # Evaluation launcher
├── tools/
│   └── utilities.py             # Helper functions
└── docker/
    ├── Dockerfile               # Container definition
    └── compose.yml              # Docker Compose config
```

## How It Works

### Training Flow

1. **Initialize**:
   ```bash
   ./scripts/train_curriculum.sh
   ```
   - Loads curriculum config (10 stages)
   - Creates environment with first stage targets (Landing_101)
   - Initializes PPO model or loads checkpoint

2. **Training Loop**:
   - Agent interacts with environment
   - PPO updates policy every 2048 steps
   - Curriculum callback tracks success rate
   - When threshold reached → advance to next stage
   - Saves checkpoint at each transition

3. **Stage Progression**:
   ```
   Episode 5000  → Stage 1 success: 72% → Advance to Stage 2a
   Episode 15000 → Stage 2a success: 77% → Advance to Stage 2b
   Episode 25000 → Stage 2b success: 73% → Advance to Stage 2c
   ...
   Episode 85000 → Stage 4 success: 57% → Training complete
   ```

### Evaluation Flow

1. **Run Evaluation**:
   ```bash
   ./scripts/eval_curriculum.sh logs_curriculum/latest/final_model.zip
   ```

2. **Test All Stages**:
   - Loads trained model
   - Tests on Stage 1 targets (50 episodes)
   - Tests on Stage 2a targets (50 episodes)
   - ...
   - Tests on Stage 4 targets (50 episodes)

3. **Generate Report**:
   - JSON output with metrics per stage
   - Success rates, rewards, collision rates
   - Identifies strengths/weaknesses

## Configuration Details

### Curriculum Config ([`configs/curriculum_config.json`](configs/curriculum_config.json))
```json
{
  "stage_1_static": {
    "targets": ["Landing_101"],
    "max_steps": 500,
    "success_threshold": 0.7,
    "min_episodes": 5000
  },
  "stage_2_floor1": {
    "targets": ["Landing_101", ..., "Landing_106"],
    "max_steps": 800,
    "success_threshold": 0.75,
    "min_episodes": 10000
  },
  ...
}
```

### Environment Config ([`configs/fixed_config.json`](configs/fixed_config.json))
```json
{
  "dt": 0.1,
  "max_vxy": 1.0,
  "max_vz": 0.5,
  "spawn_actor_name": "PlayerStart",
  "target_actor_names": ["Landing_101", ..., "Landing_606"],
  "reward": {
    "w_dist": -0.5,
    "w_energy": -0.01,
    "w_collision": -10.0,
    "w_success": 50.0
  }
}
```

### PPO Hyperparameters ([`configs/ppo_config.yaml`](configs/ppo_config.yaml))
```yaml
ppo_params:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
```

## Current Status

### ✅ Completed Components

1. **Core Environment** - Fully functional with AirSim
2. **Curriculum System** - 10 stages with auto-advancement
3. **Training Pipeline** - PPO with checkpointing and resume
4. **Evaluation System** - Multi-stage testing with metrics
5. **SLAM Integration** - Optional bridge for realistic localization
6. **Baseline Algorithms** - RRT* for comparison
7. **Energy Metrics** - Battery simulation and analysis
8. **Documentation** - Comprehensive guides and references

### 🎯 Ready to Use

```bash
# Start training (one command)
./scripts/train_curriculum.sh

# Monitor progress
tensorboard --logdir logs_curriculum/

# Evaluate model
./scripts/eval_curriculum.sh
```

## Expected Training Timeline

| Stage | Episodes | Time (1 GPU) | Cumulative |
|-------|----------|--------------|------------|
| Stage 1 | 5,000 | 2-3 days | 2-3 days |
| Stages 2a-2f | 60,000 | 10-12 days | 12-15 days |
| Stages 3a-3b | 30,000 | 5-6 days | 17-21 days |
| Stage 4 | 20,000 | 3-4 days | 20-25 days |

**Total**: ~3-4 weeks for full training (10M timesteps)

## Expected Results

After full training:

- **Easy targets (Floor 1)**: 80-95% success rate
- **Medium targets (Floors 2-3)**: 65-80% success rate
- **Hard targets (Floors 4-6)**: 50-70% success rate
- **All targets (Stage 4)**: 55-65% success rate
- **Collision rate**: <15% overall
- **Energy efficiency**: 30-50% better than RRT*

## Key Design Decisions

### Why Curriculum Learning?
- **Faster convergence**: Agent learns basics before tackling hard tasks
- **Better generalization**: Progressive exposure to diverse scenarios
- **Stability**: Avoids catastrophic forgetting from random sampling
- **Interpretability**: Clear progression metrics at each stage

### Why PPO?
- **Sample efficient**: Works well with expensive simulation
- **Stable**: Clipped objective prevents destructive updates
- **Proven**: State-of-the-art for continuous control tasks
- **Well-supported**: Excellent Stable-Baselines3 implementation

### Why AirSim?
- **Realistic physics**: High-fidelity drone dynamics
- **Visual rendering**: Supports camera-based navigation (future work)
- **Unreal Engine**: Photorealistic indoor environments
- **Active community**: Good documentation and support

## Code Quality Standards

This project follows **senior engineer coding practices**:

- ✅ **No emojis** in source code
- ✅ **Printf-style logging** (not f-strings in LOGGER calls)
- ✅ **Type hints** throughout
- ✅ **Comprehensive docstrings** with parameters and return types
- ✅ **Professional error handling** with informative messages
- ✅ **Configuration-driven** design (no magic numbers)
- ✅ **Modular architecture** with clear separation of concerns
- ✅ **PEP 8 compliant** formatting

## Monitoring and Debugging

### TensorBoard Metrics
```bash
tensorboard --logdir logs_curriculum/
```

View:
- Episode rewards over time
- Success rate per stage
- Policy loss and value loss
- Entropy (exploration)
- Curriculum stage progression
- Average episode length

### Log Files
- **Training logs**: `logs_curriculum/<timestamp>/`
- **Curriculum state**: `logs_curriculum/latest/curriculum_state.json`
- **Checkpoints**: `logs_curriculum/latest/ppo_curriculum_<steps>_steps.zip`

### Common Issues

1. **Low success rate in early stages**:
   - Check spawn/target positions in Unreal
   - Verify collision detection is working
   - Ensure battery capacity is sufficient

2. **Stage not advancing**:
   - Check `curriculum_state.json` for current metrics
   - May need to adjust success threshold
   - Verify min_episodes requirement is reasonable

3. **Training instability**:
   - Reduce learning rate
   - Increase batch size
   - Check reward scaling

## Next Steps (Potential Enhancements)

### Short Term
1. **Hyperparameter tuning**: Grid search for optimal PPO params
2. **Reward shaping**: Fine-tune weights based on early results
3. **Visualization**: Add trajectory plotting in evaluation

### Medium Term
1. **Vision-based navigation**: Use camera images instead of ground truth position
2. **Domain randomization**: Vary environment parameters for robustness
3. **Multi-agent training**: Multiple drones learning simultaneously

### Long Term
1. **Sim-to-real transfer**: Deploy on physical drone
2. **Dynamic obstacles**: Moving targets and obstacles
3. **Hierarchical RL**: High-level planner + low-level controller

## Research Context

This project builds on:
- **Curriculum Learning**: Bengio et al. (2009), OpenAI's emergent tool use
- **PPO**: Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
- **Drone RL**: Various works using AirSim for autonomous navigation
- **Progressive Neural Networks**: Rusu et al. (2016), DeepMind

## References

- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Curriculum Learning Survey](https://arxiv.org/abs/2010.13166)

## Contact and Contribution

For questions or improvements:
1. Check documentation in `docs/` folder
2. Review curriculum config for stage adjustments
3. Modify reward weights in `configs/fixed_config.json`
4. Experiment with PPO hyperparameters in `configs/ppo_config.yaml`

---

**Project Status**: Production-ready for training and evaluation
**Last Updated**: 2025-11-02
**Version**: 1.0.0