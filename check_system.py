#!/usr/bin/env python3
"""System readiness check for drone RL training."""

import sys
import importlib
from typing import List, Tuple

def check_module(name: str, package: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        if package:
            mod = importlib.import_module(name, package)
        else:
            mod = importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Drone RL Training System - Readiness Check")
    print("=" * 60)
    print()
    
    # Check required packages
    packages = [
        ('gymnasium', None),
        ('stable_baselines3', None),
        ('airsim', None),
        ('torch', None),
        ('numpy', None),
        ('pandas', None),
        ('yaml', 'pyyaml'),
        ('zmq', 'pyzmq'),
        ('cv2', 'opencv-python'),
    ]
    
    print("Checking Python packages...")
    all_good = True
    for module, display_name in packages:
        success, info = check_module(module)
        display = display_name or module
        if success:
            print(f"  ✅ {display:25s} version {info}")
        else:
            print(f"  ❌ {display:25s} NOT FOUND")
            all_good = False
    print()
    
    # Check project modules
    print("Checking project modules...")
    project_modules = [
        'envs.indoor_drone_env',
        'envs.features',
        'envs.reward',
        'training.train_ppo',
        'baselines.rrt_star_complete',
    ]
    
    for module in project_modules:
        success, info = check_module(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module} - {info}")
            all_good = False
    print()
    
    # Check configuration files
    print("Checking configuration files...")
    import os
    config_files = [
        'configs/ppo_config.yaml',
        'configs/fixed_config.json',
        'configs/settings.json',
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file} NOT FOUND")
            all_good = False
    print()
    
    # Summary
    print("=" * 60)
    if all_good:
        print("✅ System is ready for training!")
        print()
        print("Quick start:")
        print("  bash scripts/quick_train.sh")
        print()
        print("Or manually:")
        print("  python3 training/train_ppo.py \\")
        print("      --fixed-ned-json configs/fixed_config.json \\")
        print("      --ppo-config configs/ppo_config.yaml")
        print()
    else:
        print("❌ System has missing dependencies or files")
        print()
        print("To fix:")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()
