"""Utility functions for drone RL system.

This module consolidates various utility functions for:
- System readiness checks
- Coordinate conversions
- Unreal Engine actor queries
"""

import sys
import json
import math
import logging
import importlib
from typing import Tuple, Dict, Any, List
from pathlib import Path

import airsim

LOGGER = logging.getLogger(__name__)


# ============================================================================
# System Readiness Checks
# ============================================================================

def check_module(name: str, package: str = None) -> Tuple[bool, str]:
    """Check if a Python module can be imported.
    
    Args:
        name: Module name to import
        package: Display name for the package (optional)
        
    Returns:
        Tuple of (success: bool, version_or_error: str)
    """
    try:
        if package:
            mod = importlib.import_module(name, package)
        else:
            mod = importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def check_system_readiness() -> bool:
    """Check if system is ready for training.
    
    Verifies:
    - Required Python packages are installed
    - Project modules can be imported
    - Configuration files exist
    
    Returns:
        True if system is ready, False otherwise
    """
    LOGGER.info("="*60)
    LOGGER.info("Drone RL Training System - Readiness Check")
    LOGGER.info("="*60)
    
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
    
    LOGGER.info("Checking Python packages...")
    all_good = True
    for module, display_name in packages:
        success, info = check_module(module)
        display = display_name or module
        if success:
            LOGGER.info("  [OK] %s version %s", display, info)
        else:
            LOGGER.error("  [MISSING] %s", display)
            all_good = False
    
    project_modules = [
        'envs.indoor_drone_env',
        'envs.features',
        'envs.reward',
        'training.train_ppo_curriculum',
        'training.curriculum_manager',
    ]
    
    LOGGER.info("Checking project modules...")
    for module in project_modules:
        success, info = check_module(module)
        if success:
            LOGGER.info("  [OK] %s", module)
        else:
            LOGGER.error("  [MISSING] %s - %s", module, info)
            all_good = False
    
    config_files = [
        'configs/ppo_config.yaml',
        'configs/fixed_config.json',
        'configs/curriculum_config.json',
    ]
    
    LOGGER.info("Checking configuration files...")
    for config_file in config_files:
        if Path(config_file).exists():
            LOGGER.info("  [OK] %s", config_file)
        else:
            LOGGER.error("  [MISSING] %s", config_file)
            all_good = False
    
    LOGGER.info("="*60)
    if all_good:
        LOGGER.info("System is ready for training")
        LOGGER.info("Quick start: .\\scripts\\train_curriculum.bat (Windows)")
    else:
        LOGGER.error("System check failed. Please install missing dependencies")
        LOGGER.error("Run: pip install -r requirements.txt")
    
    return all_good


# ============================================================================
# Coordinate Conversions
# ============================================================================

def ue_cm_to_ned_m(
    point_cm: Dict[str, float], 
    playerstart_cm: Dict[str, float]
) -> Dict[str, float]:
    """Convert Unreal Engine coordinates (cm, Z-up) to AirSim NED (m, Z-down).
    
    Args:
        point_cm: Point in UE coordinates {x, y, z, yaw_deg (optional)}
        playerstart_cm: PlayerStart origin in UE coordinates
        
    Returns:
        Point in NED coordinates {x, y, z, yaw}
    """
    x_m = (point_cm["x"] - playerstart_cm["x"]) / 100.0
    y_m = (point_cm["y"] - playerstart_cm["y"]) / 100.0
    z_m = -(point_cm["z"] - playerstart_cm["z"]) / 100.0
    yaw = math.radians(point_cm.get("yaw_deg", 0.0))
    
    return {"x": x_m, "y": y_m, "z": z_m, "yaw": yaw}


def convert_scenario_ue_to_ned(
    scenario_path: str,
    playerstart_cm: Dict[str, float],
    output_path: str
) -> None:
    """Convert entire scenario from UE to NED coordinates.
    
    Args:
        scenario_path: Path to input scenario JSON (UE coordinates)
        playerstart_cm: PlayerStart origin in UE coordinates
        output_path: Path to output scenario JSON (NED coordinates)
    """
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    
    spawn_ned = ue_cm_to_ned_m(scenario["spawn"], playerstart_cm)
    goals_ned = [ue_cm_to_ned_m(target, playerstart_cm) 
                 for target in scenario["targets"]]
    
    output = {
        "frame": "ned_m",
        "spawn": spawn_ned,
        "targets": goals_ned
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    LOGGER.info("Converted %s → %s", scenario_path, output_path)


# ============================================================================
# Unreal Engine Actor Queries
# ============================================================================

def get_actor_positions(
    actor_names: List[str],
    ip: str = "127.0.0.1",
    port: int = 41451
) -> Dict[str, airsim.Vector3r]:
    """Query positions of actors in Unreal Engine via AirSim.
    
    Args:
        actor_names: List of actor names to query
        ip: AirSim server IP
        port: AirSim server port
        
    Returns:
        Dictionary mapping actor names to their positions
        
    Raises:
        ConnectionError: If cannot connect to AirSim
    """
    try:
        client = airsim.MultirotorClient(ip=ip, port=port)
        client.confirmConnection()
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to AirSim at {ip}:{port}. "
            "Ensure Unreal Engine is running."
        ) from exc
    
    positions = {}
    for actor_name in actor_names:
        try:
            pose = client.simGetObjectPose(actor_name)
            positions[actor_name] = pose.position
            LOGGER.debug("%s: %s", actor_name, pose.position)
        except Exception as exc:
            LOGGER.warning("Failed to query actor '%s': %s", actor_name, exc)
    
    return positions


def print_actor_positions(actor_names: List[str]) -> None:
    """Print positions of actors (for debugging/setup).
    
    Args:
        actor_names: List of actor names to query and print
    """
    try:
        positions = get_actor_positions(actor_names)
        
        LOGGER.info("Actor Positions:")
        for name, pos in positions.items():
            LOGGER.info("  %s: x=%.2f, y=%.2f, z=%.2f", 
                       name, pos.x_val, pos.y_val, pos.z_val)
    except ConnectionError as exc:
        LOGGER.error(str(exc))
        sys.exit(1)


# ============================================================================
# CLI Entry Points
# ============================================================================

def main_system_check():
    """CLI entry point for system readiness check."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    is_ready = check_system_readiness()
    sys.exit(0 if is_ready else 1)


def main_get_coords():
    """CLI entry point for getting UE actor coordinates."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Get Unreal Engine actor positions via AirSim"
    )
    parser.add_argument(
        'actors',
        nargs='+',
        help='Actor names to query (e.g., DroneSpawn TargetSpawn_1)'
    )
    parser.add_argument('--ip', default='127.0.0.1', help='AirSim IP')
    parser.add_argument('--port', type=int, default=41451, help='AirSim port')
    
    args = parser.parse_args()
    print_actor_positions(args.actors)


def main_convert_coords():
    """CLI entry point for coordinate conversion."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Convert UE coordinates (cm, Z-up) to AirSim NED (m, Z-down)"
    )
    parser.add_argument('--scenario', required=True, help='Input scenario JSON')
    parser.add_argument('--playerstart-xcm', type=float, default=0.0)
    parser.add_argument('--playerstart-ycm', type=float, default=0.0)
    parser.add_argument('--playerstart-zcm', type=float, default=0.0)
    parser.add_argument('--out', required=True, help='Output JSON path')
    
    args = parser.parse_args()
    
    playerstart_cm = {
        "x": args.playerstart_xcm,
        "y": args.playerstart_ycm,
        "z": args.playerstart_zcm
    }
    
    convert_scenario_ue_to_ned(args.scenario, playerstart_cm, args.out)


if __name__ == "__main__":
    # Default to system check if run directly
    main_system_check()
