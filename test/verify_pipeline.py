"""
Quick verification script for production pipeline.
Tests:
1. AirSim connection
2. Target detection
3. Drone0 and DroneSpawn existence
4. Curriculum config loading
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)
    
    try:
        import airsim
        print("✅ airsim imported")
    except ImportError as e:
        print(f"❌ Failed to import airsim: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch imported (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"✅ stable_baselines3 imported (version: {stable_baselines3.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import stable_baselines3: {e}")
        return False
    
    try:
        import gymnasium
        print(f"✅ gymnasium imported (version: {gymnasium.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import gymnasium: {e}")
        return False
    
    print("✅ All imports successful!\n")
    return True


def test_curriculum_config():
    """Test curriculum configuration loading."""
    print("=" * 60)
    print("TEST 2: Loading curriculum config...")
    print("=" * 60)
    
    config_path = Path(__file__).parent.parent / "configs" / "curriculum_config.json"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded successfully")
        
        # Validate structure
        if "curriculum" not in config:
            print("❌ Missing 'curriculum' key")
            return False
        
        if "global_config" not in config:
            print("❌ Missing 'global_config' key")
            return False
        
        stages = config["curriculum"]
        print(f"✅ Found {len(stages)} curriculum stages:")
        
        total_targets = set()
        for stage_name, stage_data in stages.items():
            targets = stage_data.get("targets", [])
            total_targets.update(targets)
            print(f"   - {stage_name}: {len(targets)} targets (success: {stage_data.get('success_threshold', 0):.0%})")
        
        print(f"\n✅ Total unique targets across all stages: {len(total_targets)}")
        
        # Validate global config
        gc = config["global_config"]
        print(f"\n✅ Global config:")
        print(f"   - Drone: {gc.get('drone_name', 'N/A')}")
        print(f"   - Spawn: {gc.get('spawn_actor_name', 'N/A')}")
        print(f"   - Location: {gc.get('spawn_location_ue', 'N/A')}")
        
        print("✅ Curriculum config valid!\n")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False


def test_airsim_connection():
    """Test AirSim connection and scene objects."""
    print("=" * 60)
    print("TEST 3: Testing AirSim connection...")
    print("=" * 60)
    
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ Connected to AirSim successfully")
        
        # Check if simulation is paused
        is_paused = client.simIsPause()
        print(f"   Simulation paused: {is_paused}")
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to connect to AirSim: {e}")
        print("\nTroubleshooting:")
        print("1. Is Unreal Engine project running?")
        print("2. Is AirSim plugin enabled?")
        print("3. Press 'Play' button in UE4 Editor")
        print("4. Check settings.json in Documents/AirSim")
        return None


def test_scene_objects(client):
    """Test scene objects - targets and spawn points."""
    if client is None:
        print("⏭️  Skipping scene object test (no connection)\n")
        return False
    
    print("=" * 60)
    print("TEST 4: Checking scene objects...")
    print("=" * 60)
    
    try:
        # Check for targets
        all_targets = client.simListSceneObjects("Landing_*")
        print(f"✅ Found {len(all_targets)} Landing_* targets in scene:")
        
        if len(all_targets) == 0:
            print("❌ No Landing_* targets found!")
            print("\nExpected targets:")
            print("  Floor 1: Landing_101, Landing_102, ..., Landing_106")
            print("  Floor 2: Landing_201, Landing_202, ..., Landing_206")
            print("  Floor 3: Landing_301, Landing_302, ..., Landing_306")
            print("  Floor 4: Landing_401, Landing_402, ..., Landing_406")
            print("  Floor 5: Landing_501, Landing_502, ..., Landing_506")
            return False
        
        # Group by floor
        floors = {}
        for target in all_targets:
            if len(target) >= 11:  # "Landing_XYZ"
                floor = target.split('_')[1][0]  # First digit
                if floor not in floors:
                    floors[floor] = []
                floors[floor].append(target)
        
        for floor in sorted(floors.keys()):
            print(f"   Floor {floor}: {len(floors[floor])} targets")
        
        # Check specific required targets
        required_targets = [
            "Landing_101", "Landing_102", "Landing_103", 
            "Landing_104", "Landing_105", "Landing_106",
        ]
        
        missing = [t for t in required_targets if t not in all_targets]
        if missing:
            print(f"\n⚠️  Warning: Missing some Floor 1 targets: {missing}")
        else:
            print(f"\n✅ All Floor 1 targets present (101-106)")
        
        # Check for DroneSpawn
        print("\n" + "=" * 60)
        print("TEST 5: Checking spawn points...")
        print("=" * 60)
        
        all_spawns = client.simListSceneObjects("*Spawn*")
        print(f"✅ Found {len(all_spawns)} spawn points: {all_spawns}")
        
        if "DroneSpawn" in all_spawns:
            print("✅ DroneSpawn actor found!")
            
            # Try to get pose
            try:
                pose = client.simGetObjectPose("DroneSpawn")
                pos = pose.position
                print(f"   Position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
            except Exception as e:
                print(f"   ⚠️  Could not get DroneSpawn pose: {e}")
        else:
            print("❌ DroneSpawn actor not found!")
            print("   Make sure you have an actor named 'DroneSpawn' in your UE map")
        
        # Check for Drone0
        all_vehicles = client.listVehicles()
        print(f"\n✅ Found {len(all_vehicles)} vehicle(s): {all_vehicles}")
        
        if "Drone0" in all_vehicles:
            print("✅ Drone0 vehicle found!")
        else:
            print("❌ Drone0 not found!")
            print("   Check settings.json - vehicle name should be 'Drone0'")
        
        print("\n✅ Scene verification complete!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error checking scene objects: {e}")
        return False


def test_environment_creation():
    """Test environment creation (without running)."""
    print("=" * 60)
    print("TEST 6: Testing environment creation...")
    print("=" * 60)
    
    try:
        # Add parent directory to path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from envs.indoor_drone_env import IndoorDroneEnv
        
        print("✅ IndoorDroneEnv imported successfully")
        
        # Try to create environment (don't step)
        env_config = {
            "airsim_ip": "127.0.0.1",
            "spawn_actor_name": "DroneSpawn",
            "target_actor_names": ["Landing_101"],  # Single target for test
            "max_steps": 100,
        }
        
        print("✅ Creating environment with Landing_101...")
        env = IndoorDroneEnv(**env_config)
        print("✅ Environment created successfully!")
        
        # Check observation and action spaces
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        env.close()
        print("✅ Environment test complete!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚁 PRODUCTION PIPELINE VERIFICATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Config
    results.append(("Curriculum Config", test_curriculum_config()))
    
    # Test 3: AirSim Connection
    client = test_airsim_connection()
    results.append(("AirSim Connection", client is not None))
    
    # Test 4-5: Scene Objects (only if connected)
    if client:
        results.append(("Scene Objects", test_scene_objects(client)))
    else:
        results.append(("Scene Objects", False))
    
    # Test 6: Environment Creation
    results.append(("Environment Creation", test_environment_creation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! Ready to start training:")
        print("   python training/train_ppo_curriculum_v2.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        print("   See troubleshooting steps above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
