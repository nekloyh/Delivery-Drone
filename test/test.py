import time
from envs.triq_indoor_drone_env import IndoorDroneEnv
import yaml

cfg = yaml.safe_load(open("configs/indoor.yaml"))
env = IndoorDroneEnv(cfg["env"])

print("Testing reset...")
obs, info = env.reset()
print(f"Spawn: {info['spawn_position']}")

print("\nTesting hover (10 steps with zero action)...")
for i in range(10):
    obs, reward, done, trunc, info = env.step([0, 0, 0, 0])
    print(f"Step {i+1}: stable={info['drone_stable']}, z={obs[2]:.2f}")
    if done:
        break

print("\nTesting forward movement...")
for i in range(10):
    obs, reward, done, trunc, info = env.step([0.5, 0, 0, 0])  # Move forward
    print(f"Step {i+1}: x={obs[0]:.2f}, stable={info['drone_stable']}")
    if done:
        break

env.close()
print("✓ Test completed!")
