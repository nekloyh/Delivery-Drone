import airsim
import time

# Connect to AirSim running locally (Conda environment)
client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
client.confirmConnection()
print("✓ Connected to AirSim")

# Enable API
client.enableApiControl(True)
client.armDisarm(True)
print("✓ Armed and API enabled")

# Get initial state
state = client.getMultirotorState()
pos = state.kinematics_estimated.position
print(f"✓ Drone position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

# Hover for 5 seconds
print("Testing hover...")
client.hoverAsync().join()
time.sleep(5)

# Check stability
state = client.getMultirotorState()
vel = state.kinematics_estimated.linear_velocity
speed = (vel.x_val**2 + vel.y_val**2 + vel.z_val**2)**0.5
print(f"✓ Speed after 5s hover: {speed:.3f} m/s")

if speed < 0.5:
    print("✅ SUCCESS: Drone is stable!")
else:
    print("⚠️ WARNING: Drone is moving too fast!")

# Cleanup
client.reset()
