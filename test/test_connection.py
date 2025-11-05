import airsim
import numpy as np
import time

client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
client.confirmConnection()

drone_init = client.simGetObjectPose("DroneSpawn")
drone_pose = drone_init.position

goal_spawn = ["Landing_202", "Landing_102", "Landing_201"]
goal_init = client.simGetObjectPose(goal_spawn[0])
goal_pose = goal_init.position


client.enableApiControl(True)
client.armDisarm(True)

start = np.array([drone_pose.x_val, drone_pose.y_val, drone_pose.z_val])
print("initial pos", drone_pose)
goal = np.array([goal_pose.x_val, goal_pose.y_val, goal_pose.z_val])
print("goal pos", goal_pose)

pose = airsim.Pose(airsim.Vector3r(*start), airsim.to_quaternion(0, 0, 0))
client.simSetVehiclePose(pose, ignore_collision=True)
time.sleep(0.5)
client.takeoffAsync().join()

n_steps = 50
path = np.linspace(start, goal, n_steps)
recorded_positions = []

for p in path:
    client.moveToPositionAsync(p[0], p[1], p[2], velocity=2).join()
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    recorded_positions.append([pos.x_val, pos.y_val, pos.z_val])
    print(f"Drone at: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

# --- Hạ cánh và kết thúc ---
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)