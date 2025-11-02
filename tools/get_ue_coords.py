import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

# Get spawn point
spawn_pose = client.simGetObjectPose("DroneSpawn")
print(f"DroneSpawn: {spawn_pose.position}")

# Get targets
for i in range(1, 4):
    target_pose = client.simGetObjectPose(f"TargetSpawn_{i}")
    print(f"TargetSpawn_{i}: {target_pose.position}")