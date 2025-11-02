import airsim

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()
print("Connected successfully!")

pose = client.simGetObjectPose("DroneSpawn")
if pose:
    client.simAddVehicle("Drone1", "SimpleFlight", pose, "")
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    client.takeoffAsync(vehicle_name="Drone1").join()
else:
    print("Actor not found, cannot spawn at actor position!")