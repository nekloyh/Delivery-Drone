import airsim

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()

# Liệt kê tất cả object trong scene
objs = client.simListSceneObjects()
for i, o in enumerate(objs):
    print(i, o)