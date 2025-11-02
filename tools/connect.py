import airsim

AIRSIM_HOST = "host.docker.internal"
AIRSIM_PORT = 41451

def get_airsim_client(timeout=30):
    client = airsim.MultirotorClient(ip=AIRSIM_HOST, port=AIRSIM_PORT)
    client.confirmConnection(timeout_sec=timeout)
    return client

def test_connection():
    try:
        client = get_airsim_client()
        print("Connected to AirSim at {}:{}".format(AIRSIM_HOST, AIRSIM_PORT))
    except Exception as e:
        print("[ERROR] Failed to connect to AirSim: {}".format(e))