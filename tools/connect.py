import airsim

# Default AirSim connection settings (for local Conda environment)
AIRSIM_HOST = "127.0.0.1"  # localhost for Conda environment
AIRSIM_PORT = 41451

def get_airsim_client(timeout=30):
    """Connect to AirSim running locally on the host machine.
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        Connected AirSim MultirotorClient
    """
    client = airsim.MultirotorClient(ip=AIRSIM_HOST, port=AIRSIM_PORT)
    client.confirmConnection(timeout_sec=timeout)
    return client

def test_connection():
    try:
        client = get_airsim_client()
        print("Connected to AirSim at {}:{}".format(AIRSIM_HOST, AIRSIM_PORT))
    except Exception as e:
        print("[ERROR] Failed to connect to AirSim: {}".format(e))