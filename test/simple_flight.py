import airsim, numpy as np, time

DRONE = "Drone1"
GOAL_TAG = "LandingPad"     # Tag đã đặt cho LandingPad
HOVER_ALT = -2.0            # NED: -2 m = cao 2 m so với mặt đất

def connect():
    c = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
    c.confirmConnection()
    return c

def get_pose(c, name_or_tag):
    pose = c.simGetObjectPose(name_or_tag)
    if pose is None or np.isnan(pose.position.x_val):
        raise RuntimeError(f"Không tìm thấy actor/tag: {name_or_tag}")
    return pose

def to_vec3(pose):
    return np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=float)

def reset_to_start(c):
    # Lấy PlayerStart làm A (từ vehicle pose lần đầu) hoặc lưu lại ngay lần đầu gọi
    start_pose = c.simGetVehiclePose(vehicle_name=DRONE)
    start_pose.position.z_val = HOVER_ALT  # cao 2 m
    # Freeze sim khi reposition để tránh va chạm tức thời
    c.simPause(True)
    c.simSetVehiclePose(start_pose, ignore_collision=True, vehicle_name=DRONE)
    c.simPause(False)

    c.enableApiControl(True, vehicle_name=DRONE)
    c.armDisarm(True, vehicle_name=DRONE)
    # Đảm bảo hover ổn định
    c.takeoffAsync(vehicle_name=DRONE).join()
    c.moveToZAsync(HOVER_ALT, 3, vehicle_name=DRONE).join()
    c.hoverAsync(vehicle_name=DRONE).join()
    return start_pose

def goto(c, x, y, z, speed=5.0):
    return c.moveToPositionAsync(float(x), float(y), float(z), float(speed), vehicle_name=DRONE).join()

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def main():
    c = connect()
    print("Connected to AirSim")

    # 1) Reset & hover tại A
    print("1. Reset to start position A")
    start_pose = reset_to_start(c)
    A = to_vec3(start_pose)

    # 2) Lấy toạ độ B từ Tag (goal)
    print("2. Get goal position B from tag")
    goal_pose = get_pose(c, GOAL_TAG)
    B = to_vec3(goal_pose)
    # Chọn cao độ mục tiêu phù hợp (giữ cùng cao độ hover, hoặc cụ thể bài toán)
    B[2] = HOVER_ALT

    # 3) Bay đến B
    print("3. Flying to goal position B")
    goto(c, *B)
    if dist(to_vec3(c.simGetVehiclePose(DRONE)), B) < 1.0:
        print("Reached B")

    # (tuỳ chọn) thực hiện landing tại B:
    # c.landAsync(vehicle_name=DRONE).join()
    # ... rồi takeoff lại để quay về A

    # 4) Trở về A
    print("4. Returning to start position A")
    goto(c, *A)
    if dist(to_vec3(c.simGetVehiclePose(DRONE)), A) < 1.0:
        print("Returned to A")

    c.hoverAsync(vehicle_name=DRONE).join()

if __name__ == "__main__":
    main()
