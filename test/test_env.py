# check_env.py - Kiểm tra toàn diện môi trường drone-env
import sys
import platform
import torch
import airsim

def print_header(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ")
    print("=" * 60)


def check_import(module_name, alias=None):
    try:
        if alias:
            exec(f"import {module_name} as {alias}")
            print(f"✓ {module_name} (as {alias}) → OK")
        else:
            exec(f"import {module_name}")
            print(f"✓ {module_name} → OK")
        return True
    except Exception as e:
        print(f"✗ {module_name} → LỖI: {e}")
        return False


# === BẮT ĐẦU KIỂM TRA ===
print_header("KIỂM TRA MÔI TRƯỜNG DRONE-ENV")
print(f"Python: {sys.version.split()[0]}")
print(f"Hệ điều hành: {platform.system()} {platform.release()}")

# 1. Conda packages
print_header("CONDA PACKAGES")
check_import("numpy", "np")
check_import("pandas", "pd")
check_import("matplotlib.pyplot", "plt")
check_import("seaborn", "sns")
check_import("sklearn", "sk")
check_import("cv2")  # OpenCV
check_import("torch")

# 2. PyTorch + CUDA
print_header("PYTORCH & CUDA")
if check_import("torch"):
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")

# 3. Pip packages
print_header("PIP PACKAGES")
check_import("gymnasium", "gym")
check_import("stable_baselines3", "sb3")
check_import("tensorboard")
check_import("msgpack")
check_import("transforms3d")

# 4. AirSim - Quan trọng nhất!
print_header("AIRSIM CLIENT")
airsim_ok = check_import("airsim")
if airsim_ok:
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("   Kết nối AirSim server: OK (chưa cần server chạy)")
    except Exception as e:
        print(
            f"   Cảnh báo: Chưa kết nối server → {e} (bình thường nếu chưa chạy AirSim)"
        )

# 5. Tổng kết
print_header("TỔNG KẾT")
all_ok = all(
    [
        check_import("numpy"),
        check_import("torch"),
        check_import("cv2"),
        check_import("gymnasium"),
        check_import("airsim"),
    ]
)
if all_ok:
    print("TOÀN BỘ MÔI TRƯỜNG ĐÃ HOÀN TOÀN ỔN ĐỊNH!")
    print("Bạn có thể bắt đầu code drone RL rồi!")
else:
    print("Còn lỗi! Xem chi tiết ở trên.")
