# ==========================================================
# ROS 2 Humble + CUDA 12.3 + PyTorch 2.3 + AirSim + RL Stack
# ==========================================================
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ----------------------------------------------------------
# Base system and locales
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales tzdata curl gnupg2 lsb-release ca-certificates \
    build-essential cmake git wget python3 python3-pip python3-dev \
    libopencv-dev libeigen3-dev libboost-all-dev \
 && locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# ROS 2 Humble
# ----------------------------------------------------------
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop ros-dev-tools \
    ros-humble-cv-bridge ros-humble-image-transport ros-humble-vision-msgs \
    python3-colcon-common-extensions \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Python dependencies
# ----------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade \
    "pip==24.2" \
    "setuptools<80" \
    "wheel<0.44" \
    "packaging>=23.1,<25"

# (1) Core scientific stack
RUN python3 -m pip install --no-cache-dir \
    "numpy==1.26.4" "pandas" "pyzmq" "transforms3d" \
    "matplotlib==3.8.4" "opencv-python-headless==4.9.0.80" \
    "tqdm" "rich" "seaborn" "scikit-learn"

# (2) MessagePack dependencies (must be before AirSim)
RUN python3 -m pip install --no-cache-dir \
    "msgpack==0.6.2" "msgpack-rpc-python==0.4.1"

# (3) AirSim client
RUN python3 -m pip install --no-cache-dir "airsim==1.8.1"

# (4) PyTorch + RL stack
RUN python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0" \
 && python3 -m pip install --no-cache-dir \
    "gymnasium==0.29.1" "stable-baselines3[extra]==2.3.2" "tensorboard" "psutil"

# ----------------------------------------------------------
# ROS 2 sourcing
# ----------------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

WORKDIR /workspace
CMD ["/bin/bash"]
