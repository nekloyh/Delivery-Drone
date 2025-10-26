
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir

def generate_launch_description() -> LaunchDescription:

    airsim_launch = PythonLaunchDescriptionSource([
        os.path.join(os.path.expanduser('~'), 'ros_ws', 'install', 'airsim_ros_pkgs', 'share', 'airsim_ros_pkgs', 'launch'),
        '/airsim_node.launch.py'
    ])

    orb_slam_launch = PythonLaunchDescriptionSource([
        os.path.join(os.path.expanduser('~'), 'ros_ws', 'install', 'orb_slam3_ros', 'share', 'orb_slam3_ros', 'launch'),
        '/stereo_inertial.launch.py'
    ])
    slam_params = os.path.join(
        os.path.dirname(__file__), '..', 'slam_configs', 'stereo_inertial.yaml'
    )

    airsim_node = IncludeLaunchDescription(
        airsim_launch,
        launch_arguments={

            'stereo': 'true',
            'imu': 'true'
        }.items(),
    )

    slam_node = IncludeLaunchDescription(
        orb_slam_launch,
        launch_arguments={
            'params': slam_params
        }.items(),
    )

    feature_bridge = ExecuteProcess(
        cmd=['python3', 'bridges/feature_bridge.py'],
        cwd=os.path.join(os.getcwd(), '..', '..'),
        output='screen'
    )

    return LaunchDescription([
        airsim_node,
        slam_node,
        feature_bridge,
    ])
