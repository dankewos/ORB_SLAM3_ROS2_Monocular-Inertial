from datetime import datetime

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
    Command,
)
from launch.conditions import IfCondition
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
)


def generate_launch_description():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    bag_name = f"ORB_SLAM3_{current_time}"

    #use_sim_time_param = {'use_sim_time': True}
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Whether to launch RViz.",
            ),
            DeclareLaunchArgument(
                "ip_address",
                default_value="172.17.0.3",
                description="The IP address of the luci docker container",
            ),
            DeclareLaunchArgument(
                "remote_launch",
                default_value="false",
                description="Whether or not to launch the luci launch file on\
                        the docker container",
            ),
            DeclareLaunchArgument(
                "sensor_type",
                default_value="imu-monocular",
                description="The mode which ORB_SLAM3 will run in.",
            ),
            DeclareLaunchArgument(
                "use_pangolin",
                default_value="true",
                description="Whether to use Pangolin for visualization.",
            ),
            DeclareLaunchArgument(
                "record_bag",
                default_value="false",
                description="Whether or not to record a rosbag.",
            ),
            DeclareLaunchArgument(
                "bag_name",
                default_value=bag_name,
                description="The name of the bag if record_bag is true. By\
                        default, the name of the bag will be\
                        ORB_SLAM3_YYYY-MM-DD_HH-mm-ss",
            ),
            DeclareLaunchArgument(
                "dataset_root",
                default_value="/home/maria/ros2_ws/src/ORB_SLAM3_ROS2/datasets",
                description="Root folder containing TUM-VI datasets",
            ),

            DeclareLaunchArgument(
                "dataset",
                default_value="outdoors4",
                description="Which dataset folder inside dataset_root to use",
            ),
            
            DeclareLaunchArgument(
                "speed_factor",
                default_value="1.0",
                description="Speed factor for dataset playback (1.0 = real-time)",
            ),
            ExecuteProcess(
                cmd=[
                    "python3", 
                    "/home/maria/ros2_ws/src/ORB_SLAM3_ROS2/ORB_SLAM3_ROS2/dataset_loader.py",
                    "/home/maria/ros2_ws/src/ORB_SLAM3_ROS2/datasets",
                    "outdoors4",
                    #"--ros-args", "-p", "use_sim_time:=true"
                ],
                output="screen",
            ),

            Node(
                package="orb_slam3_ros2",
                executable="imu_mono_node_cpp",
                output="screen",
                #prefix="xterm -e gdb -ex run --args",
                parameters=[
                    {
                        "sensor_type": LaunchConfiguration("sensor_type"),
                        "use_pangolin": LaunchConfiguration("use_pangolin"),
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
                #parameters=[use_sim_time_param],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                #parameters=[use_sim_time_param],
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [
                            FindPackageShare("orb_slam3_ros2"),
                            "config",
                            "point_cloud.rviz",
                        ]
                    ),
                ],
                condition=IfCondition(
                    PythonExpression(
                        ["'", LaunchConfiguration("use_rviz"), "' == 'true'"]
                    )
                ),
            ),
        ],
    )
