from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("kmu26_auv_vla_data_collector")
    default_config = f"{package_share}/config/collector.yaml"
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "buoy_release_image_topic",
                default_value="/camera_release/camera/color/image_raw/compressed",
            ),
            Node(
                package="kmu26_auv_vla_data_collector",
                executable="collector",
                name="vla_data_collector",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "buoy_release_image_topic": LaunchConfiguration(
                            "buoy_release_image_topic"
                        )
                    },
                ],
            ),
        ]
    )
