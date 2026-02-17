from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_mock = LaunchConfiguration("use_mock")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_mock", default_value="true"),
            # Permanent Nodes
            Node(
                package="face_lock",
                executable="pi_hardware",
                parameters=[{"use_mock": use_mock}],
            ),
            Node(
                package="face_lock",
                executable="face_lock_manager",
            ),
            # Lifecycle Nodes
            LifecycleNode(
                package="face_lock",
                executable="camera",
                name="camera",
                namespace="",
            ),
            LifecycleNode(
                package="face_lock",
                executable="face_recognition",
                name="face_recognition",
                namespace="",
            ),
            LifecycleNode(
                package="face_lock",
                executable="arm_controller",
                name="arm_controller",
                namespace="",
            ),
        ]
    )
