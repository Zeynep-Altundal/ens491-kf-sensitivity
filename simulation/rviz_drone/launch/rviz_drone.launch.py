"""Start the quadrotor simulation node and, unless rviz:=false, RViz with the package configuration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("rviz_drone")
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true", description="start RViz"),
        DeclareLaunchArgument("seed", default_value="0", description="seed of the sensor errors"),
        Node(
            package="rviz_drone",
            executable="sim_node",
            name="rviz_drone_sim",
            output="screen",
            parameters=[{"seed": ParameterValue(LaunchConfiguration("seed"), value_type=int)}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "rviz_drone.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
