"""
train_env.launch.py — Minimal environment for RL training.

Launches only what the SB3 training script needs:
  1. Gazebo Classic headless (gzserver)
  2. TurtleBot3 Waffle state publisher + spawn

Does NOT launch Nav2, twist_mux, or adaptive planner nodes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("adaptive_planner_ros")
    tb3_gz = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros = get_package_share_directory("gazebo_ros")

    map_arg = DeclareLaunchArgument(
        "map", default_value="dense_custom",
        description="World name (without .world)",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="True",
    )

    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle")

    world_file = PathJoinSubstitution([pkg, "worlds", [LaunchConfiguration("map"), ".world"]])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world_file, "verbose": "false"}.items(),
    )

    spawn_tb3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, "launch", "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": "-2.0",
            "y_pose": "-0.5",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    return LaunchDescription([
        map_arg,
        use_sim_time_arg,
        set_tb3_model,
        gazebo,
        spawn_tb3,
    ])
