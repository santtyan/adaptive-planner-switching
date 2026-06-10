"""
demo.launch.py — Full adaptive planner pipeline on TurtleBot3 Waffle.

Launches:
  1. Gazebo Classic (gzserver headless or gzclient for GUI)
  2. TurtleBot3 robot state publisher + spawn
  3. Nav2 bringup (AMCL + SmacPlanner2D + DWB)
  4. twist_mux
  5. density_estimator_node
  6. rl_controller_node  (requires rl_model arg)
  7. adaptive_switcher_node

Usage:
  # Headless (training / benchmark):
  ros2 launch adaptive_planner_ros demo.launch.py \
      rl_model:=models/sac_42_500k.zip map:=dense_custom headless:=True

  # GUI (demo / debugging):
  ros2 launch adaptive_planner_ros demo.launch.py \
      rl_model:=models/sac_42_500k.zip
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("adaptive_planner_ros")
    tb3_gz = get_package_share_directory("turtlebot3_gazebo")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    rl_model_arg = DeclareLaunchArgument(
        "rl_model",
        default_value=os.path.join(pkg, "models", "sac_42_500k.zip"),
        description="Path to SB3 .zip model file",
    )
    algo_arg = DeclareLaunchArgument(
        "algo", default_value="sac",
        description="RL algorithm: ppo | sac",
    )
    map_arg = DeclareLaunchArgument(
        "map", default_value="dense_custom",
        description="World name (without .world): dense_custom | turtlebot3_world",
    )
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="False",
        description="Run Gazebo without GUI (gzserver only)",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="True",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    rl_model = LaunchConfiguration("rl_model")
    algo = LaunchConfiguration("algo")
    headless = LaunchConfiguration("headless")

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle")

    # ------------------------------------------------------------------ #
    # Gazebo
    # ------------------------------------------------------------------ #
    world_file = PathJoinSubstitution([pkg, "worlds",
                                       [LaunchConfiguration("map"), ".world"]])

    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"),
                         "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_file,
                          "verbose": "false"}.items(),
        condition=IfCondition(headless),
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"),
                         "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file,
                          "verbose": "false"}.items(),
        condition=UnlessCondition(headless),
    )

    # ------------------------------------------------------------------ #
    # TurtleBot3 state publisher + spawn
    # ------------------------------------------------------------------ #
    tb3_state_pub = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz, "launch",
                         "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={
            "x_pose": "-2.0",
            "y_pose": "-0.5",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # Nav2
    # ------------------------------------------------------------------ #
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": os.path.join(pkg, "config", "nav2_params.yaml"),
            "autostart": "True",
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # twist_mux
    # ------------------------------------------------------------------ #
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        name="twist_mux",
        parameters=[
            os.path.join(pkg, "config", "twist_mux.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("cmd_vel_out", "/cmd_vel")],
    )

    # ------------------------------------------------------------------ #
    # Density estimator
    # ------------------------------------------------------------------ #
    density_estimator = Node(
        package="adaptive_planner_ros",
        executable="density_estimator_node",
        name="density_estimator_node",
        parameters=[{"use_sim_time": use_sim_time,
                     "window_m": 2.0,
                     "occ_threshold": 65,
                     "unknown_as_occupied": True}],
    )

    # ------------------------------------------------------------------ #
    # RL controller
    # ------------------------------------------------------------------ #
    rl_controller = Node(
        package="adaptive_planner_ros",
        executable="rl_controller_node",
        name="rl_controller_node",
        parameters=[{"use_sim_time": use_sim_time,
                     "model_path": rl_model,
                     "algo": algo}],
    )

    # ------------------------------------------------------------------ #
    # Adaptive switcher
    # ------------------------------------------------------------------ #
    adaptive_switcher = Node(
        package="adaptive_planner_ros",
        executable="adaptive_switcher_node",
        name="adaptive_switcher_node",
        parameters=[{"use_sim_time": use_sim_time,
                     "rho_threshold": 0.30,
                     "hysteresis": 0.05,
                     "min_dwell_s": 1.50}],
    )

    return LaunchDescription([
        # args
        rl_model_arg, algo_arg, map_arg, headless_arg, use_sim_time_arg,
        # env
        set_tb3_model,
        # nodes (order matters for dependency on Gazebo clock)
        gazebo_headless,
        gazebo_gui,
        tb3_state_pub,
        nav2,
        twist_mux,
        density_estimator,
        rl_controller,
        adaptive_switcher,
    ])
