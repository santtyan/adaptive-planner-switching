"""
multi_agent_demo.launch.py — 3 TurtleBot3 Waffle em multi_agent_dense.world.

Cada robô tem namespace isolado (/robot1, /robot2, /robot3) com:
  - robot_state_publisher próprio (URDF remapeado)
  - Nav2 completo (AMCL + SmacPlanner2D + DWB)
  - density_estimator_node (lê costmap local do namespace)
  - rl_controller_node (carrega o SAC treinado)
  - adaptive_switcher_node (rho*=0.30)

Posições de spawn (zonas livres de obstáculos):
  robot1: (-2.5, -2.5)  canto SW
  robot2: ( 2.5, -2.5)  canto SE
  robot3: ( 0.0,  2.5)  centro N

Uso (após SAC convergir):
  ros2 launch adaptive_planner_ros multi_agent_demo.launch.py \
      rl_model:=/workspace/models/best_model.zip

BLOQUEIO: aguardar convergência do SAC (ETA 21/06/2026) antes de rodar.
O CBS coordinator é lançado separadamente via:
  python3 eval/cbs_multiagent_coordinator.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


ROBOTS = [
    {"name": "robot1", "x":  -2.5, "y": -2.5, "yaw": 0.0},
    {"name": "robot2", "x":   2.5, "y": -2.5, "yaw": 3.14159},
    {"name": "robot3", "x":   0.0, "y":  2.5, "yaw": -1.5708},
]


def make_robot_group(robot: dict, pkg: str, nav2_bringup: str,
                     rl_model, use_sim_time):
    """Retorna GroupAction com namespace isolado para um robô."""
    ns   = robot["name"]
    xp   = str(robot["x"])
    yp   = str(robot["y"])
    yaw  = str(robot["yaw"])

    # URDF do TurtleBot3 Waffle
    urdf_file = os.path.join(
        get_package_share_directory("turtlebot3_description"),
        "urdf", "turtlebot3_waffle.urdf"
    )
    with open(urdf_file, "r") as f:
        robot_desc = f.read()

    nodes = [
        # Robot State Publisher com namespace
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{
                "robot_description": robot_desc,
                "use_sim_time": use_sim_time,
            }],
            remappings=[
                ("/tf",        "tf"),
                ("/tf_static", "tf_static"),
            ],
        ),

        # Spawn no Gazebo
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            name=f"spawn_{ns}",
            arguments=[
                "-entity",    ns,
                "-topic",     f"/{ns}/robot_description",
                "-x",         xp,
                "-y",         yp,
                "-z",         "0.01",
                "-Y",         yaw,
            ],
            output="screen",
        ),

        # Nav2 — params com namespace embutido
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, "launch", "bringup_launch.py")
            ),
            launch_arguments={
                "use_sim_time":  str(use_sim_time),
                "params_file":   os.path.join(pkg, "config", "nav2_params.yaml"),
                "autostart":     "True",
                "namespace":     ns,
                "use_namespace": "True",
                "map":           "",   # AMCL sem mapa estático (usa costmap)
            }.items(),
        ),

        # Density estimator (lê /{ns}/local_costmap/costmap)
        Node(
            package="adaptive_planner_ros",
            executable="density_estimator_node",
            name="density_estimator_node",
            parameters=[{
                "use_sim_time":       use_sim_time,
                "window_m":           2.0,
                "occ_threshold":      65,
                "unknown_as_occupied": True,
            }],
        ),

        # RL controller (SAC)
        Node(
            package="adaptive_planner_ros",
            executable="rl_controller_node",
            name="rl_controller_node",
            parameters=[{
                "use_sim_time": use_sim_time,
                "model_path":   rl_model,
                "algo":         "sac",
            }],
        ),

        # Adaptive switcher (rho*=0.30)
        Node(
            package="adaptive_planner_ros",
            executable="adaptive_switcher_node",
            name="adaptive_switcher_node",
            parameters=[{
                "use_sim_time":  use_sim_time,
                "rho_threshold": 0.30,
                "hysteresis":    0.05,
                "min_dwell_s":   1.50,
            }],
        ),
    ]

    return GroupAction([PushRosNamespace(ns)] + nodes)


def generate_launch_description():
    pkg        = get_package_share_directory("adaptive_planner_ros")
    nav2_bringup = get_package_share_directory("nav2_bringup")

    rl_model_arg = DeclareLaunchArgument(
        "rl_model",
        default_value="/workspace/models/best_model.zip",
        description="Caminho para o modelo SAC treinado (.zip)",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="True",
    )

    rl_model     = LaunchConfiguration("rl_model")
    use_sim_time = LaunchConfiguration("use_sim_time")

    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle")

    # Gazebo com o mundo multi-agente
    world_file = os.path.join(pkg, "worlds", "multi_agent_dense.world")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"),
                "launch", "gazebo.launch.py",
            )
        ),
        launch_arguments={"world": world_file, "verbose": "false"}.items(),
    )

    robot_groups = [
        make_robot_group(r, pkg, nav2_bringup, rl_model, use_sim_time)
        for r in ROBOTS
    ]

    return LaunchDescription([
        rl_model_arg,
        use_sim_time_arg,
        set_tb3_model,
        gazebo,
        *robot_groups,
    ])
