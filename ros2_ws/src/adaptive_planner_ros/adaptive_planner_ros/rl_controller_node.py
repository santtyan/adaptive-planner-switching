"""
rl_controller_node.py — loads a SB3 model and publishes /cmd_vel_rl.

Parameters:
    model_path   str   path to .zip SB3 model (required)
    algo         str   "ppo" | "sac"  (default: "sac")
    goal_topic   str   "/rl_controller/goal"

Subscribes: /scan, /goal_pose (or goal_topic), /gazebo/model_states
Publishes:  /cmd_vel_rl  geometry_msgs/Twist
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates

from adaptive_planner_ros.obs_utils import (
    make_observation,
    LINEAR_VEL_MAX,
    ANGULAR_VEL_MAX,
    LIDAR_MAX_RANGE,
    OBS_DIM,
)

ROBOT_NAME = "turtlebot3_waffle"


def _quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RLControllerNode(Node):

    def __init__(self) -> None:
        super().__init__("rl_controller_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("algo", "sac")
        self.declare_parameter("goal_topic", "/rl_controller/goal")

        model_path = self.get_parameter("model_path").value
        algo = self.get_parameter("algo").value.lower()
        goal_topic = self.get_parameter("goal_topic").value

        if not model_path or not os.path.exists(model_path):
            self.get_logger().error(f"model_path not found: '{model_path}'")
            raise FileNotFoundError(model_path)

        # Load model
        if algo == "sac":
            from stable_baselines3 import SAC
            self._model = SAC.load(model_path)
        elif algo == "ppo":
            from stable_baselines3 import PPO
            self._model = PPO.load(model_path)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        self.get_logger().info(f"Loaded {algo.upper()} model: {model_path}")

        # State
        self._scan: Optional[LaserScan] = None
        self._model_states: Optional[ModelStates] = None
        self._goal: Optional[PoseStamped] = None
        self._active = False  # only publish cmd_vel when activated

        # Publishers / subscribers
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel_rl", 10)

        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(ModelStates, "/gazebo/model_states",
                                 self._model_states_cb, 10)
        self.create_subscription(PoseStamped, goal_topic,
                                 self._goal_cb, 10)
        # Listen for mux lock to know when we should publish
        from std_msgs.msg import String
        self.create_subscription(String, "/cmd_vel_mux_lock",
                                 self._mux_lock_cb, 10)

        self.create_timer(1.0 / 5.0, self._control_tick)  # 5 Hz

    # ------------------------------------------------------------------ #

    def _scan_cb(self, msg): self._scan = msg
    def _model_states_cb(self, msg): self._model_states = msg
    def _goal_cb(self, msg): self._goal = msg
    def _mux_lock_cb(self, msg): self._active = (msg.data == "rl")

    def _control_tick(self) -> None:
        if not self._active or self._goal is None:
            return

        ranges = (list(self._scan.ranges)
                  if self._scan is not None
                  else [LIDAR_MAX_RANGE] * 360)

        x, y, yaw = self._get_pose()
        gx = self._goal.pose.position.x
        gy = self._goal.pose.position.y

        obs = make_observation(ranges, x, y, yaw, gx, gy)
        action, _ = self._model.predict(obs, deterministic=True)

        v = float(action[0]) * LINEAR_VEL_MAX
        omega = float(action[1]) * ANGULAR_VEL_MAX

        msg = Twist()
        msg.linear.x = float(np.clip(v, -LINEAR_VEL_MAX, LINEAR_VEL_MAX))
        msg.angular.z = float(np.clip(omega, -ANGULAR_VEL_MAX, ANGULAR_VEL_MAX))
        self._cmd_pub.publish(msg)

    def _get_pose(self):
        if self._model_states is None:
            return 0.0, 0.0, 0.0
        try:
            idx = self._model_states.name.index(ROBOT_NAME)
        except ValueError:
            return 0.0, 0.0, 0.0
        p = self._model_states.pose[idx]
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y,
                            p.orientation.z, p.orientation.w)
        return p.position.x, p.position.y, yaw


def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
