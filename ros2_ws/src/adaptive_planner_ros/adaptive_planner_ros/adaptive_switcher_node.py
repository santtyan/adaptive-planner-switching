"""
adaptive_switcher_node.py — ROS2 node implementing the density-based FSM.

FSM states:
    IDLE            → waiting for first costmap and goal
    NAV2_ACTIVE     → Nav2 is driving, RL is suppressed
    RL_ACTIVE       → RL controller is driving, Nav2 goal cancelled
    TRANSITIONING   → brief lock while switching; ignores density samples

Switching logic (evaluated at 2 Hz):
    NAV2 → RL   when ρ ≥ τ+h  AND dwell ≥ min_dwell_s
    RL   → NAV2 when ρ ≤ τ-h  AND dwell ≥ min_dwell_s

Parameters (ROS2, all overridable via --ros-args -p):
    rho_threshold   float  0.30   density switching threshold τ
    hysteresis      float  0.05   half-band h
    min_dwell_s     float  1.50   minimum time in current mode before transition
    window_m        float  2.00   density window side (passed to density_estimator)
    eval_hz         float  2.00   FSM evaluation frequency
    goal_topic      str    /goal_pose  incoming NavigateToPose goals

Topics published:
    /adaptive_planner/mode          std_msgs/String   current FSM mode
    /adaptive_planner/density       std_msgs/Float32  last ρ value
    /rl_controller/goal             geometry_msgs/PoseStamped  active goal for RL
    /cmd_vel_mux_lock               std_msgs/String   "nav2" | "rl" → twist_mux priority

Topics subscribed:
    /adaptive_planner/local_density std_msgs/Float32  from density_estimator_node
    /goal_pose                      geometry_msgs/PoseStamped  mission goal

Actions used:
    /navigate_to_pose               nav2_msgs/action/NavigateToPose
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
from nav2_msgs.action import NavigateToPose


class Mode(Enum):
    IDLE = auto()
    NAV2_ACTIVE = auto()
    RL_ACTIVE = auto()
    TRANSITIONING = auto()


class AdaptiveSwitcherNode(Node):

    def __init__(self) -> None:
        super().__init__("adaptive_switcher_node")

        # --- parameters ---
        self.declare_parameter("rho_threshold", 0.30)
        self.declare_parameter("hysteresis", 0.05)
        self.declare_parameter("min_dwell_s", 1.50)
        self.declare_parameter("eval_hz", 2.0)
        self.declare_parameter("goal_topic", "/goal_pose")

        self._tau = self.get_parameter("rho_threshold").value
        self._h = self.get_parameter("hysteresis").value
        self._min_dwell = self.get_parameter("min_dwell_s").value
        goal_topic = self.get_parameter("goal_topic").value

        # --- state ---
        self._mode = Mode.IDLE
        self._rho: float = 0.0
        self._goal: Optional[PoseStamped] = None
        self._mode_entered_at: float = time.monotonic()
        self._nav2_handle = None

        # --- action client ---
        self._nav2_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # --- publishers ---
        self._mode_pub = self.create_publisher(String, "/adaptive_planner/mode", 10)
        self._density_pub = self.create_publisher(Float32, "/adaptive_planner/density", 10)
        self._rl_goal_pub = self.create_publisher(PoseStamped, "/rl_controller/goal", 10)
        self._mux_pub = self.create_publisher(String, "/cmd_vel_mux_lock", 10)

        # --- subscribers ---
        self.create_subscription(
            Float32,
            "/adaptive_planner/local_density",
            self._density_cb,
            10,
        )
        self.create_subscription(
            PoseStamped,
            goal_topic,
            self._goal_cb,
            10,
        )

        # --- FSM timer ---
        hz = self.get_parameter("eval_hz").value
        self.create_timer(1.0 / hz, self._fsm_tick)

        self.get_logger().info(
            f"AdaptiveSwitcher ready — τ={self._tau}, h={self._h}, "
            f"dwell={self._min_dwell}s"
        )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _density_cb(self, msg: Float32) -> None:
        self._rho = msg.data
        self._density_pub.publish(msg)

    def _goal_cb(self, msg: PoseStamped) -> None:
        self.get_logger().info(
            f"New goal received at ({msg.pose.position.x:.2f}, "
            f"{msg.pose.position.y:.2f})"
        )
        self._goal = msg
        if self._mode in (Mode.IDLE, Mode.NAV2_ACTIVE):
            self._activate_nav2(msg)
        elif self._mode == Mode.RL_ACTIVE:
            # Update RL goal without switching mode
            self._rl_goal_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # FSM tick
    # ------------------------------------------------------------------ #

    def _fsm_tick(self) -> None:
        if self._mode == Mode.IDLE or self._goal is None:
            return

        dwell = time.monotonic() - self._mode_entered_at

        if self._mode == Mode.NAV2_ACTIVE:
            if self._rho >= self._tau + self._h and dwell >= self._min_dwell:
                self.get_logger().info(
                    f"[FSM] NAV2→RL  ρ={self._rho:.3f} ≥ {self._tau+self._h:.3f}"
                )
                self._transition_to_rl()

        elif self._mode == Mode.RL_ACTIVE:
            if self._rho <= self._tau - self._h and dwell >= self._min_dwell:
                self.get_logger().info(
                    f"[FSM] RL→NAV2  ρ={self._rho:.3f} ≤ {self._tau-self._h:.3f}"
                )
                self._transition_to_nav2()

        # Publish current mode
        self._mode_pub.publish(String(data=self._mode.name))

    # ------------------------------------------------------------------ #
    # Transitions
    # ------------------------------------------------------------------ #

    def _activate_nav2(self, goal: PoseStamped) -> None:
        """Send goal to Nav2 and switch to NAV2_ACTIVE."""
        if not self._nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warning("Nav2 action server not available")
            return

        req = NavigateToPose.Goal()
        req.pose = goal

        send_future = self._nav2_client.send_goal_async(req)
        send_future.add_done_callback(self._nav2_goal_response_cb)

        self._set_mode(Mode.NAV2_ACTIVE)
        self._set_mux("nav2")

    def _nav2_goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Nav2 rejected goal")
            return
        self._nav2_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav2_result_cb)

    def _nav2_result_cb(self, future) -> None:
        result = future.result()
        if result and self._mode == Mode.NAV2_ACTIVE:
            self.get_logger().info("Nav2 reached goal — returning to IDLE")
            self._set_mode(Mode.IDLE)
            self._goal = None

    def _transition_to_rl(self) -> None:
        self._set_mode(Mode.TRANSITIONING)

        # Cancel Nav2 goal synchronously before handing off
        if self._nav2_handle is not None:
            cancel_future = self._nav2_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0)
            self._nav2_handle = None

        # Hand goal to RL controller
        if self._goal is not None:
            self._rl_goal_pub.publish(self._goal)

        self._set_mux("rl")
        self._set_mode(Mode.RL_ACTIVE)

    def _transition_to_nav2(self) -> None:
        self._set_mode(Mode.TRANSITIONING)
        self._set_mux("nav2")

        if self._goal is not None:
            self._activate_nav2(self._goal)
        else:
            self._set_mode(Mode.IDLE)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_mode(self, mode: Mode) -> None:
        self._mode = mode
        self._mode_entered_at = time.monotonic()

    def _set_mux(self, source: str) -> None:
        """Publish to /cmd_vel_mux_lock so twist_mux knows which source to allow."""
        self._mux_pub.publish(String(data=source))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdaptiveSwitcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
