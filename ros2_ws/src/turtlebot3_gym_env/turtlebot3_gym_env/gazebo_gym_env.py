"""
TurtleBot3 Waffle — Gymnasium environment over ROS2/Gazebo Classic.

Observation space : Box(27,)  — via obs_utils.make_observation()
Action space      : Box(2,)   — [v_norm, omega_norm] in [-1, 1]
                                mapped to v∈[-0.22,0.22] m/s, ω∈[-2.84,2.84] rad/s

Episode termination:
  - Goal reached   (distance < GOAL_RADIUS)
  - Collision      (min lidar < COLLISION_DIST)
  - Timeout        (steps > MAX_STEPS)

Reset strategy:
  - Teleport robot via /gazebo/set_entity_state  (preserves /clock — do NOT use /reset_simulation)
  - Clear global costmap via /global_costmap/clear_entirely
  - Use /gazebo/model_states for ground-truth pose during training
    (AMCL diverges after teleport; AMCL is only used during evaluation)

Domain randomisation:
  - obstacle_density sampled from [density_min, density_max] each episode
    (requires Gazebo world with a parametric obstacle spawner, or use fixed worlds)
  - start and goal positions are sampled from a free-space list

Usage:
    env = TurtleBot3GazeboEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

NOTE: ROS2 must be initialised before constructing this class.
      Call rclpy.init() once in your training script, then instantiate the env.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
import gymnasium as gym
from gymnasium import spaces

from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from nav2_msgs.srv import ClearEntireCostmap

# Import shared observation builder — MUST be the same module used in
# rl_controller_node to guarantee identical normalisation at inference time.
from adaptive_planner_ros.obs_utils import (
    make_observation,
    LINEAR_VEL_MAX,
    ANGULAR_VEL_MAX,
    OBS_DIM,
    LIDAR_MAX_RANGE,
    COLLISION_DIST,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_NAME = "turtlebot3_waffle"
CMD_VEL_TOPIC = "/cmd_vel"
SCAN_TOPIC = "/scan"
MODEL_STATES_TOPIC = "/gazebo/model_states"

GOAL_RADIUS = 0.25          # metres — episode success threshold
# COLLISION_DIST imported from obs_utils (single source of truth)
MAX_STEPS = 600             # ~120 s at 5 Hz control loop
CONTROL_HZ = 5.0            # Hz — how fast step() publishes and reads
SCAN_TIMEOUT = 1.0          # seconds — max wait for a fresh /scan message

# Reward shaping coefficients
R_APPROACH = 10.0           # reward per metre closer to goal
R_TIME = -0.02              # per step alive penalty
R_COLLISION = -20.0         # terminal collision penalty
R_GOAL = 50.0               # terminal goal reward

# Spawn candidates validated for dense_custom.world:
# - min 0.30m from cylinder surfaces (radius 0.18m)
# - min 0.30m from walls (at ±2.1m)
# - spread across the navigable area for diverse training
DEFAULT_SPAWN_CANDIDATES: List[Tuple[float, float]] = [
    ( 0.8,  1.3),  # NE quadrant, clear
    ( 1.3,  1.3),  # NE corner area
    (-0.7,  0.3),  # center-W, good clearance
    ( 0.3,  1.3),  # N center
    (-1.2,  0.3),  # W side
    ( 0.3,  0.8),  # center
    (-1.2, -0.2),  # W center
    (-0.2,  0.8),  # center-N
    ( 1.3, -1.2),  # SE area
    (-0.2,  0.3),  # center, best wall clearance
    ( 0.3,  0.3),  # center
    (-0.2, -1.2),  # S center
]

# ---------------------------------------------------------------------------
# Internal ROS2 node
# ---------------------------------------------------------------------------

class _GazeboEnvNode(Node):
    """Minimal ROS2 node that owns the publishers/subscribers for the env."""

    def __init__(self) -> None:
        super().__init__("turtlebot3_gym_env_node")

        self._scan: Optional[LaserScan] = None
        self._model_states: Optional[ModelStates] = None
        self._scan_seq: int = -1

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        # Subscribers
        self.create_subscription(LaserScan, SCAN_TOPIC,
                                 self._scan_cb, 10)
        self.create_subscription(ModelStates, MODEL_STATES_TOPIC,
                                 self._model_states_cb, 10)

        # Service clients
        self._set_entity_cli = self.create_client(SetEntityState,
                                                   "/gazebo/set_entity_state")
        self._clear_costmap_cli = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely",
        )

        self.get_logger().info("TurtleBot3GazeboEnv node initialised")

    # ---- callbacks --------------------------------------------------------

    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_seq += 1

    def _model_states_cb(self, msg: ModelStates) -> None:
        self._model_states = msg

    # ---- helpers ----------------------------------------------------------

    def publish_cmd(self, v: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x = float(np.clip(v, -LINEAR_VEL_MAX, LINEAR_VEL_MAX))
        msg.angular.z = float(np.clip(omega, -ANGULAR_VEL_MAX, ANGULAR_VEL_MAX))
        self._cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def wait_for_scan(self, timeout: float = SCAN_TIMEOUT) -> Optional[LaserScan]:
        """Spin until a new /scan message arrives (or timeout)."""
        target_seq = self._scan_seq + 1
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._scan_seq >= target_seq:
                return self._scan
        return self._scan  # return stale if timeout

    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw) from /gazebo/model_states ground truth."""
        if self._model_states is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        if self._model_states is None:
            return 0.0, 0.0, 0.0
        try:
            idx = self._model_states.name.index(ROBOT_NAME)
        except ValueError:
            return 0.0, 0.0, 0.0
        p = self._model_states.pose[idx]
        x = p.position.x
        y = p.position.y
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y,
                           p.orientation.z, p.orientation.w)
        return x, y, yaw

    def teleport_robot(self, x: float, y: float, yaw: float) -> bool:
        """Move robot to (x, y, yaw) via /gazebo/set_entity_state.

        Does NOT reset /clock — safe to call during training.
        Returns True on success.
        """


        req = SetEntityState.Request()
        req.state.name = ROBOT_NAME
        req.state.pose.position = Point(x=x, y=y, z=0.0)
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        req.state.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        req.state.twist.linear.x = 0.0
        req.state.twist.angular.z = 0.0
        req.state.reference_frame = "world"

        future = self._set_entity_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            return future.result().success
        return False

    def clear_costmap(self) -> None:
        """Clear the global costmap (removes stale obstacle marks after reset)."""
        if not self._clear_costmap_cli.wait_for_service(timeout_sec=1.0):
            return  # Nav2 may not be running during unit tests
        future = self._clear_costmap_cli.call_async(
            ClearEntireCostmap.Request()
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class TurtleBot3GazeboEnv(gym.Env):
    """Gymnasium environment wrapping ROS2/Gazebo for TurtleBot3 Waffle.

    Observation: 27-dim vector (24 lidar + 3 polar goal) — see obs_utils.py.
    Action:      2-dim normalised [-1,1]² → (v, ω).

    Constructor arguments:
        node            : existing _GazeboEnvNode (share across envs to avoid
                          multiple ROS2 nodes); created internally if None.
        goal_candidates : list of (x, y) tuples to sample goal/start from.
        density_range   : (min, max) obstacle density for domain randomisation.
                          Currently informational — used to select Gazebo world
                          if multiple worlds are provided.
        max_steps       : episode length limit.
        seed            : RNG seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        node: Optional[_GazeboEnvNode] = None,
        goal_candidates: Optional[List[Tuple[float, float]]] = None,
        density_range: Tuple[float, float] = (0.1, 0.5),
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self._node = node or _GazeboEnvNode()
        self._goal_candidates = goal_candidates or DEFAULT_SPAWN_CANDIDATES
        self._density_range = density_range
        self._max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._goal: Tuple[float, float] = (0.0, 0.0)
        self._prev_dist: float = 0.0
        self._last_scan: Optional[LaserScan] = None

    # ---- Gymnasium API ----------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample start and goal (must be different positions)
        start, goal = self._sample_start_goal()
        self._goal = goal

        # Teleport with collision validation — resample if spawn is inside obstacle
        SAFE_SPAWN_MARGIN = COLLISION_DIST + 0.10   # 0.25 m clearance required
        MAX_SPAWN_TRIES = len(self._goal_candidates)
        self._node.stop_robot()
        for _try in range(MAX_SPAWN_TRIES):
            self._node.teleport_robot(start[0], start[1], yaw=float(
                self._rng.uniform(-math.pi, math.pi)
            ))
            time.sleep(0.3)          # allow Gazebo physics to settle
            scan = self._node.wait_for_scan(timeout=2.0)
            if _min_scan_range(scan) >= SAFE_SPAWN_MARGIN:
                break
            # Position is in collision — try a different candidate
            idxs = self._rng.choice(len(self._goal_candidates), size=2, replace=False)
            start = self._goal_candidates[int(idxs[0])]
            goal = self._goal_candidates[int(idxs[1])]
            self._goal = goal
        self._node.clear_costmap()
        self._last_scan = scan

        obs = self._build_obs()
        x, y, _ = self._node.get_robot_pose()
        self._prev_dist = math.hypot(goal[0] - x, goal[1] - y)
        self._step_count = 0

        return obs, {"goal": goal, "start": start}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Map normalised action → physical velocities
        v = float(action[0]) * LINEAR_VEL_MAX
        omega = float(action[1]) * ANGULAR_VEL_MAX
        self._node.publish_cmd(v, omega)

        # Advance simulation one control step
        scan = self._node.wait_for_scan(timeout=SCAN_TIMEOUT)
        self._last_scan = scan
        self._step_count += 1

        x, y, yaw = self._node.get_robot_pose()
        gx, gy = self._goal
        dist = math.hypot(gx - x, gy - y)

        # Termination conditions
        min_range = _min_scan_range(scan)
        collision = min_range < COLLISION_DIST
        goal_reached = dist < GOAL_RADIUS
        timeout = self._step_count >= self._max_steps

        terminated = collision or goal_reached
        truncated = timeout

        # Reward
        reward = R_APPROACH * (self._prev_dist - dist) + R_TIME
        if collision:
            reward += R_COLLISION
        if goal_reached:
            reward += R_GOAL

        self._prev_dist = dist

        obs = self._build_obs()
        info: Dict[str, Any] = {
            "dist_to_goal": dist,
            "min_scan": min_range,
            "collision": collision,
            "goal_reached": goal_reached,
        }

        if terminated:
            self._node.stop_robot()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._node.stop_robot()

    # ---- internal ---------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        scan = self._last_scan
        ranges = list(scan.ranges) if scan is not None else [LIDAR_MAX_RANGE] * 360
        x, y, yaw = self._node.get_robot_pose()
        gx, gy = self._goal
        return make_observation(ranges, x, y, yaw, gx, gy)

    def _sample_start_goal(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Sample start and goal from candidate list (must differ)."""
        idxs = self._rng.choice(len(self._goal_candidates), size=2, replace=False)
        start = self._goal_candidates[int(idxs[0])]
        goal = self._goal_candidates[int(idxs[1])]
        return start, goal


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

def _smoketest(n_episodes: int = 3) -> None:
    """Quick sanity check: run n_episodes with random actions."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=n_episodes)
    args = parser.parse_args()

    rclpy.init()
    env = TurtleBot3GazeboEnv(seed=42)

    for ep in range(args.episodes):
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,), f"Bad obs shape: {obs.shape}"
        total_reward = 0.0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep+1}: reward={total_reward:.1f} "
              f"goal={info['goal_reached']} collision={info['collision']}")

    env.close()
    rclpy.shutdown()
    print("Smoketest passed.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Convert yaw angle to quaternion (x, y, z, w)."""
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _min_scan_range(scan: Optional[LaserScan]) -> float:
    """Return minimum finite range from a LaserScan message."""
    if scan is None:
        return LIDAR_MAX_RANGE
    ranges = [r for r in scan.ranges if math.isfinite(r) and r > 0.0]
    return min(ranges) if ranges else LIDAR_MAX_RANGE


if __name__ == "__main__":
    _smoketest()
