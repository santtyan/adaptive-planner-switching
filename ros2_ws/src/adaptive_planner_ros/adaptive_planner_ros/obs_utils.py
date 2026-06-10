"""
obs_utils.py — Canonical observation builder shared by:
  - turtlebot3_gym_env (training)
  - rl_controller_node (inference)

CRITICAL: both must import this module. Never duplicate the normalization logic.
If they differ, the model predicts garbage silently.
"""
import numpy as np
from typing import Sequence

# TB3 Waffle lidar specs (LDS-01 / LDS-02)
LIDAR_MAX_RANGE = 3.5          # metres
LIDAR_N_RAYS = 360
LIDAR_N_DOWNSAMPLED = 24       # 15° per bucket
GOAL_DISTANCE_MAX = 6.0        # metres — normalise goal distance

# TB3 Waffle velocity limits used during training
LINEAR_VEL_MAX = 0.22          # m/s
ANGULAR_VEL_MAX = 2.84         # rad/s


def downsample_scan(ranges: Sequence[float], n_out: int = LIDAR_N_DOWNSAMPLED,
                    max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    """Downsample a full 360-ray scan to n_out rays by averaging each bucket.

    Replaces inf/nan with max_range before averaging.
    Returns normalised values in [0, 1].
    """
    rays = np.array(ranges, dtype=np.float32)
    rays = np.where(np.isfinite(rays), rays, max_range)
    rays = np.clip(rays, 0.0, max_range)

    bucket_size = len(rays) // n_out
    downsampled = np.array([
        rays[i * bucket_size:(i + 1) * bucket_size].mean()
        for i in range(n_out)
    ], dtype=np.float32)

    return downsampled / max_range  # → [0, 1]


def goal_to_polar(robot_x: float, robot_y: float, robot_yaw: float,
                  goal_x: float, goal_y: float) -> np.ndarray:
    """Return (r_norm, sin_theta, cos_theta) of goal in robot frame.

    r_norm: distance / GOAL_DISTANCE_MAX, clipped to [0, 1]
    theta:  angle of goal relative to robot heading
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = np.hypot(dx, dy)
    angle_world = np.arctan2(dy, dx)
    theta = angle_world - robot_yaw
    # normalise theta to [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    r_norm = float(np.clip(distance / GOAL_DISTANCE_MAX, 0.0, 1.0))
    return np.array([r_norm, float(np.sin(theta)), float(np.cos(theta))],
                    dtype=np.float32)


def make_observation(ranges: Sequence[float],
                     robot_x: float, robot_y: float, robot_yaw: float,
                     goal_x: float, goal_y: float) -> np.ndarray:
    """Build the 27-dim observation vector used during training and inference.

    Layout: [24 normalised lidar rays | r_norm | sin_theta | cos_theta]
    """
    scan_obs = downsample_scan(ranges)                               # (24,)
    goal_obs = goal_to_polar(robot_x, robot_y, robot_yaw,           # (3,)
                             goal_x, goal_y)
    return np.concatenate([scan_obs, goal_obs])                      # (27,)


OBS_DIM = LIDAR_N_DOWNSAMPLED + 3   # 27
