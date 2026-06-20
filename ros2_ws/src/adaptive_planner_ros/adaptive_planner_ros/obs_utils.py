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
GOAL_DISTANCE_MAX = 3.5        # metres — normalise goal distance
# 20/06/2026: era 6.0, mas a arena dense_custom (4×4m) tem dist máx entre spawn
# candidates de 2,92m → r_norm saturava em ≤0.5. 3.5 cobre toda a diagonal útil.

# TB3 Waffle velocity limits used during training
LINEAR_VEL_MAX = 0.22          # m/s
ANGULAR_VEL_MAX = 2.84         # rad/s

# Collision / safety thresholds — shared by training (collision termination)
# and inference (safety-stop guard). Single source of truth: never duplicate.
COLLISION_DIST = 0.15          # metres — min lidar distance treated as collision
SAFETY_STOP_DIST = 0.20        # metres — inference guard stops just before collision


def min_valid_range(ranges: Sequence[float],
                    max_range: float = LIDAR_MAX_RANGE) -> float:
    """Smallest finite lidar return, with inf/nan mapped to max_range.

    Used by the inference-time safety guard to detect imminent collision on the
    raw (un-normalised) scan, independent of the learned policy.
    """
    rays = np.array(ranges, dtype=np.float32)
    rays = np.where(np.isfinite(rays), rays, max_range)
    rays = np.clip(rays, 0.0, max_range)
    return float(rays.min()) if rays.size else max_range


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
                     goal_x: float, goal_y: float,
                     v_norm: float = 0.0, omega_norm: float = 0.0) -> np.ndarray:
    """Build the 29-dim observation vector used during training and inference.

    Layout: [24 normalised lidar rays | r_norm | sin_theta | cos_theta
             | v_norm | omega_norm]

    v_norm/omega_norm são a última ação NORMALIZADA (∈ [-1, 1]) — a velocidade
    comandada no passo anterior. Sem isso a obs é POMDP (não dá p/ inferir
    inércia de um único scan). CRÍTICO: treino (gazebo_gym_env) e inferência
    (rl_controller_node) devem alimentar os MESMOS valores, ou a política
    diverge em silêncio.
    """
    scan_obs = downsample_scan(ranges)                               # (24,)
    goal_obs = goal_to_polar(robot_x, robot_y, robot_yaw,           # (3,)
                             goal_x, goal_y)
    vel_obs = np.array([float(np.clip(v_norm, -1.0, 1.0)),         # (2,)
                        float(np.clip(omega_norm, -1.0, 1.0))],
                       dtype=np.float32)
    return np.concatenate([scan_obs, goal_obs, vel_obs])            # (29,)


OBS_DIM = LIDAR_N_DOWNSAMPLED + 3 + 2   # 29 (lidar + goal-polar + last-action)
