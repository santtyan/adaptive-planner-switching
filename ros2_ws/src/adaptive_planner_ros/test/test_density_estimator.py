"""
Tests for density_estimator.compute_local_density — no ROS needed.

Coverage:
  1. All-free window → ρ=0
  2. All-occupied window → ρ=1
  3. 30% occupied → ρ≈0.30
  4. Unknown cells (-1) treated as occupied (default)
  5. Unknown cells (-1) treated as free (unknown_as_occupied=False)
  6. Robot near map edge (window clips without error)
  7. Robot outside map bounds (returns 0.0, no crash)
"""
import math
import pytest
from unittest.mock import MagicMock

# Import without rclpy by patching at module level
import sys
import types

# Provide a minimal stub so density_estimator.py can be imported without ROS
ros_modules = ['rclpy', 'rclpy.node', 'rclpy.qos',
               'nav_msgs', 'nav_msgs.msg',
               'geometry_msgs', 'geometry_msgs.msg',
               'std_msgs', 'std_msgs.msg']
for mod in ros_modules:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# Stubs needed by density_estimator
sys.modules['rclpy'].init = lambda **k: None
sys.modules['rclpy'].spin = lambda n: None
sys.modules['rclpy'].try_shutdown = lambda: None
sys.modules['rclpy.node'].Node = object
QoSProfile = MagicMock()
for attr in ('DurabilityPolicy', 'ReliabilityPolicy', 'HistoryPolicy'):
    setattr(sys.modules['rclpy.qos'], attr, MagicMock())
sys.modules['rclpy.qos'].QoSProfile = QoSProfile
sys.modules['nav_msgs.msg'].OccupancyGrid = object
sys.modules['geometry_msgs.msg'].PoseWithCovarianceStamped = object
sys.modules['geometry_msgs.msg'].PoseStamped = object
sys.modules['std_msgs.msg'].Float32 = object

import importlib, pathlib, sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from adaptive_planner_ros.density_estimator import compute_local_density


def _make_costmap(width, height, resolution, origin_x, origin_y, fill_value):
    """Create a mock OccupancyGrid with uniform fill."""
    msg = MagicMock()
    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = resolution
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    msg.data = [fill_value] * (width * height)
    return msg


def _make_partial_costmap(width, height, resolution, origin_x, origin_y,
                           occupied_fraction, occ_value=100, seed=42):
    """Fill `occupied_fraction` of cells randomly with occ_value, rest with 0."""
    import random
    rng = random.Random(seed)
    data = [occ_value if rng.random() < occupied_fraction else 0
            for _ in range(width * height)]
    msg = MagicMock()
    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = resolution
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    msg.data = data
    return msg


def test_all_free():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, 0)
    rho = compute_local_density(cm, 2.5, 2.5, window_m=2.0)
    assert rho == pytest.approx(0.0)


def test_all_occupied():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, 100)
    rho = compute_local_density(cm, 2.5, 2.5, window_m=2.0)
    assert rho == pytest.approx(1.0)


def test_30_percent_occupied():
    cm = _make_partial_costmap(200, 200, 0.05, 0.0, 0.0, 0.30)
    # Centre of the map
    rho = compute_local_density(cm, 5.0, 5.0, window_m=2.0)
    assert 0.20 < rho < 0.40, f"Expected ~0.30, got {rho:.3f}"


def test_unknown_as_occupied_default():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, -1)
    rho = compute_local_density(cm, 2.5, 2.5, window_m=2.0, unknown_as_occupied=True)
    assert rho == pytest.approx(1.0)


def test_unknown_as_free():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, -1)
    rho = compute_local_density(cm, 2.5, 2.5, window_m=2.0, unknown_as_occupied=False)
    assert rho == pytest.approx(0.0)


def test_robot_near_edge():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, 50)
    # Robot at corner; window clips to map boundary without raising
    rho = compute_local_density(cm, 0.1, 0.1, window_m=2.0, occ_threshold=50)
    assert 0.0 <= rho <= 1.0


def test_robot_outside_map_returns_zero():
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, 100)
    rho = compute_local_density(cm, 50.0, 50.0, window_m=2.0)
    assert rho == pytest.approx(0.0)


def test_occ_threshold_boundary():
    # Cells at exactly threshold=65 should count as occupied
    cm = _make_costmap(100, 100, 0.05, 0.0, 0.0, 65)
    rho = compute_local_density(cm, 2.5, 2.5, window_m=2.0, occ_threshold=65)
    assert rho == pytest.approx(1.0)

    # Cells at 64 should NOT count
    cm64 = _make_costmap(100, 100, 0.05, 0.0, 0.0, 64)
    rho64 = compute_local_density(cm64, 2.5, 2.5, window_m=2.0, occ_threshold=65)
    assert rho64 == pytest.approx(0.0)
