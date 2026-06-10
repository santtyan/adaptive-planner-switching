"""
density_estimator.py — Local obstacle density from Nav2 costmap.

Key design decisions:
- Subscribes to /global_costmap/costmap with TRANSIENT_LOCAL QoS (required by Nav2).
- Uses a square window around the robot pose to compute local density.
- Cells with value -1 (unknown) are treated as occupied by default (conservative).
- Threshold for "occupied": value >= occ_threshold (Nav2 default: 65).
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Float32


COSTMAP_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def compute_local_density(costmap: OccupancyGrid,
                           robot_x: float, robot_y: float,
                           window_m: float = 2.0,
                           occ_threshold: int = 65,
                           unknown_as_occupied: bool = True) -> float:
    """Compute obstacle density in a square window centred on (robot_x, robot_y).

    Args:
        costmap: Nav2 OccupancyGrid message.
        robot_x/y: robot position in the costmap frame (metres).
        window_m: side length of the square window (metres).
        occ_threshold: cells with value >= this are obstacles (Nav2 default: 65).
        unknown_as_occupied: treat -1 cells as occupied.

    Returns:
        density in [0, 1].
    """
    info = costmap.info
    res = info.resolution          # metres per cell
    origin_x = info.origin.position.x
    origin_y = info.origin.position.y
    width = info.width
    height = info.height
    data = costmap.data            # row-major, len = width * height

    half = window_m / 2.0
    # Convert world window to cell indices
    col_min = max(0, int((robot_x - half - origin_x) / res))
    col_max = min(width - 1, int((robot_x + half - origin_x) / res))
    row_min = max(0, int((robot_y - half - origin_y) / res))
    row_max = min(height - 1, int((robot_y + half - origin_y) / res))

    total = 0
    occupied = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            idx = row * width + col
            if idx < 0 or idx >= len(data):
                continue
            val = data[idx]
            total += 1
            if val == -1:
                if unknown_as_occupied:
                    occupied += 1
            elif val >= occ_threshold:
                occupied += 1

    if total == 0:
        return 0.0
    return occupied / total


class DensityEstimatorNode(Node):
    """ROS2 node that publishes local obstacle density at ~2 Hz."""

    def __init__(self):
        super().__init__('density_estimator')

        self.declare_parameter('window_m', 2.0)
        self.declare_parameter('occ_threshold', 65)
        self.declare_parameter('unknown_as_occupied', True)
        self.declare_parameter('rate_hz', 2.0)

        self._costmap: OccupancyGrid | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._density = 0.0

        self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self._costmap_cb,
            COSTMAP_QOS,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_cb,
            10,
        )

        self._pub = self.create_publisher(Float32, '/adaptive_planner/local_density', 10)
        rate = self.get_parameter('rate_hz').value
        self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info('DensityEstimatorNode started')

    def _costmap_cb(self, msg: OccupancyGrid):
        self._costmap = msg

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _timer_cb(self):
        if self._costmap is None:
            return
        window_m = self.get_parameter('window_m').value
        occ_threshold = self.get_parameter('occ_threshold').value
        unknown_as_occ = self.get_parameter('unknown_as_occupied').value

        self._density = compute_local_density(
            self._costmap,
            self._robot_x, self._robot_y,
            window_m=window_m,
            occ_threshold=occ_threshold,
            unknown_as_occupied=unknown_as_occ,
        )
        msg = Float32()
        msg.data = float(self._density)
        self._pub.publish(msg)

    @property
    def density(self) -> float:
        return self._density


def main(args=None):
    rclpy.init(args=args)
    node = DensityEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
