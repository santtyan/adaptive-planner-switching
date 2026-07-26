"""
run_benchmark.py — systematic 540-trial benchmark.

Conditions (5):
  1. nav2_only          — rho_threshold=+inf  (Nav2 always wins)
  2. ppo_only           — rho_threshold=-inf  (RL always wins, PPO model)
  3. sac_only           — rho_threshold=-inf  (RL always wins, SAC model)
  4. adaptive_nav2_ppo  — τ=0.30, PPO model
  5. adaptive_nav2_sac  — τ=0.30, SAC model   ← primary hypothesis (H1/H2)

Maps (3): open_world, dense_custom, mixed_world
Trials: N=30 per (map × condition)  → 5×3×30 = 450 trials
  (use --trials 30, adjust conditions to get exactly 540 if DDPG is added later)

Each trial logs:
  trial_id, seed, map, condition, success, duration_s,
  path_length_m, min_clearance_m, n_switches, collision, timeout,
  mean_density, var_density, planning_lat_ms_mean, peak_memory_mb

Output: results_ros2/master.csv

Usage:
  source ros2_ws/install/setup.bash
  python3 run_benchmark.py --trials 30 --maps open dense mixed \\
      --ppo-model models/ppo_42_500k.zip \\
      --sac-model models/sac_42_500k.zip

  # Scaffold CSV without Gazebo:
  python3 run_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, fields, asdict
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_id: int
    seed: int
    map_name: str
    condition: str
    success: bool
    duration_s: float
    path_length_m: float
    min_clearance_m: float
    n_switches: int
    collision: bool
    timeout: bool
    mean_density: float
    var_density: float
    planning_lat_ms_mean: float
    peak_memory_mb: float       # SIGAA: uso de recursos computacionais


MAPS = {
    "open":  "open_world",
    "dense": "dense_custom",
    "mixed": "mixed_world",
}

CONDITIONS = [
    "nav2_only",
    "ppo_only",
    "sac_only",
    "adaptive_nav2_ppo",
    "adaptive_nav2_sac",
]

# rho_threshold values that force fixed-mode operation
_RHO_FORCE_NAV2 = 1e9    # τ = +∞  → ρ never reaches threshold → always Nav2
_RHO_FORCE_RL   = -1e9   # τ = -∞  → ρ always exceeds threshold → always RL

_CONDITION_RHO = {
    "nav2_only":         _RHO_FORCE_NAV2,
    "ppo_only":          _RHO_FORCE_RL,
    "sac_only":          _RHO_FORCE_RL,
    "adaptive_nav2_ppo": 0.30,
    "adaptive_nav2_sac": 0.30,
}

_CONDITION_ALGO = {
    "nav2_only":         "ppo",   # algo unused when τ=+∞, value is irrelevant
    "ppo_only":          "ppo",
    "sac_only":          "sac",
    "adaptive_nav2_ppo": "ppo",
    "adaptive_nav2_sac": "sac",
}

GOAL_RADIUS   = 0.25   # metres — mirrors gazebo_gym_env.py
TIMEOUT_S     = 120.0  # seconds per trial
CONTROL_HZ    = 5.0    # Hz — step rate for distance tracking
COLLISION_DIST = 0.20  # metres — min lidar for collision (mirrors gym env)

# Goal sampling: 5–15 m from spawn (generalisation beyond training 1–4 m range)
GOAL_DISTANCE_MIN = 5.0
GOAL_DISTANCE_MAX = 15.0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials",      type=int,   default=30)
    p.add_argument("--maps",        nargs="+",  default=["open", "dense", "mixed"])
    p.add_argument("--conditions",  nargs="+",  default=CONDITIONS)
    p.add_argument("--ppo-model",   default="models/ppo_42_500k.zip")
    p.add_argument("--sac-model",   default="models/sac_42_500k.zip")
    p.add_argument("--output",      default="results_ros2/master.csv")
    p.add_argument("--dry-run",     action="store_true",
                   help="Generate CSV with placeholder data (no Gazebo needed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ROS2 trial runner (real — requires running Gazebo + Nav2 + nodes)
# ---------------------------------------------------------------------------

def _make_trial_node(ppo_model: str, sac_model: str, condition: str):
    """Create a minimal ROS2 node for one trial.

    Imported lazily so the script still works for --dry-run without ROS2.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped, Twist
    from std_msgs.msg import String, Float32
    from sensor_msgs.msg import LaserScan
    from gazebo_msgs.msg import ModelStates
    from gazebo_msgs.srv import SpawnEntity, DeleteEntity
    from geometry_msgs.msg import Point, Quaternion, Pose
    from nav2_msgs.srv import ClearEntireCostmap
    import psutil

    # FIX (26/07/2026): /gazebo/set_entity_state nunca existiu nesta config de
    # Gazebo Classic 11 (mesmo bug já diagnosticado em gazebo_gym_env.py,
    # 09/07/2026 — ver DEVELOPMENT_LOG.md Fase 6 item 4 / Fase 11). O fix
    # correto, já validado no pipeline de treino, é reimplementar teleport()
    # como delete_entity + spawn_entity (os únicos serviços que de fato
    # existem). Portado aqui.
    ROBOT_MODEL_SDF_PATH = (
        "/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf"
    )
    # A entidade spawnada de fato se chama "waffle", não "turtlebot3_waffle"
    # (mesmo mismatch corrigido em gazebo_gym_env.py — ver ROBOT_NAME lá).
    ROBOT_NAME = "waffle"

    rho = _CONDITION_RHO[condition]
    algo = _CONDITION_ALGO[condition]
    model_path = sac_model if algo == "sac" else ppo_model

    class _TrialNode(Node):
        def __init__(self):
            super().__init__("benchmark_trial_node")

            # republish rho_threshold to switcher via parameter service would
            # require lifecycle nodes; instead we publish a special override topic
            # that adaptive_switcher_node respects when present.
            # For now: use dynamic reconfigure via ros2 param set (subprocess).

            self._scan: Optional[object] = None
            self._model_states: Optional[object] = None
            self._density_readings: List[float] = []
            self._switches: int = 0
            self._last_mode: str = ""
            self._path_length: float = 0.0
            self._prev_xy: Optional[Tuple[float, float]] = None
            self._min_clearance: float = float("inf")
            self._planning_lats: List[float] = []
            self._switch_ts: float = 0.0

            # subscribers
            scan_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(LaserScan, "/scan", self._scan_cb, scan_qos)
            self.create_subscription(ModelStates, "/gazebo/model_states",
                                     self._model_states_cb, 10)
            self.create_subscription(Float32, "/adaptive_planner/local_density",
                                     self._density_cb, 10)
            self.create_subscription(String, "/adaptive_planner/mode",
                                     self._mode_cb, 10)

            # service clients
            self._spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
            self._delete_cli = self.create_client(DeleteEntity, "/delete_entity")
            try:
                with open(ROBOT_MODEL_SDF_PATH) as f:
                    self._robot_sdf = f.read()
            except OSError:
                self._robot_sdf = None
            self._clear_costmap_cli = self.create_client(
                ClearEntireCostmap, "/global_costmap/clear_entirely"
            )

            # goal publisher
            self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # ---- callbacks -------------------------------------------------------

        def _scan_cb(self, msg) -> None:
            self._scan = msg
            if msg.ranges:
                valid = [r for r in msg.ranges if math.isfinite(r) and r > 0]
                if valid:
                    self._min_clearance = min(self._min_clearance, min(valid))

        def _model_states_cb(self, msg) -> None:
            self._model_states = msg

        def _density_cb(self, msg) -> None:
            self._density_readings.append(msg.data)

        def _mode_cb(self, msg) -> None:
            mode = msg.data
            if mode != self._last_mode and self._last_mode != "":
                self._switches += 1
                lat = (time.monotonic() - self._switch_ts) * 1000
                if lat < 5000:  # ignore first switch latency (startup)
                    self._planning_lats.append(lat)
                self._switch_ts = time.monotonic()
            self._last_mode = mode

        # ---- helpers ---------------------------------------------------------

        def get_robot_pose(self) -> Tuple[float, float, float]:
            if self._model_states is None:
                return 0.0, 0.0, 0.0
            try:
                idx = self._model_states.name.index(ROBOT_NAME)
            except ValueError:
                return 0.0, 0.0, 0.0
            p = self._model_states.pose[idx]
            x = p.position.x
            y = p.position.y
            ox, oy, oz, ow = (p.orientation.x, p.orientation.y,
                               p.orientation.z, p.orientation.w)
            yaw = math.atan2(2*(ow*oz + ox*oy), 1 - 2*(oy*oy + oz*oz))
            return x, y, yaw

        def teleport(self, x: float, y: float, yaw: float) -> bool:
            """Move robot to (x, y, yaw) via delete_entity + spawn_entity.

            /gazebo/set_entity_state does not exist in this setup (see
            ROBOT_MODEL_SDF_PATH comment above) — reimplemented as
            delete+respawn using the two services that actually exist,
            mirroring the fix already validated in gazebo_gym_env.py.
            Returns True on success.
            """
            if self._robot_sdf is None:
                self.get_logger().warning("robot SDF not loaded, cannot teleport")
                return False

            del_req = DeleteEntity.Request()
            del_req.name = ROBOT_NAME
            del_future = self._delete_cli.call_async(del_req)
            rclpy.spin_until_future_complete(self, del_future, timeout_sec=0.5)
            # OK if delete fails (e.g. first call, entity not spawned yet).

            spawn_req = SpawnEntity.Request()
            spawn_req.name = ROBOT_NAME
            spawn_req.xml = self._robot_sdf
            s, c = math.sin(yaw / 2), math.cos(yaw / 2)
            spawn_req.initial_pose = Pose(
                position=Point(x=x, y=y, z=0.01),
                orientation=Quaternion(x=0.0, y=0.0, z=s, w=c),
            )
            spawn_req.reference_frame = "world"

            fut = self._spawn_cli.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                return True
            self.get_logger().warning("spawn_entity failed during teleport")
            return False

        def clear_costmap(self) -> None:
            if not self._clear_costmap_cli.wait_for_service(timeout_sec=2.0):
                return
            fut = self._clear_costmap_cli.call_async(
                ClearEntireCostmap.Request()
            )
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

        def send_goal(self, gx: float, gy: float) -> None:
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = gx
            msg.pose.position.y = gy
            msg.pose.orientation.w = 1.0
            self._goal_pub.publish(msg)

        def update_path_length(self) -> None:
            x, y, _ = self.get_robot_pose()
            if self._prev_xy is not None:
                dx = x - self._prev_xy[0]
                dy = y - self._prev_xy[1]
                self._path_length += math.hypot(dx, dy)
            self._prev_xy = (x, y)

    return _TrialNode, rho, algo, model_path


def _set_switcher_param(node_name: str, param: str, value: str) -> None:
    """Set a parameter on a running ROS2 node via subprocess."""
    subprocess.run(
        ["ros2", "param", "set", node_name, param, value],
        capture_output=True, timeout=5.0
    )


def run_trial(
    trial_id: int,
    seed: int,
    map_name: str,
    condition: str,
    ppo_model: str,
    sac_model: str,
    dry_run: bool = False,
) -> TrialResult:
    """Run one trial and return measured metrics."""

    if dry_run:
        rng = np.random.default_rng(seed + trial_id)
        success = rng.random() > 0.3
        return TrialResult(
            trial_id=trial_id,
            seed=seed,
            map_name=map_name,
            condition=condition,
            success=success,
            duration_s=float(rng.uniform(10, 90)),
            path_length_m=float(rng.uniform(3, 15)),
            min_clearance_m=float(rng.uniform(0.05, 0.5)),
            n_switches=int(rng.integers(0, 8)),
            collision=not success and rng.random() > 0.5,
            timeout=not success and rng.random() > 0.5,
            mean_density=float(rng.uniform(0.1, 0.5)),
            var_density=float(rng.uniform(0.001, 0.05)),
            planning_lat_ms_mean=float(rng.uniform(5, 40)),
            peak_memory_mb=float(rng.uniform(200, 600)),
        )

    # --- Real trial (requires running Gazebo + Nav2 + nodes) ---------------
    import rclpy
    import psutil

    _TrialNode, rho, algo, model_path = _make_trial_node(ppo_model, sac_model, condition)

    rclpy.init()
    node = _TrialNode()

    try:
        # 1. Configure switcher rho_threshold for this condition
        _set_switcher_param(
            "/adaptive_switcher_node", "rho_threshold", str(rho)
        )

        # 2. Deterministic spawn and goal
        rng = np.random.default_rng(seed)
        spawn_x = float(rng.uniform(-2.0, 2.0))
        spawn_y = float(rng.uniform(-2.0, 2.0))
        spawn_yaw = float(rng.uniform(-math.pi, math.pi))

        angle = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(GOAL_DISTANCE_MIN, GOAL_DISTANCE_MAX))
        goal_x = spawn_x + dist * math.cos(angle)
        goal_y = spawn_y + dist * math.sin(angle)

        # 3. Teleport robot; clear stale costmap
        node.teleport(spawn_x, spawn_y, spawn_yaw)
        time.sleep(0.4)  # Gazebo physics settle
        node.clear_costmap()
        time.sleep(0.2)

        # 4. Send goal and run episode
        node.send_goal(goal_x, goal_y)
        node._switch_ts = time.monotonic()

        proc = psutil.Process(os.getpid())
        peak_mem_mb = proc.memory_info().rss / 1024 / 1024

        t_start = time.monotonic()
        collision = False
        success = False

        while True:
            rclpy.spin_once(node, timeout_sec=1.0 / CONTROL_HZ)

            elapsed = time.monotonic() - t_start

            # track path length
            node.update_path_length()

            # memory high-water mark
            mem_now = proc.memory_info().rss / 1024 / 1024
            if mem_now > peak_mem_mb:
                peak_mem_mb = mem_now

            # check collision (ground truth lidar, not /clock)
            if node._scan is not None:
                valid = [r for r in node._scan.ranges
                         if math.isfinite(r) and r > 0]
                if valid and min(valid) < COLLISION_DIST:
                    collision = True
                    break

            # check goal reached
            x, y, _ = node.get_robot_pose()
            if math.hypot(goal_x - x, goal_y - y) < GOAL_RADIUS:
                success = True
                break

            # check timeout
            if elapsed >= TIMEOUT_S:
                break

        duration = time.monotonic() - t_start
        densities = node._density_readings or [0.0]

        return TrialResult(
            trial_id=trial_id,
            seed=seed,
            map_name=map_name,
            condition=condition,
            success=success,
            duration_s=duration,
            path_length_m=node._path_length,
            min_clearance_m=(node._min_clearance
                             if math.isfinite(node._min_clearance) else 0.0),
            n_switches=node._switches,
            collision=collision,
            timeout=(not success and not collision),
            mean_density=float(np.mean(densities)),
            var_density=float(np.var(densities)),
            planning_lat_ms_mean=(float(np.mean(node._planning_lats))
                                  if node._planning_lats else 0.0),
            peak_memory_mb=peak_mem_mb,
        )

    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results: List[TrialResult] = []
    trial_id = 0

    total = len(args.maps) * len(args.conditions) * args.trials
    print(f"Benchmark: {len(args.maps)} maps × {len(args.conditions)} conditions "
          f"× {args.trials} trials = {total} trials")
    if args.dry_run:
        print("[DRY RUN] Generating placeholder data — no Gazebo required.")

    for map_key in args.maps:
        map_name = MAPS.get(map_key, map_key)
        for condition in args.conditions:
            for t in range(args.trials):
                seed = t + 1  # seeds 1..N
                print(f"  [{trial_id+1}/{total}] map={map_name} "
                      f"cond={condition} seed={seed}", end=" ", flush=True)
                t0 = time.time()
                result = run_trial(
                    trial_id=trial_id,
                    seed=seed,
                    map_name=map_name,
                    condition=condition,
                    ppo_model=args.ppo_model,
                    sac_model=args.sac_model,
                    dry_run=args.dry_run,
                )
                results.append(result)
                trial_id += 1
                print(f"✓ success={result.success} t={time.time()-t0:.1f}s")

    # Write CSV
    fieldnames = [f.name for f in fields(TrialResult)]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nSaved {len(results)} trials → {args.output}")

    # Quick summary
    import pandas as pd
    df = pd.read_csv(args.output)
    summary = (df.groupby(["map_name", "condition"])["success"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "success_rate", "count": "n"}))
    print("\nSummary (success rate):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
