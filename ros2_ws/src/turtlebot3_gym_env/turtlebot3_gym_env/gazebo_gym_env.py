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
from collections import deque
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
from nav_msgs.msg import Odometry
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
#
# Princípio (padrão-ouro p/ navegação DRL — reward shaping TurtleBot3, Wiley 2024):
# a recompensa NÃO-terminal por passo é ≤ 0, exceto o termo de progresso, que é
# potencial (telescópico) e portanto não pode ser "farmado" por oscilação.
#   - R_APPROACH: progresso de distância (único termo positivo; soma telescópica).
#   - R_HEADING : penalidade de alinhamento — 0 apontando p/ o goal, negativa fora.
#                 Quebra o ótimo local de "vagar em segurança sem ir ao objetivo".
#   - R_OMEGA   : penalidade por girar em falso.
#   - R_TIME    : penalidade leve por passo.
R_APPROACH = 10.0           # reward per metre closer to goal (potential-based)
R_HEADING = 0.5             # heading penalty scale: R_HEADING*(cos(err)-1) ≤ 0
R_OMEGA = 0.1               # angular-velocity penalty scale (per |omega_norm|)
R_TIME = -0.02              # per step alive penalty
R_COLLISION = -20.0         # terminal collision penalty
R_GOAL = 50.0               # terminal goal reward
# Obstacle reward direcional (padrão-ouro ROBOTIS 2026):
# penalidade contínua proporcional à proximidade de obstáculos À FRENTE do robô.
# Dá gradiente antes da colisão (binário só penaliza quando já colidiu).
# range: [-(1+4), 0] = [-5, 0] por step próximo a obstáculo frontal.
R_OBSTACLE_RANGE = 0.5      # m — só considera obstáculos neste raio
R_OBSTACLE_DECAY = 3.0      # decaimento exponencial: exp(-3*dist)
R_OBSTACLE_WEIGHT_POW = 6   # cos(angle)^6 — frente pesa >>lateral

# Curriculum de distância do goal: começa com goals próximos (sinal terminal
# alcançável) e expande conforme a taxa de sucesso recente sobe. Sem isso, o
# raio de 0.25 m raramente é atingido por exploração e o +R_GOAL nunca aprende.
CURRICULUM_START_DIST = 1.0     # m — distância máx. inicial start→goal
# 20/06/2026: era 10.0, irrealista — spawn candidates da arena 4×4m têm dist máx
# entre pares de 2,92m. O currículo nunca chegava perto. 3.0 ≈ diagonal útil real.
CURRICULUM_MAX_DIST = 3.0       # m — distância máx. final (≈ diagonal da arena)
CURRICULUM_STEP = 0.5           # m — incremento por promoção
CURRICULUM_WINDOW = 10          # episódios na janela de avaliação
CURRICULUM_SUCCESS_RATE = 0.6   # taxa de sucesso p/ promover o currículo

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
        self._cached_pose: tuple = (0.0, 0.0, 0.0)  # (x, y, yaw) from last teleport
        self._odom_pose: Optional[tuple] = None       # (x, y, yaw) from /odom (live)

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        # Subscribers
        self.create_subscription(LaserScan, SCAN_TOPIC,
                                 self._scan_cb, 10)
        self.create_subscription(ModelStates, MODEL_STATES_TOPIC,
                                 self._model_states_cb, 10)
        self.create_subscription(Odometry, "/odom",
                                 self._odom_cb, 10)

        # Service clients
        self._set_entity_cli = self.create_client(SetEntityState,
                                                   "/gazebo/set_entity_state")
        self._clear_costmap_cli = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely",
        )
        self._pause_cli = self.create_client(Empty, "/gazebo/pause_physics")
        self._unpause_cli = self.create_client(Empty, "/gazebo/unpause_physics")

        self.get_logger().info("TurtleBot3GazeboEnv node initialised")

    # ---- callbacks --------------------------------------------------------

    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_seq += 1

    def _model_states_cb(self, msg: ModelStates) -> None:
        self._model_states = msg

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y,
                           p.orientation.z, p.orientation.w)
        self._odom_pose = (p.position.x, p.position.y, yaw)

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
        """Return (x, y, yaw) — priority: model_states > odom > cached teleport.

        /gazebo/model_states requires libgazebo_ros_state.so (not loaded in headless
        training). /odom is always available via turtlebot3_diff_drive plugin and
        tracks the actual robot motion during each episode. Falls back to the cached
        teleport position only before the first odom message arrives.
        """
        if self._model_states is not None:
            try:
                idx = self._model_states.name.index(ROBOT_NAME)
                p = self._model_states.pose[idx]
                return (p.position.x, p.position.y,
                        _quat_to_yaw(p.orientation.x, p.orientation.y,
                                     p.orientation.z, p.orientation.w))
            except ValueError:
                pass
        if self._odom_pose is not None:
            return self._odom_pose
        return self._cached_pose

    def pause_physics(self) -> None:
        """Pausa a física do Gazebo — elimina race condition teleport vs scan."""
        if self._pause_cli.wait_for_service(timeout_sec=0.5):
            future = self._pause_cli.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def unpause_physics(self) -> None:
        """Retoma a física — o próximo scan publicado é garantidamente pós-teleport."""
        if self._unpause_cli.wait_for_service(timeout_sec=0.5):
            future = self._unpause_cli.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

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
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        if future.done() and future.result().success:
            self._cached_pose = (x, y, yaw)
            return True
        return False

    def clear_costmap(self) -> None:
        """Clear the global costmap (removes stale obstacle marks after reset)."""
        if not self._clear_costmap_cli.wait_for_service(timeout_sec=0.1):
            return  # Nav2 may not be running during unit tests
        future = self._clear_costmap_cli.call_async(
            ClearEntireCostmap.Request()
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)


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
        curriculum: bool = True,
    ) -> None:
        super().__init__()

        self._node = node or _GazeboEnvNode()
        self._goal_candidates = goal_candidates or DEFAULT_SPAWN_CANDIDATES
        self._density_range = density_range
        self._max_steps = max_steps

        # Curriculum de distância do goal (ver constantes CURRICULUM_*).
        # Desligado (curriculum=False) → amostra do mapa inteiro, como antes.
        self._curriculum = curriculum
        self._curr_max_dist = (
            CURRICULUM_START_DIST if curriculum else CURRICULUM_MAX_DIST
        )
        self._recent_outcomes: deque = deque(maxlen=CURRICULUM_WINDOW)

        # low=-1.0: a obs inclui sin_theta e a última ação (v_norm, omega_norm),
        # ambos ∈ [-1, 1]. Lidar e r_norm ∈ [0, 1] cabem dentro do intervalo.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._goal: Tuple[float, float] = (0.0, 0.0)
        self._prev_dist: float = 0.0
        self._last_scan: Optional[LaserScan] = None
        # Última ação normalizada (v_norm, omega_norm) ∈ [-1,1], alimentada na obs.
        self._last_action: Tuple[float, float] = (0.0, 0.0)

    # ---- Gymnasium API ----------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Promove o currículo se a taxa de sucesso recente cruzou o limiar.
        self._maybe_advance_curriculum()

        # Sample start and goal (must be different positions)
        start, goal = self._sample_start_goal()
        self._goal = goal

        # Teleport com pause/unpause — padrão ROBOTIS turtlebot3_machine_learning.
        # pause garante que o scan seguinte ao unpause é da nova posição (sem stale).
        SAFE_SPAWN_MARGIN = COLLISION_DIST + 0.10   # 0.25 m clearance required
        MAX_SPAWN_TRIES = len(self._goal_candidates)
        self._node.stop_robot()
        self._node.pause_physics()
        for _try in range(MAX_SPAWN_TRIES):
            self._node.teleport_robot(start[0], start[1], yaw=float(
                self._rng.uniform(-math.pi, math.pi)
            ))
            self._node.unpause_physics()
            scan = self._node.wait_for_scan(timeout=1.5)  # scan garantidamente pós-teleport
            if _min_scan_range(scan) >= SAFE_SPAWN_MARGIN:
                break
            # Posição em colisão — resampa e tenta novamente
            self._node.pause_physics()
            start, goal = self._sample_start_goal()
            self._goal = goal
        self._node.clear_costmap()
        self._last_scan = scan
        self._last_action = (0.0, 0.0)   # robô parado no início do episódio

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
        # Guarda a ação normalizada p/ alimentar a próxima obs (resolve POMDP).
        self._last_action = (float(action[0]), float(action[1]))

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

        # Reward (ver bloco de constantes p/ a racional do padrão-ouro):
        #   - progresso de distância (potencial, único termo positivo)
        #   - penalidade de heading: 0 apontando p/ goal, negativa fora
        #   - penalidade de giro em falso
        #   - penalidade leve por passo
        heading_err = _wrap_angle(math.atan2(gy - y, gx - x) - yaw)
        r_progress = R_APPROACH * (self._prev_dist - dist)
        r_heading = R_HEADING * (math.cos(heading_err) - 1.0)   # ≤ 0
        r_omega = -R_OMEGA * abs(float(action[1]))              # ≤ 0
        r_obstacle = _obstacle_reward(scan)                      # ≤ 0, contínuo
        reward = r_progress + r_heading + r_omega + r_obstacle + R_TIME
        if collision:
            reward += R_COLLISION
        if goal_reached:
            reward += R_GOAL

        self._prev_dist = dist

        obs = self._build_obs()
        # success_rate da janela do curriculum — usado pelo StopOnSuccessCallback
        # p/ encerrar o treino quando o agente domina o goal mais distante.
        sr = (
            sum(self._recent_outcomes) / len(self._recent_outcomes)
            if self._recent_outcomes else 0.0
        )
        info: Dict[str, Any] = {
            "dist_to_goal": dist,
            "min_scan": min_range,
            "collision": collision,
            "goal_reached": goal_reached,
            "heading_err": heading_err,
            "success_rate": sr,
            "curr_max_dist": self._curr_max_dist,
            "at_max_curriculum": self._curr_max_dist >= CURRICULUM_MAX_DIST,
        }

        if terminated or truncated:
            self._node.stop_robot()
            self._record_episode_outcome(goal_reached)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._node.stop_robot()

    # ---- internal ---------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        scan = self._last_scan
        ranges = list(scan.ranges) if scan is not None else [LIDAR_MAX_RANGE] * 360
        x, y, yaw = self._node.get_robot_pose()
        gx, gy = self._goal
        v_norm, omega_norm = self._last_action
        return make_observation(ranges, x, y, yaw, gx, gy, v_norm, omega_norm)

    def _sample_start_goal(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Sample start and goal from candidate list (must differ).

        Respeita o currículo: o par escolhido tem dist(start, goal) ≤
        self._curr_max_dist. Se nenhum par couber no limite (currículo muito
        apertado p/ os candidatos disponíveis), cai no par válido mais próximo.
        """
        n = len(self._goal_candidates)
        cand = self._goal_candidates
        # Todos os pares (i, j) i≠j com sua distância euclidiana.
        pairs = [
            (i, j, math.hypot(cand[i][0] - cand[j][0], cand[i][1] - cand[j][1]))
            for i in range(n) for j in range(n) if i != j
        ]
        within = [p for p in pairs if p[2] <= self._curr_max_dist]
        pool = within if within else [min(pairs, key=lambda p: p[2])]
        choice = pool[int(self._rng.integers(len(pool)))]
        start = cand[int(choice[0])]
        goal = cand[int(choice[1])]
        return start, goal

    # ---- curriculum -------------------------------------------------------

    def _record_episode_outcome(self, goal_reached: bool) -> None:
        """Registra o resultado do episódio p/ a janela do currículo."""
        self._recent_outcomes.append(1 if goal_reached else 0)

    def _maybe_advance_curriculum(self) -> None:
        """Expande a distância máx. de goal se a taxa de sucesso recente é alta."""
        if not self._curriculum:
            return
        if self._curr_max_dist >= CURRICULUM_MAX_DIST:
            return
        if len(self._recent_outcomes) < CURRICULUM_WINDOW:
            return
        success_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        if success_rate >= CURRICULUM_SUCCESS_RATE:
            self._curr_max_dist = min(
                self._curr_max_dist + CURRICULUM_STEP, CURRICULUM_MAX_DIST
            )
            self._recent_outcomes.clear()
            self._node.get_logger().info(
                f"[Curriculum] taxa={success_rate:.0%} → max_dist "
                f"promovida p/ {self._curr_max_dist:.1f} m"
            )


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

    spawn_collisions = 0
    for ep in range(args.episodes):
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,), f"Bad obs shape: {obs.shape}"
        # Gate: o robô NÃO pode nascer em colisão (tunneling / spawn ruim).
        # min_scan no 1º passo válido deve ser ≥ COLLISION_DIST.
        first = env._build_obs()  # noqa: SLF001 — checagem de sanidade do spawn
        min0 = _min_scan_range(env._last_scan)
        if min0 < COLLISION_DIST:
            spawn_collisions += 1
            print(f"[SMOKETEST][WARN] ep{ep+1}: spawn em colisão (min_scan={min0:.2f})")
        total_reward = 0.0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep+1}: reward={total_reward:.1f} "
              f"goal={info['goal_reached']} collision={info['collision']} "
              f"min_scan0={min0:.2f}")
    assert spawn_collisions == 0, (
        f"{spawn_collisions}/{args.episodes} spawns em colisão — "
        f"revisar DEFAULT_SPAWN_CANDIDATES ou max_step_size do .world"
    )

    env.close()
    rclpy.shutdown()
    print("Smoketest passed.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Normaliza um ângulo para o intervalo [-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


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


def _obstacle_reward(scan: Optional[LaserScan]) -> float:
    """Penalidade direcional contínua para obstáculos próximos À FRENTE.

    Padrão-ouro ROBOTIS turtlebot3_machine_learning (2026):
    - só penaliza obstáculos dentro de R_OBSTACLE_RANGE metros
    - peso direcional cos(angle)^6 → frente pesa ~10x mais que lateral
    - decaimento exp(-3*dist) → 0.1m penaliza ~10x mais que 0.3m
    - resultado: -(1 + 4*weighted_decay) ∈ [-5, 0]

    Complementa R_COLLISION (binário) dando gradiente antes da colisão.
    """
    if scan is None:
        return 0.0
    n = len(scan.ranges)
    if n == 0:
        return 0.0
    angle_increment = scan.angle_increment
    angle_min = scan.angle_min
    ranges = np.array(scan.ranges, dtype=np.float32)
    angles = angle_min + np.arange(n) * angle_increment  # ângulos em rad (frame robô)
    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= R_OBSTACLE_RANGE)
    if not np.any(valid):
        return 0.0
    r = ranges[valid]
    a = angles[valid]
    # Peso direcional: frente (a≈0) tem peso máximo
    raw_w = np.cos(a) ** R_OBSTACLE_WEIGHT_POW + 0.1
    raw_w = np.clip(raw_w, 0.0, None)
    w = raw_w / (np.sum(raw_w) + 1e-8)
    safe_dist = np.clip(r - 0.25, 1e-2, 3.5)
    decay = np.exp(-R_OBSTACLE_DECAY * safe_dist)
    weighted_decay = float(np.dot(w, decay))
    return -(1.0 + 4.0 * weighted_decay)


if __name__ == "__main__":
    _smoketest()
