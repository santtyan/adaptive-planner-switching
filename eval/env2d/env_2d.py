"""
env_2d.py — Ambiente 2D leve compatível com o Gazebo env.

Unicycle kinematics + raycasting NumPy puro.
Mesma obs (29-dim), mesmo action space, mesmo reward do gazebo_gym_env.py.
Roda a ~1000 Hz em CPU (vs 5 Hz do Gazebo).

Uso:
    from eval.env2d.env_2d import Env2D
    env = Env2D(world="sparse")   # "sparse" | "dense" | "very_dense"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros"))

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Reutiliza exatamente o mesmo make_observation do Gazebo
from obs_utils import (
    make_observation, OBS_DIM,
    LIDAR_MAX_RANGE, LIDAR_N_DOWNSAMPLED,
    LINEAR_VEL_MAX, ANGULAR_VEL_MAX,
    COLLISION_DIST, GOAL_DISTANCE_MAX,
)

# ── Constantes ────────────────────────────────────────────────
# ATENÇÃO (09/07): NÃO eram idênticas ao gazebo_gym_env.py — o Gazebo usa
# R_SURVIVAL=0.0, este arquivo tinha 0.1 hardcoded. Divergência que quebrava a
# regra "validar no 2D antes do Gazebo". R_SURVIVAL agora é configurável via env
# var R_SURVIVAL_OVERRIDE para testar a config EXATA do Gazebo (default 0.1).
MAX_STEPS    = 200
R_SURVIVAL   = float(os.environ.get("R_SURVIVAL_OVERRIDE", "0.1"))
R_COLLISION  = -100.0
R_GOAL       = 100.0
R_APPROACH   = 10.0
GOAL_RADIUS  = 0.25   # m

DT           = 0.2    # s — equivale a 5 Hz do LIDAR
ROBOT_RADIUS = 0.17   # m — TB3 Waffle

# ── Definição de mundos (obstáculos circulares: cx, cy, raio) ─
WORLDS = {
    "sparse": {
        "size": 4.0,        # arena 4×4 m
        "obstacles": [      # ρ ≈ 0.05
            (1.0,  1.0, 0.20),
            (-1.0, 0.5, 0.20),
            (0.5, -1.2, 0.20),
        ],
    },
    "dense": {
        "size": 4.0,
        "obstacles": [      # ρ ≈ 0.30
            ( 1.0,  1.0, 0.20), (-1.0,  0.5, 0.20), ( 0.5, -1.2, 0.20),
            (-0.5,  1.5, 0.20), ( 1.5, -0.5, 0.20), (-1.5, -0.8, 0.20),
            ( 0.0,  0.5, 0.15), ( 0.8,  0.0, 0.15), (-0.8, -1.5, 0.15),
            ( 1.2, -1.5, 0.15),
        ],
    },
    "very_dense": {
        "size": 4.0,
        "obstacles": [      # ρ ≈ 0.50
            ( 1.0,  1.0, 0.20), (-1.0,  0.5, 0.20), ( 0.5, -1.2, 0.20),
            (-0.5,  1.5, 0.20), ( 1.5, -0.5, 0.20), (-1.5, -0.8, 0.20),
            ( 0.0,  0.5, 0.15), ( 0.8,  0.0, 0.15), (-0.8, -1.5, 0.15),
            ( 1.2, -1.5, 0.15), ( 0.3,  1.2, 0.15), (-1.2,  1.2, 0.15),
            ( 1.8,  0.8, 0.15), (-0.3, -0.5, 0.15), ( 0.0, -1.8, 0.15),
        ],
    },
}


def _scan(ox: float, oy: float, yaw: float,
          obstacles: list, arena: float,
          n_rays: int = 360,
          max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    """Scan LIDAR vetorizado — todos os raios em paralelo com NumPy.

    Evita loop Python por raio; opera sobre arrays (n_rays,) inteiros.
    ~100× mais rápido que a versão com _raycast() por raio.
    """
    angles = yaw + np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dx = np.cos(angles)   # (n_rays,)
    dy = np.sin(angles)

    best = np.full(n_rays, max_range, dtype=np.float32)
    half = arena / 2.0

    # ── Paredes ────────────────────────────────────────────────
    # Parede x=+half
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (half - ox) / dx
        hit = (t > 1e-6) & (np.abs(oy + t * dy) <= half)
        best = np.where(hit & (t < best), t, best)
        # Parede x=-half
        t = (-half - ox) / dx
        hit = (t > 1e-6) & (np.abs(oy + t * dy) <= half)
        best = np.where(hit & (t < best), t, best)
        # Parede y=+half
        t = (half - oy) / dy
        hit = (t > 1e-6) & (np.abs(ox + t * dx) <= half)
        best = np.where(hit & (t < best), t, best)
        # Parede y=-half
        t = (-half - oy) / dy
        hit = (t > 1e-6) & (np.abs(ox + t * dx) <= half)
        best = np.where(hit & (t < best), t, best)

    # ── Obstáculos circulares ──────────────────────────────────
    for (cx, cy, cr) in obstacles:
        fx = cx - ox; fy = cy - oy
        tca = fx * dx + fy * dy          # (n_rays,)
        d2  = fx*fx + fy*fy - tca*tca
        r2  = (cr + ROBOT_RADIUS) ** 2
        mask = (tca > 0) & (d2 < r2)
        thc  = np.sqrt(np.clip(r2 - d2, 0, None))
        t    = tca - thc
        best = np.where(mask & (t > 1e-6) & (t < best), t, best)

    return best.astype(np.float32)


class Env2D(gym.Env):
    """Ambiente 2D compatível com o Gazebo env (obs 29-dim, ação 2-dim)."""

    metadata = {"render_modes": []}

    def __init__(self, world: str = "sparse", seed: int = 42,
                 curriculum_max_dist: float = 3.0):
        super().__init__()
        assert world in WORLDS, f"world deve ser um de {list(WORLDS)}"
        cfg = WORLDS[world]
        self.obstacles  = cfg["obstacles"]
        self.arena      = cfg["size"]
        self.curr_dist  = curriculum_max_dist

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._rng    = np.random.default_rng(seed)
        self._step   = 0
        self._x      = 0.0
        self._y      = 0.0
        self._yaw    = 0.0
        self._gx     = 1.0
        self._gy     = 0.0
        self._v_norm = 0.0
        self._w_norm = 0.0
        self._prev_dist = 1.0

    # ── Spawn ─────────────────────────────────────────────────
    def _sample_free(self) -> tuple:
        """Amostra posição livre de obstáculos e paredes."""
        half = self.arena / 2.0 - 0.3
        for _ in range(200):
            x = self._rng.uniform(-half, half)
            y = self._rng.uniform(-half, half)
            if all(np.hypot(x - cx, y - cy) > cr + ROBOT_RADIUS + 0.1
                   for cx, cy, cr in self.obstacles):
                return float(x), float(y)
        return 0.0, 0.0   # fallback

    def _sample_goal(self, rx: float, ry: float) -> tuple:
        """Goal livre e dentro do raio de currículo."""
        for _ in range(200):
            gx, gy = self._sample_free()
            dist = np.hypot(gx - rx, gy - ry)
            if 0.5 < dist <= self.curr_dist:
                return gx, gy
        # fallback: goal na direção aleatória
        ang = self._rng.uniform(0, 2 * np.pi)
        d   = self._rng.uniform(0.5, min(self.curr_dist, 1.5))
        return rx + d * np.cos(ang), ry + d * np.sin(ang)

    # ── Gymnasium API ─────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x, self._y = self._sample_free()
        self._yaw = self._rng.uniform(-np.pi, np.pi)
        self._gx, self._gy = self._sample_goal(self._x, self._y)
        self._prev_dist = np.hypot(self._gx - self._x, self._gy - self._y)
        self._v_norm = 0.0
        self._w_norm = 0.0
        self._step = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self._step += 1
        v_norm = float(np.clip(action[0], -1.0, 1.0))
        w_norm = float(np.clip(action[1], -1.0, 1.0))
        v  = v_norm  * LINEAR_VEL_MAX
        w  = w_norm  * ANGULAR_VEL_MAX

        # Unicycle kinematics
        self._yaw += w * DT
        self._yaw  = (self._yaw + np.pi) % (2 * np.pi) - np.pi
        nx = self._x + v * np.cos(self._yaw) * DT
        ny = self._y + v * np.sin(self._yaw) * DT

        # Colisão com parede
        half = self.arena / 2.0 - ROBOT_RADIUS
        wall_hit = not (-half <= nx <= half and -half <= ny <= half)

        # Colisão com obstáculo
        obs_hit = any(
            np.hypot(nx - cx, ny - cy) < cr + ROBOT_RADIUS
            for cx, cy, cr in self.obstacles
        )

        collision = wall_hit or obs_hit
        if not collision:
            self._x, self._y = nx, ny

        self._v_norm = v_norm
        self._w_norm = w_norm

        # Distância ao goal
        dist = float(np.hypot(self._gx - self._x, self._gy - self._y))
        goal_reached = dist < GOAL_RADIUS

        # Reward (idêntico ao gazebo_gym_env.py)
        rprox = float(np.clip(1.0 - dist / max(self._prev_dist, 1e-3), 0.0, 1.0))
        if collision:
            reward = R_COLLISION + rprox
        elif goal_reached:
            reward = R_GOAL
        else:
            r_progress = R_APPROACH * max(0.0, self._prev_dist - dist)
            reward = R_SURVIVAL + r_progress

        self._prev_dist = dist
        terminated = collision or goal_reached
        truncated  = (self._step >= MAX_STEPS) and not terminated

        info = {
            "goal_reached": goal_reached,
            "collision":    collision,
            "distance":     dist,
        }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        ranges = _scan(self._x, self._y, self._yaw,
                       self.obstacles, self.arena)
        return make_observation(
            ranges, self._x, self._y, self._yaw,
            self._gx, self._gy,
            self._v_norm, self._w_norm,
        )

    def close(self):
        pass
