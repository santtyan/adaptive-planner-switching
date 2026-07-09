"""
record_gazebo_episode.py — Grava a trajetória REAL do robô no Gazebo
(sparse.world) usando o mesmo expert de campo potencial do BC, e gera um
GIF animado a partir dos dados de pose reais.

Motivo: o gzclient (janela gráfica do Gazebo) não sincroniza a malha do
robô nesta configuração (bug de renderização não resolvido, ver
[[project-treino-sparse-08jul]]). O robô e a física funcionam
perfeitamente (confirmado via /odom, /joint_states, /scan) — este script
prova isso visualmente sem depender do gzclient.

Executar DENTRO do container:
    docker compose exec train-all bash -c "
      source /opt/ros/humble/setup.bash &&
      source /workspace/ros2_ws/install/setup.bash &&
      python3 -u /workspace/ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/train/record_gazebo_episode.py \
        --episodes 3 --out /workspace/results_ros2/gazebo_episode.json"
"""
import os
import sys
import json
import argparse

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode

# Obstáculos reais do sparse.world — mesmo padrão usado em train_gazebo_bc.py
OBSTACLES = [
    (-1.2, 1.2, 0.20),
    (1.0, 0.0, 0.20),
    (-0.3, -1.3, 0.20),
]
ARENA_HALF = 2.0


def expert_action(x: float, y: float, yaw: float, gx: float, gy: float) -> np.ndarray:
    dx, dy = gx - x, gy - y
    dist_goal = np.hypot(dx, dy)
    fx, fy = dx / max(dist_goal, 1e-6), dy / max(dist_goal, 1e-6)
    for cx, cy, cr in OBSTACLES:
        ox, oy = x - cx, y - cy
        dist_obs = np.hypot(ox, oy) - cr
        influence = 0.6
        if 0 < dist_obs < influence:
            strength = (1.0 / max(dist_obs, 0.05) - 1.0 / influence)
            fx += strength * ox / max(np.hypot(ox, oy), 1e-6)
            fy += strength * oy / max(np.hypot(ox, oy), 1e-6)
    desired_yaw = np.arctan2(fy, fx)
    yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    w_norm = float(np.clip(yaw_err / (np.pi / 2), -1.0, 1.0))
    v_norm = float(np.clip(1.0 - abs(yaw_err) / (np.pi / 2), 0.15, 1.0))
    return np.array([v_norm, w_norm], dtype=np.float32)


def record_episode(env: TurtleBot3GazeboEnv, max_steps: int = 300) -> dict:
    obs, info = env.reset()
    gx, gy = env._goal
    x, y, yaw = env._node.get_robot_pose()
    frames = [{"x": x, "y": y, "yaw": yaw, "goal": False, "collision": False}]
    done = False
    steps = 0
    goal_reached = collision = False
    while not done and steps < max_steps:
        x, y, yaw = env._node.get_robot_pose()
        action = expert_action(x, y, yaw, gx, gy)
        obs, r, term, trunc, info = env.step(action)
        x, y, yaw = env._node.get_robot_pose()
        goal_reached = info.get("goal_reached", False)
        collision = info.get("collision", False)
        frames.append({"x": x, "y": y, "yaw": yaw,
                        "goal": goal_reached, "collision": collision})
        done = term or trunc
        steps += 1
    outcome = "goal" if goal_reached else "collision" if collision else "timeout"
    return {"frames": frames, "goal": [gx, gy], "outcome": outcome}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/workspace/results_ros2/gazebo_episode.json")
    args = p.parse_args()

    rclpy.init()
    node = _GazeboEnvNode()
    env = TurtleBot3GazeboEnv(node=node, seed=args.seed)

    episodes = []
    for ep in range(args.episodes):
        print(f"Gravando episódio {ep+1}/{args.episodes}...", flush=True)
        data = record_episode(env)
        print(f"  -> {len(data['frames'])} frames, outcome={data['outcome']}", flush=True)
        episodes.append(data)

    out = {
        "obstacles": OBSTACLES,
        "arena_half": ARENA_HALF,
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Salvo em {args.out}", flush=True)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
