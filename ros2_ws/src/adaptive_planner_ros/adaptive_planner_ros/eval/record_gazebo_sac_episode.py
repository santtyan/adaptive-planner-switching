"""
record_gazebo_sac_episode.py — Grava trajetórias REAIS do robô no Gazebo
navegando com a política SAC treinada (não o expert de campo potencial de
train/record_gazebo_episode.py), no mesmo formato JSON, para gerar GIF.

Fecha a lacuna de material visual do Gazebo real (sessão 25/07/2026,
docs/PLANO_CORRECAO.md) após confirmar, via diagnose_physics_window.py e
o benchmark_sac_vs_nav2.py, que o robô navega normalmente em Gazebo real
com o modelo SAC treinado -- o bug "robô não ganha velocidade" documentado
anteriormente não se reproduziu no fluxo real (só em posições de spawn
extremas fora do protocolo real, ver achado nesta sessão).

Executar DENTRO do container:
    python3 record_gazebo_sac_episode.py --model /workspace/models/best_model.zip \
        --episodes 3 --out /workspace/results_ros2/gazebo_sac_episode.json
"""
import os
import sys
import json
import argparse

import numpy as np
import rclpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode


def record_episode(env: TurtleBot3GazeboEnv, model, max_steps: int = 200, seed=None) -> dict:
    obs, info = env.reset(seed=seed)
    gx, gy = env._goal
    x, y, yaw = env._node.get_robot_pose()
    frames = [{"x": x, "y": y, "yaw": yaw, "goal": False, "collision": False}]
    done = False
    steps = 0
    goal_reached = collision = False
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
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
    p.add_argument("--model", default="/workspace/models/best_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=200)
    p.add_argument("--out", default="/workspace/results_ros2/gazebo_sac_episode.json")
    p.add_argument("--only-success", action="store_true",
                    help="grava só episódios com goal_reached=True (procura entre --episodes tentativas)")
    args = p.parse_args()

    from stable_baselines3 import SAC
    model = SAC.load(args.model)

    rclpy.init()
    node = _GazeboEnvNode()
    env = TurtleBot3GazeboEnv(node=node, seed=args.seed_start, curriculum=False)

    episodes = []
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        print(f"Gravando episódio {ep+1}/{args.episodes} (seed={seed})...", flush=True)
        data = record_episode(env, model, seed=seed)
        print(f"  -> {len(data['frames'])} frames, outcome={data['outcome']}", flush=True)
        if (not args.only_success) or data["outcome"] == "goal":
            episodes.append(data)
            if args.only_success:
                break

    out = {
        "obstacles": [(-1.2, 1.2, 0.20), (1.0, 0.0, 0.20), (-0.3, -1.3, 0.20)],
        "arena_half": 2.0,
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSalvo em {args.out} ({len(episodes)} episódios)")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
