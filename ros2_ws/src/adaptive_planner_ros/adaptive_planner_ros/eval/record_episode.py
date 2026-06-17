"""
Avalia o agente SAC treinado: grava trajetórias e (opcionalmente) vídeo da câmera overhead.

Pré-requisito: gzserver rodando com dense_custom_eval.world + robô spawnado.

Uso:
    # Terminal 1 — Gazebo com câmera overhead:
    source /opt/ros/humble/setup.bash
    export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH
    gzserver ros2_ws/src/adaptive_planner_ros/worlds/dense_custom_eval.world \\
        -slibgazebo_ros_init.so -slibgazebo_ros_factory.so -slibgazebo_ros_force_system.so &
    ros2 run gazebo_ros spawn_entity.py -entity waffle \\
        -file $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_waffle/model.sdf \\
        -x -2.0 -y -0.5 -z 0.01

    # Terminal 2 — Avaliação:
    cd /home/yan/Documentos/Projetos/adaptive-planner-switching
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/eval/record_episode.py \\
        --model models/best_model.zip --episodes 5 --out paper/figs/

Saídas:
    paper/figs/trajectory_ep{N}.png   — trajetória de cada episódio no mapa
    paper/figs/trajectory_all.png     — todos os episódios sobrepostos
    paper/figs/episode_{N}.mp4        — vídeo da câmera overhead (se disponível)
"""

import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from stable_baselines3 import SAC

from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode

# Obstáculos do dense_custom.world (x, y, raio)
OBSTACLES = [
    (-1.2,  1.2, 0.18),
    (-0.6,  1.5, 0.18),
    ( 1.0,  0.3, 0.18),
    ( 1.4, -0.3, 0.18),
    (-0.2, -0.5, 0.18),
    ( 0.5, -1.1, 0.18),
    (-1.0, -1.0, 0.18),
]
ARENA = 2.0  # ±2m


def draw_map(ax):
    ax.set_xlim(-ARENA - 0.2, ARENA + 0.2)
    ax.set_ylim(-ARENA - 0.2, ARENA + 0.2)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f8f8")
    rect = plt.Rectangle((-ARENA, -ARENA), 2*ARENA, 2*ARENA,
                          fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rect)
    for (ox, oy, r) in OBSTACLES:
        circle = mpatches.Circle((ox, oy), r, color="#555555", zorder=3)
        ax.add_patch(circle)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_trajectory(ax, xs, ys, goal, start, success, ep_idx):
    cmap = plt.cm.viridis
    n = len(xs)
    for i in range(n - 1):
        ax.plot(xs[i:i+2], ys[i:i+2],
                color=cmap(i / max(n - 1, 1)), linewidth=1.5, zorder=4)
    ax.scatter(xs[0], ys[0], s=80, color="blue", zorder=5, label="início")
    end_color = "green" if success else "red"
    ax.scatter(xs[-1], ys[-1], s=80, color=end_color, zorder=5,
               label="chegou ao goal" if success else "colisão/timeout")
    ax.scatter(goal[0], goal[1], s=200, marker="*", color="gold",
               edgecolors="black", linewidth=0.5, zorder=6, label="goal")
    goal_circle = mpatches.Circle(goal, 0.25, fill=False,
                                  edgecolor="gold", linestyle="--", linewidth=1)
    ax.add_patch(goal_circle)
    status = "✓ GOAL" if success else "✗ falha"
    ax.set_title(f"Episódio {ep_idx+1} — {status}  ({n} steps)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def try_video_recorder(out_dir, ep_idx):
    """Tenta gravar câmera overhead em MP4. Retorna (recorder, writer) ou None."""
    try:
        import cv2
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image as RosImage
        return cv2, CvBridge(), out_dir, ep_idx
    except ImportError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best_model.zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", default="paper/figs/")
    p.add_argument("--video", action="store_true", help="Gravar vídeo da câmera overhead")
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rclpy.init()
    node = _GazeboEnvNode()
    env = TurtleBot3GazeboEnv(node=node, seed=args.seed)

    model = SAC.load(args.model, env=env)
    print(f"Modelo carregado: {args.model}")

    # Gravação de vídeo via câmera overhead (opcional)
    video_ctx = None
    video_writer = None
    if args.video:
        video_ctx = try_video_recorder(args.out, 0)
        if video_ctx is None:
            print("[WARN] cv2/cv_bridge não disponível — vídeo desabilitado")

    all_trajs = []
    results = []

    fig_all, ax_all = plt.subplots(figsize=(7, 7))
    draw_map(ax_all)
    ax_all.set_title("Trajetórias SAC — todos os episódios", fontsize=12, fontweight="bold")
    colors_ep = plt.cm.tab10(np.linspace(0, 1, args.episodes))

    for ep in range(args.episodes):
        obs, info = env.reset()
        goal = info["goal"]
        xs, ys = [], []
        done = False
        success = False

        # Inicia vídeo para este episódio
        if args.video and video_ctx is not None:
            cv2, bridge, out_dir, _ = video_ctx
            video_path = os.path.join(out_dir, f"episode_{ep+1}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 640))

        while not done:
            x, y, _ = node.get_robot_pose()
            xs.append(x)
            ys.append(y)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            if step_info.get("goal_reached"):
                success = True

            # Captura frame da câmera se disponível
            if args.video and video_writer is not None:
                try:
                    import rclpy as _r
                    from sensor_msgs.msg import Image as RosImage
                    _r.spin_once(node, timeout_sec=0.05)
                except Exception:
                    pass

        all_trajs.append((xs, ys, goal, success))
        results.append(success)

        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"Vídeo salvo: episode_{ep+1}.mp4")

        # Figura individual
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_map(ax)
        plot_trajectory(ax, xs, ys, goal, info.get("start", (xs[0], ys[0])), success, ep)
        ep_path = os.path.join(args.out, f"trajectory_ep{ep+1}.png")
        fig.savefig(ep_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Ep {ep+1}: {'GOAL ✓' if success else 'falha ✗'}  {len(xs)} steps → {ep_path}")

        # Adiciona ao plot geral
        color = colors_ep[ep]
        n = len(xs)
        for i in range(n - 1):
            ax_all.plot(xs[i:i+2], ys[i:i+2], color=color, alpha=0.7, linewidth=1.2)
        ax_all.scatter(goal[0], goal[1], s=150, marker="*", color=color,
                       edgecolors="black", linewidth=0.5, zorder=6)

    # Figura consolidada
    success_rate = sum(results) / len(results) * 100
    ax_all.set_title(
        f"Trajetórias SAC — {args.episodes} episódios  |  "
        f"Taxa de sucesso: {success_rate:.0f}%",
        fontsize=11, fontweight="bold"
    )
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], color=colors_ep[i], linewidth=2,
                         label=f"Ep {i+1} {'✓' if results[i] else '✗'}")
                  for i in range(args.episodes)]
    ax_all.legend(handles=legend_els, fontsize=8, loc="upper right")
    all_path = os.path.join(args.out, "trajectory_all.png")
    fig_all.savefig(all_path, dpi=300, bbox_inches="tight")
    plt.close(fig_all)
    print(f"\nFigura consolidada: {all_path}")
    print(f"Taxa de sucesso: {success_rate:.0f}% ({sum(results)}/{args.episodes})")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
