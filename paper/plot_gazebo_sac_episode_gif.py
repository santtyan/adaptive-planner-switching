"""
plot_gazebo_sac_episode_gif.py — GIF animado de um episódio REAL do SAC
navegando no Gazebo Classic (sparse.world), a partir do JSON gravado por
ros2_ws/.../eval/record_gazebo_sac_episode.py.

Fecha a lacuna de material visual do Gazebo real (sessão 25/07/2026) --
mesmo estilo dos GIFs do gêmeo 2D (visualize_bc_gif.py,
visualize_urban_dynamic_gif.py), agora com dado de robô físico simulado
real, não só o gêmeo 2D de raycasting.

Uso:
    python3 paper/plot_gazebo_sac_episode_gif.py --json /tmp/gazebo_sac_episode.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

FIGS2D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "2d")
ROBOT_RADIUS = 0.17
GOAL_RADIUS = 0.20


def draw_arena(ax, obstacles, arena_half):
    rect = plt.Rectangle((-arena_half, -arena_half), 2 * arena_half, 2 * arena_half,
                          fill=False, edgecolor="#37474F", lw=2)
    ax.add_patch(rect)
    for cx, cy, cr in obstacles:
        c = Circle((cx, cy), cr, color="#607D8B", alpha=0.85, zorder=3)
        ax.add_patch(c)
    ax.set_xlim(-arena_half - 0.2, arena_half + 0.2)
    ax.set_ylim(-arena_half - 0.2, arena_half + 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def make_gif(data, episode_idx=0, fps=10):
    obstacles = data["obstacles"]
    arena_half = data["arena_half"]
    ep = data["episodes"][episode_idx]
    frames_data = ep["frames"]
    gx, gy = ep["goal"]
    outcome = ep["outcome"]
    total = len(frames_data)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_trail = plt.cm.plasma

    def draw_frame(i):
        ax.clear()
        draw_arena(ax, obstacles, arena_half)
        xs = [f["x"] for f in frames_data[:i+1]]
        ys = [f["y"] for f in frames_data[:i+1]]
        if len(xs) > 1:
            for k in range(len(xs) - 1):
                t = k / max(total - 1, 1)
                ax.plot(xs[k:k+2], ys[k:k+2], "-", color=cmap_trail(t),
                        lw=2.2, alpha=0.85, zorder=4)
        f = frames_data[i]
        x, y, yaw, goal, coll = f["x"], f["y"], f["yaw"], f["goal"], f["collision"]
        color_robot = "#F44336" if coll else "#4CAF50" if goal else "#2196F3"
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, color=color_robot, zorder=5, alpha=0.92))
        ax.annotate("", xy=(x + 0.25*np.cos(yaw), y + 0.25*np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.2))
        ax.plot(gx, gy, "*", color="#FF5722", ms=16, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#FF5722", alpha=0.18))
        last = (i == total - 1)
        status = "✓ GOAL!" if goal else "✗ COLISÃO" if coll else ("✗ TIMEOUT" if last else f"passo {i}/{total-1}")
        ax.set_title(f"SAC no Gazebo Classic real — sparse.world\n{status}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ani = animation.FuncAnimation(fig, draw_frame, frames=total, interval=1000 // fps)
    os.makedirs(FIGS2D, exist_ok=True)
    gif_path = os.path.join(FIGS2D, "fig_gazebo_sac_episode.gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  ✓ 2d/fig_gazebo_sac_episode.gif [{outcome}] ({total} frames)")

    # versão estática (PNG/PDF) para o relatório em PDF
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_arena(ax, obstacles, arena_half)
    xs = [f["x"] for f in frames_data]
    ys = [f["y"] for f in frames_data]
    n = len(xs)
    cmap = plt.cm.plasma
    for i in range(n - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=cmap(i / max(n-1, 1)), lw=2, zorder=4)
    ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=12, zorder=6,
            label="Início", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(gx, gy, "*", color="#F44336", ms=16, zorder=6,
            label="Goal", markeredgecolor="white", markeredgewidth=1)
    ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#F44336", alpha=0.15, zorder=2))
    status = "✓ Goal" if outcome == "goal" else "✗ Colisão" if outcome == "collision" else "Timeout"
    ax.set_title(f"Trajetória SAC no Gazebo Classic real (sparse.world)\n{n} passos | {status}", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS2D, f"fig_gazebo_sac_trajectory.{ext}")
        plt.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_gazebo_sac_trajectory.png/pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--episode", type=int, default=0)
    args = p.parse_args()
    data = json.load(open(args.json))
    make_gif(data, args.episode)
