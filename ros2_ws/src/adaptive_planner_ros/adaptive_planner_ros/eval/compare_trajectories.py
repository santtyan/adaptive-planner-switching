"""
Figura comparativa lado-a-lado: trajetórias A* vs SAC vs Adaptive (3 painéis).

Pode rodar de dois modos:
  1. Com dados reais de um benchmark JSON (gerado por benchmark_sac_vs_nav2.py).
  2. Com trajetórias sintéticas/placeholder (--synthetic) — útil antes do treino
     convergir para validar o layout da figura.

Uso:
    # Sintético (não precisa do Gazebo nem do modelo treinado):
    python3 compare_trajectories.py --synthetic --out paper/figs/

    # Com dados reais (pós-treinamento):
    python3 compare_trajectories.py --data results_ros2/benchmark_results.json --out paper/figs/

Saídas:
    paper/figs/trajectory_comparison.png  — figura 3-painéis 300 dpi
    paper/figs/trajectory_comparison.pdf  — versão PDF para LaTeX
"""

import argparse
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Geometria (deve coincidir com record_episode.py) ─────────────────────────
OBSTACLES = [
    (-1.2,  1.2, 0.18),
    (-0.6,  1.5, 0.18),
    ( 1.0,  0.3, 0.18),
    ( 1.4, -0.3, 0.18),
    (-0.2, -0.5, 0.18),
    ( 0.5, -1.1, 0.18),
    (-1.0, -1.0, 0.18),
]
ARENA = 2.0

COLORS = {
    "astar":    "#1f77b4",  # azul
    "sac":      "#d62728",  # vermelho
    "adaptive": "#2ca02c",  # verde
}

LABELS = {
    "astar":    "A* (Nav2)",
    "sac":      "SAC (RL)",
    "adaptive": "Adaptativo (ρ*=0.30)",
}


# ── Sintético ────────────────────────────────────────────────────────────────

def _arc(start, goal, n=60, noise=0.05, detour_factor=0.0):
    """Trajetória suave do start ao goal com ruído e desvio opcional."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, n)
    xs = start[0] + t * (goal[0] - start[0])
    ys = start[1] + t * (goal[1] - start[1])
    # desvio lateral
    perp = np.array([-(goal[1] - start[1]), goal[0] - start[0]])
    norm = np.linalg.norm(perp) + 1e-9
    perp = perp / norm
    xs += perp[0] * detour_factor * np.sin(np.pi * t)
    ys += perp[1] * detour_factor * np.sin(np.pi * t)
    # ruído
    xs += rng.normal(0, noise, n)
    ys += rng.normal(0, noise, n)
    return list(zip(xs.tolist(), ys.tolist()))


def _avoid_obstacles(traj):
    """Desvia grosseiramente de obstáculos (heurística visual para placeholder)."""
    out = []
    for x, y in traj:
        for ox, oy, r in OBSTACLES:
            d = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if d < r + 0.15:
                push = (r + 0.20) / (d + 1e-9)
                x += (x - ox) * (push - 1)
                y += (y - oy) * (push - 1)
        out.append((x, y))
    return out


def _synthetic_episode(start, goal, planner: str):
    if planner == "astar":
        # A*: segue uma rota mais direta (planejada), com menos ruído
        traj = _arc(start, goal, n=80, noise=0.03, detour_factor=0.25)
        traj = _avoid_obstacles(traj)
        outcome = "goal"
    elif planner == "sac":
        # SAC: trajetória mais orgânica, eventualmente chega
        traj = _arc(start, goal, n=120, noise=0.08, detour_factor=0.60)
        traj = _avoid_obstacles(traj)
        outcome = "goal"
    else:  # adaptive
        # Adaptive: começa como A*, troca para SAC perto dos obstáculos
        half = len(_arc(start, goal, n=80, noise=0.03, detour_factor=0.20)) // 2
        t1 = _arc(start, goal, n=80, noise=0.03, detour_factor=0.20)[:half]
        t2 = _arc(t1[-1], goal, n=60, noise=0.06, detour_factor=0.35)
        traj = t1 + t2
        traj = _avoid_obstacles(traj)
        outcome = "goal"
    return {"trajectory": traj, "outcome": outcome, "planner": planner}


def build_synthetic_data():
    start = (-1.5, -1.5)
    goal = (1.5, 1.5)
    return {
        "start": start,
        "goal": goal,
        "episodes": {
            "astar": [_synthetic_episode(start, goal, "astar")],
            "sac": [_synthetic_episode(start, goal, "sac")],
            "adaptive": [_synthetic_episode(start, goal, "adaptive")],
        },
    }


# ── Desenho ──────────────────────────────────────────────────────────────────

def draw_map(ax):
    ax.set_xlim(-ARENA - 0.15, ARENA + 0.15)
    ax.set_ylim(-ARENA - 0.15, ARENA + 0.15)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")
    for ox, oy, r in OBSTACLES:
        c = plt.Circle((ox, oy), r, color="#333333", alpha=0.80, zorder=5)
        ax.add_patch(c)
    ax.grid(color="white", linewidth=0.5, zorder=0)


def draw_panel(ax, episodes, planner: str, start, goal):
    draw_map(ax)
    color = COLORS[planner]

    for ep in episodes:
        traj = ep["trajectory"]
        if len(traj) < 2:
            continue
        xs, ys = zip(*traj)
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8, zorder=6)
        # Marcador de direção (seta a 70% do caminho)
        mid = int(len(traj) * 0.70)
        if mid < len(traj) - 1:
            dx = traj[mid + 1][0] - traj[mid][0]
            dy = traj[mid + 1][1] - traj[mid][1]
            ax.annotate("", xy=(traj[mid + 1][0], traj[mid + 1][1]),
                        xytext=(traj[mid][0], traj[mid][1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        zorder=7)

    # Start / Goal
    ax.plot(*start, marker="^", ms=10, color="#228B22", zorder=8, label="Início")
    ax.plot(*goal, marker="*", ms=13, color="#FFD700", zorder=8, label="Goal",
            markeredgecolor="#555555", markeredgewidth=0.5)

    ax.set_title(LABELS[planner], fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="Usar trajetórias placeholder (não precisa de Gazebo)")
    p.add_argument("--data", default=None,
                   help="JSON de resultados do benchmark (pós-treinamento)")
    p.add_argument("--episode", type=int, default=0,
                   help="Índice do episódio a plotar (com --data)")
    p.add_argument("--out", default="paper/figs")
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.data and not args.synthetic:
        with open(args.data) as f:
            raw = json.load(f)
        # Espera estrutura: {"start": [...], "goal": [...], "episodes": {"astar": [...], ...}}
        data = raw
        start = tuple(data["start"])
        goal = tuple(data["goal"])
        episodes = data["episodes"]
    else:
        data = build_synthetic_data()
        start = data["start"]
        goal = data["goal"]
        episodes = data["episodes"]
        print("[INFO] Usando trajetórias sintéticas (placeholder).")

    planners = ["astar", "sac", "adaptive"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150,
                             gridspec_kw={"wspace": 0.08})

    for ax, planner in zip(axes, planners):
        eps = episodes.get(planner, [])
        if not eps:
            eps = [{"trajectory": [], "outcome": "none", "planner": planner}]
        draw_panel(ax, eps, planner, start, goal)

    # Legenda global
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#228B22",
               markersize=10, label="Início"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
               markersize=13, markeredgecolor="#555555", label="Goal"),
        mpatches.Patch(facecolor="#333333", alpha=0.80, label="Obstáculo"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    suffix = " (placeholder sintético)" if args.synthetic or not args.data else ""
    fig.suptitle(
        f"Comparação de Trajetórias: A* vs SAC vs Adaptativo{suffix}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    path_png = os.path.join(args.out, "trajectory_comparison.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    print(f"Salvo: {path_png}")

    if not args.no_pdf:
        path_pdf = os.path.join(args.out, "trajectory_comparison.pdf")
        fig.savefig(path_pdf, bbox_inches="tight")
        print(f"Salvo: {path_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
