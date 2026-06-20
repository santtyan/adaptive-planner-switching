"""
Figura: espaço de observação 29-dim do SAC + arquitetura completa do sistema.
Gera:
  paper/figs/fig_obs_29dim.png/.pdf
  paper/figs/fig_system_architecture.png/.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyArrow

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)

BLUE   = "#1565C0"
RED    = "#c62828"
GREEN  = "#2e7d32"
GRAY   = "#607d8b"
ORANGE = "#e65100"
PURPLE = "#6a1b9a"
TEAL   = "#00695c"

def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Obs 29-dim detalhada
# ─────────────────────────────────────────────────────────────────────────────
def fig_obs_29dim():
    fig, (ax_robot, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.5),
                                            gridspec_kw={"width_ratios": [1, 2]})

    # ── Painel esquerdo: TurtleBot3 com anotações de obs ──────────────────────
    ax_robot.set_xlim(-2.8, 2.8)
    ax_robot.set_ylim(-2.8, 2.8)
    ax_robot.set_aspect("equal")
    ax_robot.set_facecolor("#f5f5f5")
    ax_robot.set_title("TurtleBot3 Waffle — componentes da obs", fontsize=10, fontweight="bold")

    # Arena
    ax_robot.add_patch(Rectangle((-2, -2), 4, 4, fill=False, edgecolor="#333", lw=2))

    # LIDAR rays (24 uniformes) — simulados
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    ranges = np.array([1.8,1.6,1.4,1.1,0.6,0.8,1.2,1.7,
                       1.9,1.5,1.3,1.0,0.7,0.9,1.4,1.8,
                       1.6,1.2,0.8,1.1,1.5,1.7,1.9,1.8])
    for i, (ang, r) in enumerate(zip(angles, ranges)):
        color = BLUE if i % 6 != 0 else "#ff5533"
        ax_robot.plot([0, r*np.cos(ang)], [0, r*np.sin(ang)],
                      color=color, lw=1.2, alpha=0.7)
    ax_robot.text(0.5, 2.3, "24 raios\nLIDAR (obs[0:24])", fontsize=8,
                  color=BLUE, ha="center", fontweight="bold")

    # Robô
    ax_robot.add_patch(Circle((0, 0), 0.22, color=GRAY, zorder=5))
    ax_robot.text(0, 0, "R", ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold", zorder=6)

    # Goal
    gx, gy = 1.3, 1.1
    ax_robot.plot(gx, gy, "*", color=GREEN, markersize=18, zorder=7)
    ax_robot.annotate("Goal\nd=obs[24]\nθ=obs[25]\nφ=obs[26]",
                      xy=(gx, gy), xytext=(1.8, -0.5),
                      fontsize=8, color=GREEN, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    # Velocidades
    ax_robot.annotate("v = obs[27]\nω = obs[28]\n(ação anterior)",
                      xy=(0.22, 0), xytext=(-2.5, -1.0),
                      fontsize=8, color=ORANGE, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax_robot.set_xlabel("x (m)", fontsize=9)
    ax_robot.set_ylabel("y (m)", fontsize=9)
    ax_robot.grid(alpha=0.2)

    # ── Painel direito: diagrama de blocos da obs ──────────────────────────────
    ax_bar.set_xlim(0, 29)
    ax_bar.set_ylim(-0.5, 3.5)
    ax_bar.set_facecolor("#f9f9f9")
    ax_bar.set_title("Vetor de observação — 29 dimensões", fontsize=10, fontweight="bold")

    groups = [
        (0,  24, BLUE,   "LIDAR downsampled\n360→24 raios uniformes\nobs[0:24]"),
        (24, 3,  GREEN,  "Goal polar\n(dist, sin θ, cos θ)\nobs[24:27]"),
        (27, 2,  ORANGE, "Ação anterior\n(v norm, ω norm)\nobs[27:29]"),
    ]

    y_bar = 1.5
    for start, length, color, label in groups:
        ax_bar.barh(y_bar, length, left=start, height=0.6,
                    color=color, alpha=0.85, edgecolor="white", lw=1.5)
        cx = start + length / 2
        ax_bar.text(cx, y_bar, str(length), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        ax_bar.text(cx, y_bar - 0.55, label, ha="center", va="top",
                    fontsize=8, color=color, fontweight="bold")

    # Índices
    for i in range(0, 30, 4):
        ax_bar.text(i, y_bar + 0.4, str(i), ha="center", fontsize=7, color="#555")
    ax_bar.axvline(24, color="#999", lw=1, ls="--")
    ax_bar.axvline(27, color="#999", lw=1, ls="--")

    ax_bar.set_xlabel("Índice da dimensão", fontsize=10)
    ax_bar.set_xlim(-0.5, 29.5)
    ax_bar.set_yticks([])

    # Info POMDP
    ax_bar.text(14.5, 2.8,
                "Resolve POMDP: velocidade + última ação eliminam ambiguidade de inércia\n"
                "Normalização: LIDAR /3.5m, dist /CURRICULUM_MAX_DIST, ações já ∈[−1,1]",
                ha="center", fontsize=8.5, color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#e8f4fd", ec=BLUE, alpha=0.9))

    fig.suptitle("Espaço de Observação SAC — 29 dimensões (resolve POMDP)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig_obs_29dim")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Arquitetura completa do sistema (sensor → decisão → atuação)
# ─────────────────────────────────────────────────────────────────────────────
def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, color, fs=9):
        ax.add_patch(Rectangle((x, y), w, h, color=color, alpha=0.88,
                               edgecolor="black", lw=1.5, zorder=3,
                               capstyle="round"))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=4,
                wrap=True)

    def arr(x1, y1, x2, y2, label="", lc="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=lc, lw=1.8), zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha="center", fontsize=7.5, color=lc, zorder=6)

    # ── Sensores ──────────────────────────────────────────────────────────────
    box(0.2, 3.0, 1.6, 1.0, "LIDAR\n360 raios", GRAY)
    box(0.2, 1.6, 1.6, 1.0, "Odometria\n(v, ω)", GRAY)
    box(0.2, 4.4, 1.6, 1.0, "Goal\n(x, y)", GRAY)

    # ── Pré-processamento ──────────────────────────────────────────────────────
    box(2.2, 2.5, 1.8, 2.0, "obs_utils\ndownsample\n360→24\nnorm.", TEAL, fs=8)
    arr(1.8, 3.5, 2.2, 3.5, "360 raios", GRAY)
    arr(1.8, 2.1, 2.2, 3.0, "v, ω", GRAY)
    arr(1.8, 4.9, 2.2, 4.5, "goal polar", GRAY)

    # ── obs 29-dim ────────────────────────────────────────────────────────────
    box(4.2, 2.8, 1.4, 1.4, "obs\n29-dim", BLUE, fs=10)
    arr(4.0, 3.5, 4.2, 3.5, "29-dim", TEAL)

    # ── ρ estimator ───────────────────────────────────────────────────────────
    box(4.2, 5.2, 1.4, 1.0, "ρ estimator\n(costmap)", PURPLE)
    arr(1.8, 4.9, 4.2, 5.7, "", GRAY)

    # ── Switcher π(ρ) ─────────────────────────────────────────────────────────
    ax.add_patch(Circle((7.0, 5.7), 0.55, color=ORANGE, zorder=3, ec="black", lw=1.5))
    ax.text(7.0, 5.7, "π(ρ)", ha="center", va="center", color="white",
            fontweight="bold", fontsize=11, zorder=4)
    arr(5.6, 5.7, 6.45, 5.7, "ρ", PURPLE)

    # ── Planners ──────────────────────────────────────────────────────────────
    box(8.0, 6.0, 2.2, 0.9, "A* / Nav2\n(SmacPlanner2D)", BLUE)
    box(8.0, 4.6, 2.2, 0.9, "SAC Policy\n(Stable-Baselines3)", RED)
    arr(7.55, 5.95, 8.0, 6.45, "ρ < 0.30", BLUE)
    arr(7.55, 5.45, 8.0, 5.05, "ρ ≥ 0.30", RED)

    # obs → SAC
    arr(5.6, 3.5, 8.0, 5.0, "obs[0:29]", BLUE)

    # ── Fusão de saída ─────────────────────────────────────────────────────────
    box(10.5, 5.2, 1.6, 1.0, "Velocidades\n(v, ω)", GREEN)
    arr(10.2, 6.45, 10.5, 5.8, "", GREEN)
    arr(10.2, 5.05, 10.5, 5.4, "", GREEN)

    # ── TurtleBot3 ────────────────────────────────────────────────────────────
    box(12.3, 5.1, 1.5, 1.2, "TurtleBot3\nWaffle\n(cmd_vel)", GRAY)
    arr(12.1, 5.7, 12.3, 5.7, "cmd_vel", GREEN)

    # ── SAC Training loop ─────────────────────────────────────────────────────
    box(6.0, 1.0, 2.2, 1.2, "SAC Training\n(replay buffer\ngradient_steps=4)", RED, fs=8)
    ax.annotate("", xy=(8.0, 4.7), xytext=(8.0, 2.2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5, ls="--"), zorder=5)
    ax.text(8.2, 3.4, "policy\nupdate", fontsize=7.5, color=RED)

    # reward
    ax.annotate("", xy=(7.0, 2.2), xytext=(10.5, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, ls="--",
                               connectionstyle="arc3,rad=0.3"), zorder=5)
    ax.text(9.0, 0.5, "reward (scan, dist, heading)", fontsize=7.5, color=RED, ha="center")
    arr(5.6, 3.5, 6.0, 2.2, "obs,\nreward", RED)

    # ── Labels de fases ───────────────────────────────────────────────────────
    ax.text(1.0, 0.3, "① Sensores", fontsize=8, ha="center", color=GRAY, style="italic")
    ax.text(3.1, 0.3, "② Pré-proc.", fontsize=8, ha="center", color=TEAL, style="italic")
    ax.text(7.0, 0.3, "③ Decisão π(ρ)", fontsize=8, ha="center", color=ORANGE, style="italic")
    ax.text(11.0, 0.3, "④ Atuação", fontsize=8, ha="center", color=GREEN, style="italic")

    ax.set_title("Arquitetura Completa — Adaptive Planner Switching em ROS2/Gazebo\n"
                 "Fase 2: A*/Nav2 (global, ρ<0.30) ↔ SAC/SB3 (local, ρ≥0.30)",
                 fontsize=12, fontweight="bold")
    _save(fig, "fig_system_architecture")


if __name__ == "__main__":
    fig_obs_29dim()
    fig_system_architecture()
    print("Todas as figuras geradas em paper/figs/")
