#!/usr/bin/env python3
"""
Figuras científicas da arquitetura SAC e ambiente de navegação.

Gera figuras prontas para artigo (CONPEEX / relatório final / IEEE).
Uso:
    python3 eval/plot_sac_architecture_figures.py --out paper/figs/
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY, PURPLE, ORANGE = (
    "#2c6fbb", "#c0392b", "#27ae60", "#7f8c8d", "#8e44ad", "#e67e22"
)


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  ok {name}.png/.pdf")


def fig_lidar_downsampling(out):
    """360 raios LIDAR → 24 amostras uniformes (redução de dimensionalidade)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={"projection": "polar"})

    angles_full = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # Simular scan com obstáculos
    ranges_full = 3.5 * np.ones(360)
    ranges_full[30:60] = np.linspace(3.5, 0.8, 30)   # obstáculo à frente-direita
    ranges_full[55:75] = 0.8
    ranges_full[160:200] = np.linspace(3.5, 1.2, 40)  # obstáculo atrás
    ranges_full[195:210] = 1.2

    ax = axes[0]
    ax.plot(angles_full, ranges_full, color=BLUE, lw=0.8, alpha=0.7)
    ax.fill_between(angles_full, 0, ranges_full, alpha=0.15, color=BLUE)
    ax.set_title("LIDAR bruto\n(360 raios)", pad=12)
    ax.set_ylim(0, 4)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1m", "2m", "3m"], fontsize=8)
    ax.grid(True, alpha=0.3)

    # 24 amostras uniformes
    idx24 = np.linspace(0, 359, 24, dtype=int)
    angles24 = angles_full[idx24]
    ranges24 = ranges_full[idx24]

    ax2 = axes[1]
    ax2.vlines(angles24, 0, ranges24, color=RED, lw=2, alpha=0.9)
    ax2.scatter(angles24, ranges24, color=RED, s=40, zorder=5)
    ax2.fill_between(angles_full, 0, ranges_full, alpha=0.08, color=BLUE)
    ax2.set_title("Obs SAC\n(24 amostras uniformes)", pad=12)
    ax2.set_ylim(0, 4)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["1m", "2m", "3m"], fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Downsampling LIDAR: 360 → 24 raios uniformes\n"
                 "Preserva a estrutura do ambiente; reduz dimensão da obs de 360 → 24",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig_lidar_downsampling")


def fig_curriculum_schedule(out):
    """Curriculum de distância: curr_max_dist aumenta com success_rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Painel 1: distância máxima do curriculum ao longo do treino
    steps = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 200]) * 1000
    dist = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0])
    ax = axes[0]
    ax.step(steps / 1000, dist, where="post", color=BLUE, lw=2.5)
    ax.fill_between(steps / 1000, 1.0, dist, step="post", alpha=0.2, color=BLUE)
    ax.axhline(3.0, ls="--", c=GREEN, lw=1.5, label="CURRICULUM_MAX_DIST=3.0m")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("curr_max_dist (m)")
    ax.set_title("Progressão do curriculum\n(sr ≥ 0.60 → promove distância)")
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 3.5)

    # Painel 2: diagrama arena com goal em diferentes distâncias
    ax2 = axes[1]
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect("equal")
    ax2.add_patch(Rectangle((-2, -2), 4, 4, fill=False, edgecolor="black", lw=2))

    robot = Circle((0, 0), 0.17, color=BLUE, zorder=5)
    ax2.add_patch(robot)
    ax2.text(0, 0, "R", ha="center", va="center", color="white",
             fontweight="bold", fontsize=9)

    for d, c, label in [(1.0, RED, "1m (início)"),
                        (2.0, ORANGE, "2m (meio)"),
                        (3.0, GREEN, "3m (max)")]:
        circle = Circle((0, 0), d, fill=False, edgecolor=c, ls="--", lw=1.5,
                        alpha=0.8)
        ax2.add_patch(circle)
        ax2.annotate(label, xy=(d * 0.707, d * 0.707), fontsize=8, color=c,
                     fontweight="bold")

    # Obstáculos (simplificado)
    for ox, oy in [(0.8, 0.5), (-0.5, 1.0), (1.2, -0.8), (-1.0, -0.5), (0.3, -1.2)]:
        ax2.add_patch(Circle((ox, oy), 0.18, color=GRAY, alpha=0.7))

    ax2.set_title("Arena 4×4m — raios do curriculum\n(goal sortado até curr_max_dist)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Curriculum Learning por distância — SAC adaptive planner",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig_curriculum_schedule")


def fig_sac_reward_shaping(out):
    """Reward shaping: potencial Φ = -dist_to_goal (progress reward)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Painel 1: r_progress vs distância ao goal
    ax = axes[0]
    d_prev = np.linspace(0.1, 3.0, 100)
    d_curr_closer = d_prev - 0.1   # robô se aproximou 0.1m
    d_curr_farther = d_prev + 0.1  # robô se afastou 0.1m
    r_approach = (d_prev - np.clip(d_curr_closer, 0, None))  # > 0
    r_retreat = (d_prev - d_curr_farther)                    # < 0
    ax.plot(d_prev, r_approach, color=GREEN, lw=2, label="Aproximando (+)")
    ax.plot(d_prev, r_retreat, color=RED, lw=2, label="Afastando (−)")
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xlabel("Distância ao goal (m)")
    ax.set_ylabel("r_progress por step")
    ax.set_title("Reward de progresso (potencial)\nr = Φ(s') − Φ(s), Φ = −dist")
    ax.legend(fontsize=9)

    # Painel 2: r_heading vs erro de heading
    ax2 = axes[1]
    heading_err = np.linspace(-np.pi, np.pi, 300)
    r_heading = np.cos(heading_err) - 1.0   # ∈ [-2, 0]
    ax2.plot(np.degrees(heading_err), r_heading, color=BLUE, lw=2)
    ax2.axvline(0, ls="--", c=GREEN, alpha=0.7, label="heading perfeito (0)")
    ax2.axvline(90, ls="--", c=ORANGE, alpha=0.7, label="90° de erro")
    ax2.axvline(-90, ls="--", c=ORANGE, alpha=0.7)
    ax2.set_xlabel("Erro de heading (graus)")
    ax2.set_ylabel("r_heading por step")
    ax2.set_title("Reward de heading\nr = cos(err) − 1 ∈ [−2, 0]")
    ax2.legend(fontsize=9)

    fig.suptitle("Reward shaping — progresso + heading (componentes contínuos)",
                 fontsize=11)
    _save(fig, out, "fig_sac_reward_shaping")


def fig_planb_trigger(out):
    """Plan-B: critério de disparo @ 300k steps se ep_rew_mean < 50."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    steps = np.arange(0, 501) * 1000
    # Cenário convergência normal
    np.random.seed(42)
    noise = np.random.randn(501) * 8
    reward_ok = np.clip(-80 + 130 * (1 - np.exp(-steps / 180000)) + noise, -100, 20)
    # Cenário platô (Plan-B)
    noise2 = np.random.randn(501) * 6
    reward_plateau = np.clip(-75 + 30 * (1 - np.exp(-steps / 100000)) + noise2, -100, -20)

    ax.plot(steps / 1000, reward_ok, color=GREEN, lw=1.5, alpha=0.8, label="Convergência normal")
    ax.plot(steps / 1000, reward_plateau, color=RED, lw=1.5, alpha=0.8, label="Platô (Plan-B dispara)")
    ax.axvline(300, ls="--", c=ORANGE, lw=2, label="Plan-B @ 300k steps")
    ax.axhline(-50, ls=":", c=RED, lw=1.5, label="Threshold = −50")
    ax.fill_between([300, 500], [-50, -50], [-100, -100], alpha=0.1, color=RED,
                    label="Zona Plan-B (platô + abaixo do threshold)")
    ax.annotate("Plan-B dispara:\nDroQ ou TD3+BC",
                xy=(300, -70), xytext=(330, -45),
                arrowprops=dict(arrowstyle="->", color=ORANGE),
                fontsize=9, color=ORANGE, fontweight="bold")
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("ep_rew_mean")
    ax.set_title("Critério Plan-B: se ep_rew_mean < −50 @ 300k steps\n"
                 "→ ativar DroQ (dropout+LayerNorm) ou TD3+BC warm-start")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-110, 30)
    ax.set_xlim(0, 500)
    _save(fig, out, "fig_planb_trigger")


def fig_spawn_safety(out):
    """Diagrama spawn seguro: SAFE_SPAWN_MARGIN vs COLLISION_DIST."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Arena
    ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, edgecolor="black", lw=2))

    # Robô no spawn
    robot_x, robot_y = -1.5, -0.5
    robot = Circle((robot_x, robot_y), 0.17, color=BLUE, zorder=5, alpha=0.9)
    ax.add_patch(robot)
    ax.text(robot_x, robot_y, "R", ha="center", va="center", color="white",
            fontweight="bold", fontsize=9)

    # Raios de segurança
    collision = Circle((robot_x, robot_y), 0.15, fill=False,
                       edgecolor=RED, ls="-", lw=2, zorder=6)
    safe_margin = Circle((robot_x, robot_y), 0.35, fill=False,
                         edgecolor=ORANGE, ls="--", lw=2, zorder=6)
    ax.add_patch(collision)
    ax.add_patch(safe_margin)

    ax.annotate("COLLISION_DIST\n= 0.15m", xy=(robot_x + 0.15, robot_y),
                xytext=(robot_x + 0.5, robot_y - 0.6),
                arrowprops=dict(arrowstyle="->", color=RED), fontsize=8, color=RED)
    ax.annotate("SAFE_SPAWN_MARGIN\n= 0.35m", xy=(robot_x + 0.35, robot_y),
                xytext=(robot_x + 0.7, robot_y + 0.5),
                arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=8, color=ORANGE)

    # Obstáculos
    for ox, oy in [(0.8, 0.5), (-0.5, 1.0), (1.2, -0.8), (-1.0, -0.5),
                   (0.3, -1.2), (1.5, 1.2), (-1.5, 1.3)]:
        ax.add_patch(Circle((ox, oy), 0.18, color=GRAY, alpha=0.7, zorder=4))

    # Goal
    ax.scatter([1.5], [1.5], s=200, marker="*", color=GREEN, zorder=6)
    ax.text(1.5, 1.7, "Goal", ha="center", fontsize=9, color=GREEN, fontweight="bold")

    # Legenda
    handles = [
        mpatches.Patch(color=BLUE, label="Robô (spawn)"),
        mpatches.Patch(color=RED, label=f"COLLISION_DIST = 0.15m"),
        mpatches.Patch(color=ORANGE, label=f"SAFE_SPAWN_MARGIN = 0.35m"),
        mpatches.Patch(color=GRAY, label="Obstáculos"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_title("Spawn seguro: min_scan ≥ SAFE_SPAWN_MARGIN\n"
                 "Valida posição antes do primeiro step (smoketest gate)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _save(fig, out, "fig_spawn_safety")


def fig_adaptive_switching(out):
    """Diagrama do switching adaptativo: ρ → A* vs SAC."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(ax, x, y, w, h, label, color, fontsize=10):
        ax.add_patch(Rectangle((x, y), w, h, color=color, alpha=0.85,
                               edgecolor="black", lw=1.5, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

    def arrow(ax, x1, y1, x2, y2, label="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", fontsize=9, color=color)

    # Blocos
    box(ax, 0.3, 2.5, 1.8, 1.0, "Sensor\n(LIDAR)", GRAY)
    box(ax, 2.5, 2.5, 1.8, 1.0, "ρ estimator\n(CBS/densidade)", PURPLE)
    box(ax, 5.0, 4.0, 2.0, 0.9, "A* / Nav2\n(global, ρ < ρ*)", BLUE)
    box(ax, 5.0, 1.0, 2.0, 0.9, "SAC policy\n(local, ρ ≥ ρ*)", RED)
    box(ax, 7.5, 2.5, 2.0, 1.0, "cmd_vel\n(TurtleBot3)", GREEN)

    # Switch
    ax.add_patch(Circle((4.4, 3.0), 0.35, color=ORANGE, zorder=3, ec="black", lw=1.5))
    ax.text(4.4, 3.0, "π(ρ)", ha="center", va="center", color="white",
            fontweight="bold", fontsize=10, zorder=4)

    # Setas
    arrow(ax, 2.1, 3.0, 2.5, 3.0)
    arrow(ax, 4.3, 3.0, 2.5 + 1.8, 3.0)
    arrow(ax, 4.4, 3.35, 5.0, 4.45, "ρ < ρ*=0.30", BLUE)
    arrow(ax, 4.4, 2.65, 5.0, 1.45, "ρ ≥ ρ*=0.30", RED)
    arrow(ax, 7.0, 4.45, 7.5, 3.5)
    arrow(ax, 7.0, 1.45, 7.5, 2.5)

    # Rho star
    ax.text(4.4, 0.3, "ρ* = 0.30\n(crossover point H2)", ha="center",
            fontsize=9, color=ORANGE, fontweight="bold")

    ax.set_title("Arquitetura do adaptive planner — switching por densidade ρ\n"
                 "A* (global) quando espaço livre; SAC (local) em congestão",
                 fontsize=11)
    _save(fig, out, "fig_adaptive_switching")


def fig_world_density_comparison(out):
    """Comparação visual dos 3 mundos: sparse / dense / very_dense."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    configs = [
        ("Sparse\n(ρ ≈ 0.15)", 4, GREEN, "Avaliação H2\n(baseline)"),
        ("Dense\n(ρ ≈ 0.38)", 11, ORANGE, "Treino SAC\n(acima de ρ*)"),
        ("Very Dense\n(ρ ≈ 0.52)", 18, RED, "Avaliação H2\n(stress test)"),
    ]

    np.random.seed(7)
    for ax, (title, n_obs, c, subtitle) in zip(axes, configs):
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, edgecolor="black", lw=2))

        # Paredes internas (2 fixas)
        ax.add_patch(Rectangle((-1.5, -0.1), 1.0, 0.2, color=GRAY, alpha=0.8))
        ax.add_patch(Rectangle((0.5, -0.1), 1.0, 0.2, color=GRAY, alpha=0.8))

        # Obstáculos aleatórios
        placed = []
        for _ in range(n_obs * 10):
            if len(placed) >= n_obs:
                break
            ox = np.random.uniform(-1.7, 1.7)
            oy = np.random.uniform(-1.7, 1.7)
            if all(np.hypot(ox - px, oy - py) > 0.45 for px, py in placed):
                if np.hypot(ox, oy) > 0.5 and np.hypot(ox - 1.5, oy - 1.5) > 0.5:
                    ax.add_patch(Circle((ox, oy), 0.18, color=c, alpha=0.75, zorder=3))
                    placed.append((ox, oy))

        # Robô
        ax.add_patch(Circle((-1.5, -1.5), 0.17, color=BLUE, zorder=5))
        ax.text(-1.5, -1.5, "R", ha="center", va="center", color="white",
                fontweight="bold", fontsize=8)

        # Goal
        ax.scatter([1.5], [1.5], s=150, marker="*", color=GREEN, zorder=6)

        rho = [0.15, 0.38, 0.52][[0, 1, 2][[i for i, (t, *_) in enumerate(configs) if t == title][0]]]
        ax.set_title(f"{title}\nn_obs={n_obs}, ρ≈{rho}", fontweight="bold", color=c)
        ax.text(0, -2.5, subtitle, ha="center", fontsize=8, color=c)
        ax.set_xlabel("x (m)")
        if ax == axes[0]:
            ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.15)
        ax.axvline(0.30, ls=":", c=GRAY, alpha=0.3)

    fig.suptitle("Mundos de avaliação — densidade crescente (sparse → very_dense)\n"
                 "ρ* = 0.30: A* superior à esquerda, SAC superior à direita",
                 fontsize=11)
    _save(fig, out, "fig_world_density_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Gerando figuras de arquitetura SAC em {args.out} ...")
    fig_lidar_downsampling(args.out)
    fig_curriculum_schedule(args.out)
    fig_sac_reward_shaping(args.out)
    fig_planb_trigger(args.out)
    fig_spawn_safety(args.out)
    fig_adaptive_switching(args.out)
    fig_world_density_comparison(args.out)
    print("OK — 7 figuras (.png + .pdf).")


if __name__ == "__main__":
    main()
