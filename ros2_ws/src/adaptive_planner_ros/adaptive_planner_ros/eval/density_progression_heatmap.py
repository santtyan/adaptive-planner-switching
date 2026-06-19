"""
Figura comparativa de progressão de densidade — 3 worlds lado a lado.

Mostra, para cada nível de dificuldade (sparse / medium / very_dense),
o mapa de densidade ρ e qual planejador o switcher escolhe (A* ou SAC)
com ρ*=0.30.

Uso:
    python3 density_progression_heatmap.py --out paper/figs/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ── Definição dos três ambientes ─────────────────────────────────────────────

WORLDS = {
    "Esparso\n(3 obstáculos, ρ̄≈0.10)": [
        (-1.2,  1.2, 0.18),
        ( 1.0,  0.0, 0.18),
        (-0.3, -1.3, 0.18),
    ],
    "Médio\n(7 obstáculos, ρ̄≈0.35)": [
        (-1.2,  1.2, 0.18),
        (-0.6,  1.5, 0.18),
        ( 1.0,  0.3, 0.18),
        ( 1.4, -0.3, 0.18),
        (-0.2, -0.5, 0.18),
        ( 0.5, -1.1, 0.18),
        (-1.0, -1.0, 0.18),
    ],
    "Denso\n(13 obstáculos, ρ̄≈0.50)": [
        (-1.5,  1.5, 0.18), (-0.7,  1.5, 0.18), ( 0.5,  1.5, 0.18), ( 1.4,  1.5, 0.18),
        (-1.3,  0.5, 0.18), ( 0.1,  0.5, 0.18), ( 1.3,  0.5, 0.18),
        (-1.6, -0.4, 0.18), (-0.6, -0.4, 0.18), ( 0.5, -0.4, 0.18), ( 1.5, -0.4, 0.18),
        (-1.0, -1.4, 0.18), ( 0.8, -1.4, 0.18),
    ],
}

ARENA      = 2.0
RHO_STAR   = 0.30
SIGMA      = 0.50
RESOLUTION = 0.04


def density_map(xs, ys, obstacles):
    rho = np.zeros_like(xs)
    for ox, oy, _ in obstacles:
        d2 = (xs - ox) ** 2 + (ys - oy) ** 2
        rho += np.exp(-d2 / (2 * SIGMA ** 2))
    return np.clip(rho, 0.0, 1.0)


def draw_obstacles(ax, obstacles, alpha=0.85):
    for ox, oy, r in obstacles:
        c = plt.Circle((ox, oy), r, color="#222222", alpha=alpha, zorder=5)
        ax.add_patch(c)


def mean_rho(rho_grid):
    return float(np.mean(rho_grid))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs")
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    coords = np.arange(-ARENA, ARENA, RESOLUTION)
    xs, ys = np.meshgrid(coords, coords)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Progressão de densidade de obstáculos — decisão do switcher (ρ* = 0,30)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    cmap_switch = ListedColormap(["#2166ac", "#d73027"])  # azul=A*, vermelho=SAC
    cmap_density = "YlOrRd"

    for ax, (title, obstacles) in zip(axes, WORLDS.items()):
        rho = density_map(xs, ys, obstacles)
        decision = (rho >= RHO_STAR).astype(float)  # 0=A*, 1=SAC
        rho_mean = mean_rho(rho)

        # painel de switching
        im = ax.imshow(
            decision,
            origin="lower",
            extent=[-ARENA, ARENA, -ARENA, ARENA],
            cmap=cmap_switch,
            vmin=0, vmax=1,
            alpha=0.65,
            interpolation="bilinear",
        )

        # contorno da fronteira de decisão
        ax.contour(
            xs, ys, rho,
            levels=[RHO_STAR],
            colors=["black"],
            linewidths=1.5,
            linestyles="--",
            zorder=4,
        )

        draw_obstacles(ax, obstacles)

        # borda da arena
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("#333333")

        ax.set_xlim(-ARENA, ARENA)
        ax.set_ylim(-ARENA, ARENA)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)

        # ρ̄ médio no canto
        ax.text(
            0.97, 0.04, f"ρ̄ = {rho_mean:.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#333333", alpha=0.7),
        )

    # legenda compartilhada
    legend_handles = [
        mpatches.Patch(color="#2166ac", label="A* (ρ < 0,30)"),
        mpatches.Patch(color="#d73027", label="SAC (ρ ≥ 0,30)"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                   label="Fronteira de decisão (ρ*)"),
        mpatches.Patch(color="#222222", label="Obstáculo"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.9,
    )

    plt.tight_layout()

    out_png = os.path.join(args.out, "density_progression.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Salvo: {out_png}")

    if not args.no_pdf:
        out_pdf = os.path.join(args.out, "density_progression.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Salvo: {out_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    main()
