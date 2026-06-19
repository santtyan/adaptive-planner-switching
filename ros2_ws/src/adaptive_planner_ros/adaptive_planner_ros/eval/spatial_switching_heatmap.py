"""
Heatmap espacial do critério de switching ρ no mapa de navegação.

Mostra, célula a célula, qual planejador o switcher escolheria (A* ou SAC)
baseado na densidade de obstáculos estimada ρ e no threshold ρ*=0.30.

Pode rodar SEM Gazebo ativo — usa modelo de densidade analítico (gaussianas
centradas nos obstáculos) para gerar o mapa ρ e decidir A*/SAC.

Uso:
    python3 spatial_switching_heatmap.py --out paper/figs/
    python3 spatial_switching_heatmap.py --rho-star 0.30 --resolution 0.05 --out paper/figs/

Saídas:
    paper/figs/switching_heatmap.png   — heatmap A*/SAC + contorno ρ*
    paper/figs/switching_heatmap.pdf   — versão PDF para LaTeX
    paper/figs/density_heatmap.png     — mapa de densidade ρ bruto
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# ── Geometria do mapa (igual ao record_episode.py) ──────────────────────────
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

# Raio de influência de cada obstáculo no campo de densidade (σ do gaussiano)
DENSITY_SIGMA = 0.50  # m — quanto cada obstáculo "contamina" a vizinhança

RHO_STAR_DEFAULT = 0.30


def compute_density_map(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Densidade ρ em cada célula (xs, ys): soma de gaussianas dos obstáculos, clipada em [0,1]."""
    rho = np.zeros_like(xs)
    for ox, oy, _ in OBSTACLES:
        d2 = (xs - ox) ** 2 + (ys - oy) ** 2
        rho += np.exp(-d2 / (2 * DENSITY_SIGMA ** 2))
    return np.clip(rho, 0.0, 1.0)


def draw_obstacles(ax, alpha: float = 0.85):
    for ox, oy, r in OBSTACLES:
        c = plt.Circle((ox, oy), r, color="#444444", alpha=alpha, zorder=5)
        ax.add_patch(c)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rho-star", type=float, default=RHO_STAR_DEFAULT,
                   help="Threshold de switching (padrão 0.30)")
    p.add_argument("--resolution", type=float, default=0.04,
                   help="Tamanho de célula em metros (padrão 0.04)")
    p.add_argument("--sigma", type=float, default=DENSITY_SIGMA,
                   help="Raio de influência de cada obstáculo (m)")
    p.add_argument("--out", default="paper/figs",
                   help="Diretório de saída para as figuras")
    p.add_argument("--no-pdf", action="store_true",
                   help="Não gerar PDF (só PNG)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    global DENSITY_SIGMA
    DENSITY_SIGMA = args.sigma

    # Grade de células
    edge = np.arange(-ARENA, ARENA + args.resolution, args.resolution)
    xs, ys = np.meshgrid(edge, edge)

    rho = compute_density_map(xs, ys)

    # ── Figura 1: heatmap de densidade ρ ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.contourf(xs, ys, rho, levels=50, cmap="YlOrRd", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Densidade de obstáculos ρ", fontsize=11)
    ax.contour(xs, ys, rho, levels=[args.rho_star], colors=["#1a6faf"],
               linewidths=2.0, linestyles="--")
    draw_obstacles(ax)
    ax.set_xlim(-ARENA - 0.1, ARENA + 0.1)
    ax.set_ylim(-ARENA - 0.1, ARENA + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_title(f"Mapa de Densidade ρ  (σ={args.sigma:.2f} m)", fontsize=12, fontweight="bold")
    legend_line = Line2D([0], [0], color="#1a6faf", lw=2, ls="--",
                         label=f"ρ* = {args.rho_star:.2f} (threshold)")
    ax.legend(handles=[legend_line], loc="upper right", fontsize=9)
    fig.tight_layout()
    path_density = os.path.join(args.out, "density_heatmap.png")
    fig.savefig(path_density, dpi=150, bbox_inches="tight")
    print(f"Salvo: {path_density}")
    plt.close(fig)

    # ── Figura 2: heatmap de switching A*/SAC ───────────────────────────────
    # 0 = A* (ρ < ρ*),  1 = SAC (ρ ≥ ρ*)
    decision = (rho >= args.rho_star).astype(float)

    cmap_switch = ListedColormap(["#4393c3", "#d6604d"])  # azul=A*, vermelho=SAC

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.pcolormesh(xs, ys, decision, cmap=cmap_switch, shading="auto",
                  vmin=0, vmax=1, alpha=0.75, zorder=1)

    # Contorno do threshold
    ax.contour(xs, ys, rho, levels=[args.rho_star], colors=["#111111"],
               linewidths=2.0, linestyles="--", zorder=3)

    draw_obstacles(ax, alpha=0.9)

    ax.set_xlim(-ARENA - 0.1, ARENA + 0.1)
    ax.set_ylim(-ARENA - 0.1, ARENA + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(
        f"Decisão de Switching: A* vs SAC  (ρ* = {args.rho_star:.2f})",
        fontsize=12, fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor="#4393c3", alpha=0.75, label="A* (ρ < ρ*)"),
        mpatches.Patch(facecolor="#d6604d", alpha=0.75, label="SAC (ρ ≥ ρ*)"),
        Line2D([0], [0], color="#111111", lw=2, ls="--",
               label=f"Fronteira ρ* = {args.rho_star:.2f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    fig.tight_layout()
    path_switch = os.path.join(args.out, "switching_heatmap.png")
    fig.savefig(path_switch, dpi=150, bbox_inches="tight")
    print(f"Salvo: {path_switch}")

    if not args.no_pdf:
        path_pdf = os.path.join(args.out, "switching_heatmap.pdf")
        fig.savefig(path_pdf, bbox_inches="tight")
        print(f"Salvo: {path_pdf}")

    plt.close(fig)

    # ── Resumo ───────────────────────────────────────────────────────────────
    total_cells = decision.size
    sac_cells = int(decision.sum())
    astar_cells = total_cells - sac_cells
    print(
        f"\nResumo: {astar_cells}/{total_cells} células → A* ({100*astar_cells/total_cells:.1f}%) | "
        f"{sac_cells}/{total_cells} → SAC ({100*sac_cells/total_cells:.1f}%)"
    )


if __name__ == "__main__":
    main()
