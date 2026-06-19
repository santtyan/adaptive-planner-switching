"""
Visualização comparativa: plano CBS discreto vs schedule TPG cinemático.

Mostra como o TPG adiciona buffers de segurança (delta) entre waypoints
de agentes que ocupam a mesma célula em momentos adjacentes — transformando
o plano discreto em schedule executável em robô real.

Gera: paper/figs/cbs_tpg_comparison.png + .pdf
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT_PNG = "paper/figs/cbs_tpg_comparison.png"
DELTA = 0.20   # safety buffer (metros) — mesmo valor do tpg.py original
DT    = 0.50   # segundos por passo de grid

COLORS = ["#1565C0", "#c62828", "#2e7d32"]
NAMES  = ["agent0", "agent1", "agent2"]

# ── Cenário: 3 agentes, grid 5x5, sem obstáculos ──────────────────────────
# Plano CBS (discreto) — produzido pelo CBS real nesse cenário
SCENARIO = {
    "map": {"dimensions": [5, 5], "obstacles": []},
    "agents": [
        {"name": "agent0", "start": [0, 0], "goal": [4, 0]},
        {"name": "agent1", "start": [4, 0], "goal": [0, 0]},
        {"name": "agent2", "start": [2, 4], "goal": [2, 0]},
    ],
}

# Planos CBS resultantes (solução ótima para esse cenário)
CBS_PLANS = {
    "agent0": [
        (0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0),
    ],
    "agent1": [
        (0, 4, 0), (1, 4, 0), (2, 3, 0), (3, 2, 0), (4, 1, 0), (5, 0, 0),
    ],
    "agent2": [
        (0, 2, 4), (1, 2, 3), (2, 2, 2), (3, 2, 1), (4, 2, 0),
    ],
}


def compute_tpg_schedule(cbs_plans, delta=DELTA, dt=DT):
    """
    Constrói schedule TPG: para cada waypoint de cada agente,
    calcula o tempo de execução real adicionando buffers quando
    dois agentes passam pelo mesmo ponto em passos adjacentes.

    Retorna dict agent → lista de (t_real, x, y).
    """
    # Primeiro, converter para posições contínuas (1 grid cell = 1 metro)
    schedule = {}
    for agent, plan in cbs_plans.items():
        schedule[agent] = []
        for (t, x, y) in plan:
            t_real = t * dt
            schedule[agent].append([t_real, float(x), float(y)])

    # Detectar conflitos de passagem e adicionar buffer
    agents = list(cbs_plans.keys())
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            pa, pb = schedule[a], schedule[b]
            for ia in range(len(pa)):
                for ib in range(len(pb)):
                    # Mesma posição em tempo próximo (dentro de 1.5 * dt)
                    xa, ya = pa[ia][1], pa[ia][2]
                    xb, yb = pb[ib][1], pb[ib][2]
                    ta, tb = pa[ia][0], pb[ib][0]
                    if xa == xb and ya == yb and abs(ta - tb) < dt * 1.5:
                        # Agente que chega depois aguarda o buffer
                        if ta > tb:
                            for k in range(ia, len(pa)):
                                pa[k][0] += delta * 2
                        else:
                            for k in range(ib, len(pb)):
                                pb[k][0] += delta * 2

    return schedule


def draw_grid_plan(ax, cbs_plans, obstacles, grid_dim, title):
    """Desenha o plano CBS discreto no grid."""
    gx, gy = grid_dim
    ax.set_xlim(-0.5, gx - 0.5)
    ax.set_ylim(-0.5, gy - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("x (células)", fontsize=9)
    ax.set_ylabel("y (células)", fontsize=9)

    # Grid lines
    for x in range(gx + 1):
        ax.axvline(x - 0.5, color="#cccccc", lw=0.5)
    for y in range(gy + 1):
        ax.axhline(y - 0.5, color="#cccccc", lw=0.5)

    # Obstáculos
    for (ox, oy) in obstacles:
        ax.add_patch(plt.Rectangle((ox - 0.5, oy - 0.5), 1, 1,
                                    color="#444444", zorder=2))

    # Trajetórias
    for idx, (agent, plan) in enumerate(cbs_plans.items()):
        color = COLORS[idx % len(COLORS)]
        xs = [p[1] for p in plan]
        ys = [p[2] for p in plan]
        ax.plot(xs, ys, "o-", color=color, lw=2, markersize=6,
                zorder=3, label=agent)

        # Start e Goal
        ax.plot(xs[0], ys[0], "s", color=color, markersize=10, zorder=4)
        ax.plot(xs[-1], ys[-1], "*", color=color, markersize=14, zorder=4)

        # Timesteps
        for t, x, y in plan:
            ax.text(x + 0.07, y + 0.07, str(t),
                    fontsize=7, color=color, zorder=5)

    ax.legend(loc="upper right", fontsize=8)


def draw_tpg_timeline(ax, cbs_plans, tpg_schedule, title):
    """Desenha timeline: CBS discreto vs TPG cinemático."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_ylabel("Agente", fontsize=9)

    agents = list(cbs_plans.keys())
    y_positions = {a: i for i, a in enumerate(agents)}

    max_t_cbs = max(p[0] * DT for plan in cbs_plans.values() for p in plan)
    max_t_tpg = max(p[0] for plan in tpg_schedule.values() for p in plan)
    ax.set_xlim(-0.3, max(max_t_tpg, max_t_cbs) + 0.5)
    ax.set_ylim(-0.5, len(agents) - 0.5)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    for idx, agent in enumerate(agents):
        color = COLORS[idx % len(COLORS)]
        y = y_positions[agent]
        cbs_plan = cbs_plans[agent]
        tpg_plan = tpg_schedule[agent]

        # CBS discreto — pontos circulares
        for (t, x, xc) in cbs_plan:
            ax.plot(t * DT, y - 0.15, "o", color=color,
                    markersize=8, alpha=0.5, zorder=3)

        # TPG schedule — pontos com buffer
        for (t_real, x, xc) in tpg_plan:
            ax.plot(t_real, y + 0.15, "D", color=color,
                    markersize=7, zorder=4)

        # Linha conectando CBS → TPG
        for i in range(len(cbs_plan)):
            t_cbs = cbs_plan[i][0] * DT
            t_tpg = tpg_plan[i][0]
            if abs(t_tpg - t_cbs) > 0.01:
                ax.annotate("", xy=(t_tpg, y + 0.15),
                            xytext=(t_cbs, y - 0.15),
                            arrowprops=dict(arrowstyle="->",
                                           color=color, lw=1.0, alpha=0.6))

    # Legenda
    cbs_patch = mpatches.Patch(color="#888888",
                                label="CBS discreto (⬤ passos de grid)")
    tpg_patch = mpatches.Patch(color="#333333",
                                label=f"TPG cinemático (◆ com buffer δ={DELTA}m)")
    ax.legend(handles=[cbs_patch, tpg_patch], loc="lower right", fontsize=8)


def draw_safety_buffer(ax, tpg_schedule, title):
    """Trajetórias contínuas no plano x-y com buffers de segurança."""
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(alpha=0.2)

    for idx, (agent, plan) in enumerate(tpg_schedule.items()):
        color = COLORS[idx % len(COLORS)]
        xs = [p[1] for p in plan]
        ys = [p[2] for p in plan]
        ax.plot(xs, ys, "D-", color=color, lw=2, markersize=7,
                label=agent, zorder=3)
        ax.plot(xs[0], ys[0], "s", color=color, markersize=10, zorder=4)
        ax.plot(xs[-1], ys[-1], "*", color=color, markersize=14, zorder=4)

        # Buffer de segurança em cada waypoint
        for (t, x, y) in plan:
            circle = plt.Circle((x, y), DELTA, color=color, alpha=0.08, zorder=2)
            ax.add_patch(circle)

    ax.legend(loc="upper right", fontsize=8)

    # Anotação
    ax.annotate(f"Buffer δ={DELTA}m\n(segurança cinemática)",
                xy=(2.0, 2.0), xytext=(3.0, 3.5),
                fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa"))


def main():
    os.makedirs("paper/figs", exist_ok=True)

    tpg_schedule = compute_tpg_schedule(CBS_PLANS)

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        "CBS Discreto → TPG Cinemático: adição de buffers de segurança para execução em robô real\n"
        f"(grid 5×5, 3 agentes, δ={DELTA}m, {DT}s/passo)",
        fontsize=12, fontweight="bold", y=1.02
    )

    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    draw_grid_plan(ax1, CBS_PLANS,
                   SCENARIO["map"]["obstacles"],
                   SCENARIO["map"]["dimensions"],
                   "① Plano CBS (discreto, grid)\nCoordena colisões no tempo de grid")

    ax2 = fig.add_subplot(gs[1])
    draw_tpg_timeline(ax2, CBS_PLANS, tpg_schedule,
                      "② Timeline: CBS vs TPG\nBuffers adicionados onde há conflito cinemático")

    ax3 = fig.add_subplot(gs[2])
    draw_safety_buffer(ax3, tpg_schedule,
                       "③ Schedule TPG no espaço\nRaio δ = zona de segurança por waypoint")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Salvo: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
