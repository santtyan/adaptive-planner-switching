"""
F1 do docs/PLANO_CORRECAO.md — Pareto sucesso x custo.

Sustenta a única tese que sobrevive à auditoria: sob planejadores reais,
o rho-criterion empata em sucesso com o melhor planejador fixo (A*) a uma
fração do custo de decisão do A*.

Fontes:
  - sucesso: results_abstract/h1_real_2d_mixed_pool.csv (n=1500, regenerado)
  - custo:   results_abstract/h1_real_2d_cost_distribution.json (valores corrigidos, B5)

Rodar: python3 eval/env2d/plot_pareto_success_vs_cost.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    df = pd.read_csv(os.path.join(BASE, "results_abstract", "h1_real_2d_mixed_pool.csv"))
    cost = json.load(open(os.path.join(BASE, "results_abstract", "h1_real_2d_cost_distribution.json")))

    worlds = ["sparse", "dense", "very_dense"]
    labels = {"sparse": "Sparse (ρ baixo)", "dense": "Dense (ρ médio)", "very_dense": "Very dense (ρ alto)"}

    astar_success = [df[df.world == w].astar.mean() * 100 for w in worlds]
    bc_success = [df[df.world == w].bc.mean() * 100 for w in worlds]
    adaptive_success = [df[df.world == w].adaptive.mean() * 100 for w in worlds]

    astar_cost = [np.mean(cost[w]["astar_ms"]) for w in worlds]
    bc_cost = [np.mean(cost[w]["bc_ms"]) for w in worlds]

    # NOTA (B8, docs/PLANO_CORRECAO.md): o custo do adaptativo NÃO está instrumentado --
    # nenhum script mede o tempo real do ρ-criterion (que inclui replanejamento A* a cada
    # switch). Uma primeira tentativa aqui ponderava astar_cost/bc_cost por `used_astar.mean()`,
    # mas `used_astar` é calculado sobre rho0 (densidade só no reset) e em dense/very_dense
    # quase todo trial roteia para BC -- o resultado colava o adaptativo em cima do BC e
    # escondia visualmente o custo de replanejamento. Isso seria enganoso. Omitido até a
    # Etapa 1.4 instrumentar o custo real (perf_counter em astar_policy.reset() por switch).
    adaptive_cost = None

    fig, ax = plt.subplots(figsize=(7.5, 6))

    colors = {"A*": "#1f77b4", "BC": "#ff7f0e"}
    markers = {"sparse": "o", "dense": "s", "very_dense": "^"}

    for i, w in enumerate(worlds):
        ax.scatter(astar_cost[i], astar_success[i], color=colors["A*"], marker=markers[w], s=140,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.scatter(bc_cost[i], bc_success[i], color=colors["BC"], marker=markers[w], s=140,
                   edgecolor="black", linewidth=0.6, zorder=3)

    for name, c in colors.items():
        ax.scatter([], [], color=c, label=name, s=100, edgecolor="black", linewidth=0.6)
    for w, m in markers.items():
        ax.scatter([], [], color="gray", marker=m, label=labels[w], s=100, edgecolor="black", linewidth=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Custo de decisão (ms, escala log)")
    ax.set_ylabel("Taxa de sucesso (%)")
    ax.set_title("Sucesso vs. custo de decisão — A* real vs. BC real\n"
                  "(n=1.500 trials pareados, pool misto de densidades)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_ylim(55, 102)

    ax.annotate("ρ-criterion OMITIDO desta figura:\ncusto do adaptativo ainda não medido\n(ver B8, Etapa 1.4)",
                xy=(0.5, 0.5), xycoords="axes fraction", fontsize=8.5, ha="center", va="center",
                color="#888888", style="italic",
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    out_dir = os.path.join(BASE, "paper", "figs", "pareto")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_pareto_success_vs_cost.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Salvo:", path)

    # versão maior para slides/Canva
    fig.set_size_inches(10, 8)
    slides_dir = os.path.join(BASE, "paper", "figs", "slides")
    os.makedirs(slides_dir, exist_ok=True)
    path = os.path.join(slides_dir, "fig_pareto_success_vs_cost.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo (slides):", path)

    print("\nDados usados:")
    for i, w in enumerate(worlds):
        print(f"  {w}: A*={astar_success[i]:.1f}%/{astar_cost[i]:.2f}ms  "
              f"BC={bc_success[i]:.1f}%/{bc_cost[i]:.3f}ms  "
              f"adapt_sucesso={adaptive_success[i]:.1f}% (custo do adaptativo NÃO plotado -- ver B8)")


if __name__ == "__main__":
    main()
