"""
plot_urban_cost.py — Custo de decisão (ms/passo) por condição no cenário
urbano (urban_grid), A*/BC/critério adaptativo, dado real instrumentado
via rerun_urban.py --measure_time.

Lê results_abstract/urban_grid_results.csv (colunas *_decision_ms) e gera
fig_urban_cost.png/pdf — mesma paleta de cor de plot_pareto_success_vs_cost.py
(A*=#1f77b4, BC=#ff7f0e) para consistência visual entre os slides de custo.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "results_abstract", "urban_grid_results.csv")
FIGS = os.path.join(ROOT, "paper", "figs", "2d")

CONDITION_ORDER = ["static", "dynamic", "dynamic_multi", "dynamic_fast"]
CONDITION_LABELS = {
    "static": "Estático\n(sem obstáculo)",
    "dynamic": "1 obstáculo\n(original)",
    "dynamic_multi": "3 obstáculos\n(múltiplos eixos)",
    "dynamic_fast": "3 obstáculos\n(2x velocidade)",
}

COLORS = {"A*": "#1f77b4", "BC": "#ff7f0e", "Adaptativo": "#2ca02c"}


def main():
    df = pd.read_csv(CSV)

    means = {"A*": [], "BC": [], "Adaptativo": []}
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        means["A*"].append(sub["astar_decision_ms"].mean())
        means["BC"].append(sub["bc_decision_ms"].mean())
        means["Adaptativo"].append(sub["adaptive_decision_ms"].mean())

    x = np.arange(len(CONDITION_ORDER))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for i, (method, vals) in enumerate(means.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=COLORS[method],
                       edgecolor="black", linewidth=0.6, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=9)
    ax.set_ylabel("Custo de decisão (ms/passo)", fontsize=11)
    ax.set_title("Cenário urbano: custo de decisão real por condição\n"
                 "(n=500 trials/condição, dados reais, perf_counter)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig_urban_cost.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_urban_cost.png/pdf")


if __name__ == "__main__":
    main()
