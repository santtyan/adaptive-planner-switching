"""
Figura de sucesso vs. condição dinâmica no cenário urbano (fecha a lacuna
"múltiplos obstáculos dinâmicos", sessão 25/07/2026, docs/PLANO_CORRECAO.md,
objetivo 2 do Plano de Trabalho PI08078-2024).

Fonte: results_abstract/urban_grid_results.csv (4 condições, n=500 cada,
gerado por eval/env2d/rerun_urban.py)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs")


def main():
    df = pd.read_csv(os.path.join(RES, "urban_grid_results.csv"))

    conditions = ["static", "dynamic", "dynamic_multi", "dynamic_fast"]
    labels = ["Estático\n(sem obstáculo)", "1 obstáculo\n(original)",
              "3 obstáculos\n(múltiplos eixos)", "3 obstáculos\n(2x velocidade)"]
    methods = ["astar", "bc", "adaptive"]
    method_labels = {"astar": "A* real", "bc": "BC real", "adaptive": "ρ-criterion"}
    colors = {"astar": "#4C72B0", "bc": "#DD8452", "adaptive": "#55A868"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(conditions))
    width = 0.25

    for i, m in enumerate(methods):
        vals = [df[df.condition == c][m].mean() * 100 for c in conditions]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=method_labels[m], color=colors[m])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Taxa de sucesso (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cenário urbano: degradação de sucesso com número e velocidade\n"
                  "de obstáculos dinâmicos (n=500 trials/condição, dados reais)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(FIGS, "2d")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_2d_urban_dynamic_multi.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Salvo:", path)

    fig.set_size_inches(11, 6.5)
    slides_dir = os.path.join(FIGS, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    path = os.path.join(slides_dir, "fig_2d_urban_dynamic_multi.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo (slides):", path)

    print("\nResumo:")
    for c, lbl in zip(conditions, labels):
        row = df[df.condition == c]
        n = len(row)
        vals = {m: row[m].mean() * 100 for m in methods}
        print(f"  {c:15s} (n={n}): A*={vals['astar']:.1f}%  BC={vals['bc']:.1f}%  "
              f"adapt={vals['adaptive']:.1f}%")


if __name__ == "__main__":
    main()
