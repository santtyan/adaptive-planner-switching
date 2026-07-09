"""
plot_h1_boxplots.py — Boxplots de distribuição e outliers para a
revalidação de H1 com planejadores reais (09/07/2026). Complementa
fig_2d_h1_real_validation.png (que só mostra médias) com a distribuição
completa: custo de decisão do A* real (que tem outliers reais — alguns
trials exigem busca bem mais longa que a média) e do BC (praticamente sem
variância, forward pass constante), e a distribuição de ρ_local por mundo.

Fontes:
    results_abstract/h1_real_2d_cost_distribution.json  (150 trials A*, 300 BC, por mundo)
    results_abstract/h1_real_2d_mixed_pool.csv            (500 trials, ρ_local por mundo)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs", "2d")

worlds = ["sparse", "dense", "very_dense"]
labels = ["Sparse", "Dense", "Very dense"]

with open(os.path.join(RES, "h1_real_2d_cost_distribution.json")) as f:
    cost = json.load(f)

df = pd.read_csv(os.path.join(RES, "h1_real_2d_mixed_pool.csv"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# (a) Custo A* — boxplot com outliers
ax = axes[0]
astar_data = [cost[w]["astar_ms"] for w in worlds]
bp = ax.boxplot(astar_data, labels=labels, patch_artist=True, showmeans=True,
                 flierprops=dict(marker="o", markerfacecolor="#C44E52", markersize=4, alpha=0.6))
for patch in bp["boxes"]:
    patch.set_facecolor("#4C72B0"); patch.set_alpha(0.5)
ax.set_ylabel("Custo de decisão A* (ms)")
ax.set_title(f"(a) Custo A* real — n={len(astar_data[0])}/mundo\ncírculos = outliers")
for i, d in enumerate(astar_data):
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    n_out = int(np.sum(np.array(d) > q3 + 1.5 * iqr))
    ax.annotate(f"{n_out} outlier(s)", (i + 1, max(d)), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=7, color="#C44E52")

# (b) Custo BC — mesma escala pra comparação visual direta
ax = axes[1]
bc_data = [cost[w]["bc_ms"] for w in worlds]
bp = ax.boxplot(bc_data, labels=labels, patch_artist=True, showmeans=True,
                 flierprops=dict(marker="o", markerfacecolor="#55A868", markersize=3, alpha=0.5))
for patch in bp["boxes"]:
    patch.set_facecolor("#55A868"); patch.set_alpha(0.5)
ax.set_ylabel("Custo de decisão BC (ms)")
ax.set_title(f"(b) Custo BC real — n={len(bc_data[0])}/mundo\npraticamente sem variância")

# (c) rho_local por mundo, colorido por sucesso/falha do adaptativo
ax = axes[2]
rho_data = [df[df.world == w].rho0.values for w in worlds]
bp = ax.boxplot(rho_data, labels=labels, patch_artist=True, showmeans=True,
                 flierprops=dict(marker="o", markerfacecolor="#8172B2", markersize=4, alpha=0.6))
for patch in bp["boxes"]:
    patch.set_facecolor("#8172B2"); patch.set_alpha(0.5)
ax.axhline(0.30, color="red", ls="--", lw=1.2, label="ρ*=0,30 (limiar original)")
ax.set_ylabel("ρ_local no reset (fração LIDAR < 1m)")
ax.set_title("(c) Distribuição de ρ_local por mundo\n(500 trials, pool misto)")
ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Distribuição e outliers — revalidação de H1 com planejadores reais (09/07/2026)",
             fontsize=12, fontweight="bold")
fig.tight_layout()

for ext in ["png", "pdf"]:
    path = os.path.join(FIGS, f"fig_2d_h1_boxplots.{ext}")
    fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
plt.close()
print("Figura salva em paper/figs/2d/fig_2d_h1_boxplots.png/.pdf")

# Relatório textual de outliers (pra citar no texto)
print("\n=== Outliers do custo A* (> Q3 + 1.5*IQR) ===")
for w, d in zip(worlds, astar_data):
    d = np.array(d)
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    outliers = d[d > q3 + 1.5 * iqr]
    print(f"  {w}: mediana={np.median(d):.1f}ms, Q3={q3:.1f}ms, "
          f"{len(outliers)} outlier(s) acima de {q3+1.5*iqr:.1f}ms, max={d.max():.1f}ms")
