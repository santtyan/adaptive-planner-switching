"""
plot_h1_real_validation.py — Figura da revalidação de H1 com planejadores
reais no gêmeo 2D (09/07/2026): taxa de sucesso E custo computacional,
medidos no MESMO ambiente/trials — substitui a motivação por taxa de
sucesso (não se sustenta com dados reais) pela motivação por custo
computacional (confirmada com dados reais, mais forte que a versão mock).

Fontes:
    results_abstract/h1_real_2d_mixed_pool.csv       (sucesso, pool misto pareado)
    results_abstract/h1_real_2d_cost_distribution.json  (custo, 150 amostras A*/300 BC por mundo)

NOTA (correção B5, docs/PLANO_CORRECAO.md, 25/07/2026): esta figura tinha os valores de custo
hardcoded (9,32/26,44/32,73 ms) que não batiam com o JSON de distribuição de custo (8,00/20,90/
29,21 ms) -- provavelmente herdados de uma execução anterior ao arquivo atual. Corrigido para ler
do JSON diretamente, eliminando a duplicação de fonte de verdade.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs")

df = pd.read_csv(os.path.join(RES, "h1_real_2d_mixed_pool.csv"))
cost_dist = json.load(open(os.path.join(RES, "h1_real_2d_cost_distribution.json")))
worlds = ["sparse", "dense", "very_dense"]
labels = ["Sparse\n(ρ baixo)", "Dense\n(ρ médio)", "Very dense\n(ρ alto)"]

astar_sr = [df[df.world == w].astar.mean() * 100 for w in worlds]
bc_sr = [df[df.world == w].bc.mean() * 100 for w in worlds]

# custo lido de h1_real_2d_cost_distribution.json (150 amostras A* / 300 BC por mundo)
astar_cost_ms = [float(np.mean(cost_dist[w]["astar_ms"])) for w in worlds]
bc_cost_ms = [float(np.mean(cost_dist[w]["bc_ms"])) for w in worlds]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
x = np.arange(len(worlds))
w = 0.35
ax.bar(x - w/2, astar_sr, w, label="A* (real)", color="#4C72B0")
ax.bar(x + w/2, bc_sr, w, label="BC (real, casado por densidade)", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Taxa de sucesso (%)")
ax.set_ylim(0, 105)
ax.set_title("(a) Sucesso — A* real permanece à frente em todo regime")
ax.legend(fontsize=8)
for i, (a, b) in enumerate(zip(astar_sr, bc_sr)):
    ax.text(i - w/2, a + 2, f"{a:.0f}%", ha="center", fontsize=8)
    ax.text(i + w/2, b + 2, f"{b:.0f}%", ha="center", fontsize=8)

ax = axes[1]
ax.plot(x, astar_cost_ms, "o-", color="#4C72B0", label="A* (busca completa)", lw=2)
ax.plot(x, bc_cost_ms, "s-", color="#55A868", label="BC (1 forward pass)", lw=2)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Custo de decisão (ms, log)")
ratio_very_dense = astar_cost_ms[-1] / bc_cost_ms[-1]
ax.set_title(f"(b) Custo — BC ~{ratio_very_dense:.0f}× mais barato em very_dense")
ax.legend(fontsize=8)
for i, (a, b) in enumerate(zip(astar_cost_ms, bc_cost_ms)):
    ax.annotate(f"{a:.1f}ms", (i, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.annotate(f"{b:.3f}ms", (i, b), textcoords="offset points", xytext=(5, -12), fontsize=8)

fig.suptitle("Revalidação de H1 com planejadores REAIS — mesmo ambiente, mesmos trials (09/07/2026)",
             fontsize=12, fontweight="bold")
fig.tight_layout()

for ext in ["png", "pdf"]:
    path = os.path.join(FIGS, "2d", f"fig_2d_h1_real_validation.{ext}")
    fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
plt.close()
print("Figura salva em paper/figs/2d/fig_2d_h1_real_validation.png/.pdf")
