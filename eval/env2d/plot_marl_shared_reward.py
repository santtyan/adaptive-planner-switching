"""
plot_marl_shared_reward.py — Figura comparando RL independente vs MARL
centralizado com reward compartilhada (N=4, sparse), 09/07/2026.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "paper", "figs", "marl")
os.makedirs(FIGS, exist_ok=True)

labels = ["RL independente\n(SAC)", "MARL centralizado\n150k passos", "MARL centralizado\n600k passos"]
goal = [57, 25, 50]
coll = [60, 0, 0]
colors = ["#2196F3", "#FFA726", "#EF5350"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
ax.bar(labels, goal, color=colors)
ax.set_ylabel("Taxa de chegada ao goal (%)")
ax.set_ylim(0, 100)
ax.set_title("(a) Taxa de goal")
for i, v in enumerate(goal):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=9)

ax = axes[1]
ax.bar(labels, coll, color=colors)
ax.set_ylabel("Colisão inter-robô (%)")
ax.set_ylim(0, 100)
ax.set_title("(b) Colisão entre robôs")
for i, v in enumerate(coll):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=9)

fig.suptitle("RL independente vs. MARL centralizado (reward compartilhada) — N=4, sparse, dados reais",
             fontsize=11, fontweight="bold")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(FIGS, f"fig_marl_shared_reward_comparison.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close()
print("Figura salva em paper/figs/marl/fig_marl_shared_reward_comparison.png/.pdf")
