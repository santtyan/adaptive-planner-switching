"""
Curva de aprendizado ao vivo — lê /tmp/rew_values.txt e /tmp/step_values.txt
gerados por:
  docker compose logs train-all | grep "ep_rew_mean" | grep -oP ... > /tmp/rew_values.txt
  docker compose logs train-all | grep "total_timesteps" | grep -oP ... > /tmp/step_values.txt

Gera: paper/figs/fig_sac_learning_curve_live.png + .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

OUT = "paper/figs/fig_sac_learning_curve_live.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

steps = np.loadtxt("/tmp/step_values.txt")
rews  = np.loadtxt("/tmp/rew_values.txt")

# Garantir mesma extensão
n = min(len(steps), len(rews))
steps, rews = steps[:n], rews[:n]

# Suavização (janela 20 pontos)
window = min(20, n // 4) if n > 8 else 1
rews_smooth = uniform_filter1d(rews, size=window)

# Linha de learning_starts
ls_idx = np.searchsorted(steps, 10000)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
fig.patch.set_facecolor("white")

# Área de learning_starts
ax.axvspan(0, 10000, alpha=0.08, color="#888888", label="learning_starts (coleta aleatória)")
ax.axvline(10000, color="#888888", lw=1.2, ls="--")
ax.text(10200, rews.max() * 0.92, "SAC começa\na aprender", fontsize=8,
        color="#555555", va="top")

# Dados brutos
ax.plot(steps, rews, color="#aaccff", lw=0.7, alpha=0.5, label="ep_rew_mean (bruto)")
# Suavizado
ax.plot(steps, rews_smooth, color="#1565C0", lw=2.2, label=f"suavizado (janela={window})")

# Melhor ponto
best_idx = np.argmax(rews_smooth)
ax.scatter(steps[best_idx], rews_smooth[best_idx], color="#c62828", s=80, zorder=5,
           label=f"melhor: {rews_smooth[best_idx]:.0f} @ {steps[best_idx]/1000:.0f}k")

ax.set_xlabel("Timesteps", fontsize=11)
ax.set_ylabel("ep_rew_mean (reward por episódio)", fontsize=11)
ax.set_title("Curva de Aprendizado SAC — TurtleBot3 Waffle em Gazebo (ao vivo)\n"
             "Obs: 29-dim | Reward: potencial + heading + obstacle | Curriculum: 1→3 m",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)

# Anotação do total de steps
ax.annotate(f"Atual: {steps[-1]/1000:.1f}k steps\n~{steps[-1]/500000*100:.0f}% do teto",
            xy=(steps[-1], rews_smooth[-1]),
            xytext=(steps[-1] - len(steps)*0.3, rews_smooth[-1] - abs(rews.min())*0.15),
            fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa"))

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Salvo: {OUT}  ({n} pontos, {steps[-1]/1000:.1f}k steps)")
plt.close()
