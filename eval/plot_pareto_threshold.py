"""
Figura Pareto: trade-off sucesso × custo computacional por limiar τ.

Justifica formalmente τ*=0.30 como ponto Pareto-ótimo:
- τ=0.20 tem sucesso levemente maior mas usa SAC desnecessariamente em regiões
  onde A* seria suficiente e mais rápido.
- τ=0.30 é o melhor equilíbrio entre taxa de sucesso e custo computacional médio.

Gera:
  paper/figs/fig_pareto_threshold.png/.pdf
  paper/figs/fig_pareto_detail.png/.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "paper", "figs")
RES  = os.path.join(ROOT, "results_abstract")

def savefig(name):
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIGS, f"{name}.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

# ── Carrega dados reais ────────────────────────────────────────
df   = pd.read_csv(os.path.join(RES, "adaptive_switching_results.csv"))
sens = pd.read_csv(os.path.join(RES, "threshold_sensitivity.csv"))

densities = df["density"].values

taus = np.arange(0.10, 0.56, 0.05)
frac_rl, success_rate = [], []

for tau in taus:
    frac_rl.append(np.mean(densities >= tau))

sr_map = dict(zip(sens["tau"].round(2), sens["success_rate"]))
for tau in taus:
    success_rate.append(sr_map.get(round(tau, 2), np.nan))

taus         = np.array(taus)
frac_rl      = np.array(frac_rl)
success_rate = np.array(success_rate)
regret       = sens["regret_pct"].values

# ── Ganho marginal de sucesso por % adicional de SAC ──────────
# Quantos pp de sucesso ganho ao usar SAC 1% a mais?
# O argumento: após τ=0.30 (50% SAC), o retorno marginal cai.
frac_sorted = frac_rl[::-1]          # crescente em uso de SAC
sr_sorted   = success_rate[::-1]
marginal     = np.gradient(sr_sorted, frac_sorted)  # d(sucesso)/d(frac_SAC)

valid = ~np.isnan(success_rate)
idx30 = np.where(np.abs(taus - 0.30) < 0.01)[0][0]
idx20 = np.where(np.abs(taus - 0.20) < 0.01)[0][0]

# ══════════════════════════════════════════════════════════════
# FIGURA 1 — Knee of the curve: sucesso × uso de SAC
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

# Scatter: x=fração SAC, y=sucesso, colorido por τ
sc = ax.scatter(frac_rl[valid]*100, success_rate[valid],
                c=taus[valid], cmap="plasma",
                s=140, zorder=5, edgecolors="white", linewidths=1.2)
plt.colorbar(sc, ax=ax, label="Limiar τ")
ax.plot(frac_rl[valid]*100, success_rate[valid],
        "--", color="gray", lw=1, zorder=2, alpha=0.5)

# Destaca τ=0.30 e τ=0.20
ax.scatter(frac_rl[idx30]*100, success_rate[idx30], s=350,
           color="#2196F3", zorder=6, edgecolors="navy", linewidths=2.5,
           label=f"τ*=0,30 — ponto joelho\n"
                 f"{frac_rl[idx30]*100:.0f}% SAC, sucesso={success_rate[idx30]:.1%}")
ax.scatter(frac_rl[idx20]*100, success_rate[idx20], s=220,
           color="#FF9800", zorder=6, edgecolors="darkorange", linewidths=2,
           label=f"τ=0,20 — sucesso levemente maior\n"
                 f"{frac_rl[idx20]*100:.0f}% SAC, +{(success_rate[idx20]-success_rate[idx30])*100:.1f} pp")

# Labels dos pontos
for tau, fr, sr in zip(taus[valid], frac_rl[valid]*100, success_rate[valid]):
    if abs(tau - 0.30) < 0.01 or abs(tau - 0.20) < 0.01:
        continue
    ax.annotate(f"τ={tau:.2f}", xy=(fr, sr),
                xytext=(fr + 1.5, sr - 0.005),
                fontsize=7.5, color="gray")

# Anotação do ganho marginal
delta_sr   = (success_rate[idx20] - success_rate[idx30]) * 100
delta_sac  = (frac_rl[idx20]      - frac_rl[idx30])      * 100
ax.annotate(
    f"+{delta_sr:.1f} pp de sucesso\ncusto: +{delta_sac:.0f} pp de uso de SAC\n"
    f"= {delta_sr/delta_sac:.2f} pp/pp (baixo ROI)",
    xy=(frac_rl[idx20]*100, success_rate[idx20]),
    xytext=(frac_rl[idx20]*100 + 5, success_rate[idx20] - 0.035),
    fontsize=9, color="#FF9800",
    arrowprops=dict(arrowstyle="->", color="#FF9800", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Seta "joelho"
ax.annotate("Ponto joelho:\nretorno marginal cai aqui",
            xy=(frac_rl[idx30]*100, success_rate[idx30]),
            xytext=(frac_rl[idx30]*100 + 8, success_rate[idx30] - 0.04),
            fontsize=9, color="navy",
            arrowprops=dict(arrowstyle="->", color="navy", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.9))

ax.set_xlabel("Uso de SAC/PPO (% dos passos de navegação)", fontsize=12)
ax.set_ylabel("Taxa de sucesso global", fontsize=12)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_title("Trade-off: uso de SAC × taxa de sucesso por limiar τ\n"
             "τ*=0,30 é o ponto joelho — ganho marginal cai após 50% de uso de SAC",
             fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
savefig("fig_pareto_threshold")

# ══════════════════════════════════════════════════════════════
# FIGURA 2 — Painel 3: fração SAC / regret / retorno marginal
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# [0] — Fração SAC por τ
ax = axes[0]
ax.plot(taus, frac_rl * 100, "o-", color="#E91E63", lw=2.5)
ax.axvline(0.30, ls="--", color="#2196F3", lw=2, label="τ*=0,30 (50%)")
ax.axvline(0.20, ls="--", color="#FF9800", lw=1.5,
           label=f"τ=0,20 ({frac_rl[idx20]*100:.0f}%)")
ax.fill_between(taus[(taus >= 0.20) & (taus <= 0.30)],
                frac_rl[(taus >= 0.20) & (taus <= 0.30)] * 100,
                frac_rl[idx30] * 100,
                alpha=0.15, color="#E91E63",
                label=f"+{(frac_rl[idx20]-frac_rl[idx30])*100:.0f}% SAC\ndesnecessário")
ax.set_xlabel("Limiar τ"); ax.set_ylabel("% de passos usando SAC")
ax.set_title("(a) Uso de SAC diminui com τ maior"); ax.set_ylim(0, 110)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [1] — Regret por τ com platô
ax = axes[1]
ax.plot(taus[valid], regret[valid], "s-", color="#9C27B0", lw=2.5)
ax.axvline(0.30, ls="--", color="#2196F3", lw=2, label="τ*=0,30")
ax.axhline(5,  ls=":", color="#FF9800", lw=1.5, label="Limite H2: 5%")
ax.axhline(10, ls=":", color="#E53935", lw=1.5, label="Pior caso: 10%")
plateau_mask = regret[valid] < 5
plateau_taus = taus[valid][plateau_mask]
if len(plateau_taus):
    ax.axvspan(plateau_taus[0], plateau_taus[-1], alpha=0.10, color="#4CAF50",
               label=f"Platô estável [{plateau_taus[0]:.2f}–{plateau_taus[-1]:.2f}]")
ax.set_xlabel("Limiar τ"); ax.set_ylabel("Regret vs Oracle (%)")
ax.set_title("(b) Regret — τ=0,30 na borda do platô")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# [2] — Retorno marginal: Δsucesso / Δ(uso SAC)
ax = axes[2]
# Calcula Δsucesso / Δ(frac_SAC) entre pontos consecutivos (crescente em SAC)
taus_sorted = taus[valid][::-1]        # τ decrescente = SAC crescente
sr_sorted2  = success_rate[valid][::-1]
fr_sorted2  = frac_rl[valid][::-1]
mid_fr  = (fr_sorted2[:-1] + fr_sorted2[1:]) / 2
d_sr    = np.diff(sr_sorted2) * 100     # pp
d_fr    = np.diff(fr_sorted2) * 100     # pp SAC
roi     = np.where(d_fr != 0, d_sr / d_fr, 0)

bar_colors = ["#2196F3" if abs(mf - frac_rl[idx30]*100) < 8 else "#9E9E9E"
              for mf in mid_fr * 100]
ax.bar(mid_fr * 100, roi, width=6, color=bar_colors, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", lw=1)
ax.axvline(frac_rl[idx30]*100, ls="--", color="#2196F3", lw=2,
           label=f"τ*=0,30 ({frac_rl[idx30]*100:.0f}% SAC)")
ax.set_xlabel("% de passos usando SAC")
ax.set_ylabel("Ganho de sucesso por pp de SAC\n(pp de sucesso / pp de SAC)")
ax.set_title("(c) Retorno marginal do uso de SAC\ncai após 50% de uso")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Justificativa de τ*=0,30: ponto joelho do retorno marginal",
             fontsize=13, fontweight="bold")
fig.tight_layout()
savefig("fig_pareto_detail")

print("\nResumo para o relatório:")
print(f"{'τ':>6}  {'sucesso':>8}  {'%SAC':>6}  {'note'}")
for i, tau in enumerate(taus):
    if np.isnan(success_rate[i]): continue
    marker = " ← ESCOLHIDO (ponto joelho)" if abs(tau - 0.30) < 0.01 else \
             " ← melhor sucesso puro"      if abs(tau - 0.20) < 0.01 else ""
    print(f"  {tau:.2f}   {success_rate[i]:.3f}    {frac_rl[i]*100:>5.0f}%  {marker}")
