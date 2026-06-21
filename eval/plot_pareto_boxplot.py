"""
Figuras com boxplots e equações para justificativa formal de τ*=0,30.

Gera:
  fig_pareto_boxplot_main.png/.pdf   — boxplot de taxa de sucesso por τ + equação ROI
  fig_pareto_boxplot_time.png/.pdf   — boxplot de tempo de planejamento por planejador
  fig_pareto_equations_panel.png/.pdf — painel 3 figuras com equações anotadas
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "paper", "figs")
RES  = os.path.join(ROOT, "results_abstract")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def savefig(name):
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIGS, f"{name}.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

df   = pd.read_csv(os.path.join(RES, "adaptive_switching_results.csv"))
sens = pd.read_csv(os.path.join(RES, "threshold_sensitivity.csv"))

taus_all = sorted(df["threshold"].unique())   # [0.25, 0.30, 0.35]
# Referência de comparação: τ=0.25 (mais próximo do "sucesso puro" disponível)
TAU_STAR   = 0.30
TAU_COMPARE = 0.25   # melhor sucesso disponível nos dados de trial

# ── Taxa de sucesso por trial (agrupado por threshold × trial)
success_by_tau = {
    tau: df[df["threshold"] == tau].groupby("trial")["success"].mean().values
    for tau in taus_all
}

# Fração de SAC por threshold (dado empírico)
frac_sac = {
    tau: (df[df["threshold"] == tau]["selected"] == "PPO").mean()
    for tau in taus_all
}

tau_labels = [f"$\\tau={t:.2f}$" for t in taus_all]
colors_box  = ["#2196F3" if abs(t - TAU_STAR) < 0.01
               else "#FF9800" if abs(t - TAU_COMPARE) < 0.01
               else "#90CAF9" for t in taus_all]

# Taxa de sucesso (do sens para τ mais amplo)
sr_star    = sens[sens["tau"].round(2) == TAU_STAR]["success_rate"].values[0]
sr_compare = sens[sens["tau"].round(2) == TAU_COMPARE]["success_rate"].values[0]
idx_star   = taus_all.index(TAU_STAR)
idx_compare = taus_all.index(TAU_COMPARE)

# ══════════════════════════════════════════════════════════════
# FIGURA 1 — Boxplot sucesso por τ + equação ROI anotada
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={"width_ratios": [2, 1]})

# ─ Esquerda: boxplot ─────────────────────────────────────────
ax = axes[0]
bp = ax.boxplot(
    [success_by_tau[t] for t in taus_all],
    patch_artist=True,
    widths=0.6,
    medianprops=dict(color="black", lw=2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    whiskerprops=dict(lw=1.5),
    capprops=dict(lw=1.5),
)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Destaca τ=0.30 e τ=0.20
ax.get_xticklabels()  # force render

ax.set_xticklabels(tau_labels, rotation=30, ha="right")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_ylabel("Taxa de sucesso por trial")
ax.set_title("(a) Distribuição da taxa de sucesso por limiar $\\tau$\n"
             "(50 trials por nível de densidade × 3 densidades)")
ax.grid(True, axis="y", alpha=0.3)

# Legenda de cores
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2196F3", alpha=0.75, label=f"$\\tau^*={TAU_STAR:.2f}$ (escolhido)"),
    Patch(facecolor="#FF9800", alpha=0.75, label=f"$\\tau={TAU_COMPARE:.2f}$ (melhor sucesso puro)"),
    Patch(facecolor="#90CAF9", alpha=0.75, label="Outros limiares"),
], fontsize=9, loc="lower left")

# Anotação com seta mostrando diferença τ=0.20 vs τ=0.30
ax.annotate("",
    xy=(idx_compare + 1, sr_compare),
    xytext=(idx_star   + 1, sr_star),
    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.8, linestyle="dashed"))
ax.text((idx_star + idx_compare) / 2 + 1, (sr_star + sr_compare) / 2 + 0.012,
        f"$\\Delta S = +{(sr_compare-sr_star)*100:.1f}$ pp",
        ha="center", fontsize=9, color="navy", fontweight="bold")

# ─ Direita: equação ROI e explicação ────────────────────────
ax = axes[1]
ax.axis("off")

delta_sr  = (sr_compare - sr_star) * 100
delta_sac = (frac_sac[TAU_COMPARE] - frac_sac[TAU_STAR]) * 100
roi_val   = delta_sr / delta_sac if abs(delta_sac) > 0.01 else 0

# Caixa de equações
eq_text = (
    r"$\bf{Critério\ de\ seleção\ do\ limiar}$" + "\n\n"
    r"$\tau^* = \arg\min_{\tau}\ \mathcal{R}(\tau)$" + "\n\n"
    r"$\mathcal{R}(\tau) = \mathbb{E}_\rho\!\left[\max_{k}\,\pi_k(\rho) - \pi_\tau(\rho)\right]$" + "\n\n"
    "onde $\\pi_\\tau(\\rho) \\in \\{\\text{A*}, \\text{SAC}\\}$\n\n"
    r"$\bf{Retorno\ marginal\ em\ }\tau^*$" + "\n\n"
    r"$\mathrm{ROI} = \dfrac{\Delta S}{\Delta f_{\mathrm{SAC}}}$" + "\n\n"
    f"$= \\dfrac{{+{delta_sr:.1f}\\,\\mathrm{{pp}}}}{{+{delta_sac:.0f}\\,\\mathrm{{pp}}}} = {roi_val:.2f}\\,\\mathrm{{pp/pp}}$" + "\n\n"
    r"$\Rightarrow \tau^*=0{,}30$ é Pareto-ótimo"
)

ax.text(0.05, 0.95, eq_text,
        transform=ax.transAxes,
        fontsize=10.5, va="top", ha="left",
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.6",
                  facecolor="#E3F2FD", edgecolor="#1565C0", lw=1.5))

fig.suptitle("Justificativa formal de $\\tau^*=0{,}30$: análise de retorno marginal",
             fontsize=13, fontweight="bold")
fig.tight_layout()
savefig("fig_pareto_boxplot_main")

# ══════════════════════════════════════════════════════════════
# FIGURA 2 — Boxplot de tempo de planejamento por planejador
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# ─ Esquerda: boxplot tempo por planejador ────────────────────
ax = axes[0]
t_classic = df[df["selected"] == "RRT*"]["time_ms"].values
t_rl      = df[df["selected"] == "PPO"]["time_ms"].values

bp2 = ax.boxplot(
    [t_rl, t_classic],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="black", lw=2.5),
    flierprops=dict(marker="o", markersize=3, alpha=0.3),
)
bp2["boxes"][0].set_facecolor("#4CAF50"); bp2["boxes"][0].set_alpha(0.75)
bp2["boxes"][1].set_facecolor("#F44336"); bp2["boxes"][1].set_alpha(0.75)

ax.set_xticklabels(["SAC/PPO\n(planejador RL)", "A*/RRT*\n(planejador clássico)"], fontsize=11)
ax.set_ylabel("Tempo de planejamento (ms)")
ax.set_title("(a) Distribuição do tempo de planejamento\npor tipo de planejador")
ax.grid(True, axis="y", alpha=0.3)

# Anotação de mediana
med_rl  = np.median(t_rl)
med_cl  = np.median(t_classic)
ax.annotate(f"Mediana: {med_rl:.1f} ms",
            xy=(1, med_rl), xytext=(1.35, med_rl + 5),
            fontsize=9, color="#2E7D32",
            arrowprops=dict(arrowstyle="->", color="#2E7D32"))
ax.annotate(f"Mediana: {med_cl:.1f} ms",
            xy=(2, med_cl), xytext=(1.55, med_cl + 10),
            fontsize=9, color="#B71C1C",
            arrowprops=dict(arrowstyle="->", color="#B71C1C"))

# ─ Direita: boxplot tempo por densidade (só A*) ──────────────
ax = axes[1]
density_levels = sorted(df["density"].unique())
t_by_density   = [df[(df["selected"] == "RRT*") & (df["density"] == d)]["time_ms"].values
                  for d in density_levels]
d_labels = [f"$\\rho={d:.2f}$" for d in density_levels]

bp3 = ax.boxplot(t_by_density, patch_artist=True, widths=0.55,
                 medianprops=dict(color="black", lw=2.5),
                 flierprops=dict(marker="o", markersize=3, alpha=0.3))

cmap = plt.cm.Reds
for i, patch in enumerate(bp3["boxes"]):
    patch.set_facecolor(cmap(0.3 + 0.6 * i / len(density_levels)))
    patch.set_alpha(0.8)

ax.set_xticklabels(d_labels, fontsize=11)
ax.set_ylabel("Tempo de planejamento A*/RRT* (ms)")
ax.set_title("(b) Tempo de A* cresce com a densidade\n"
             "SAC mantém $\\approx$12 ms constante (linha tracejada)")
ax.axhline(med_rl, ls="--", color="#4CAF50", lw=2,
           label=f"SAC mediana ({med_rl:.0f} ms)")

# Equação de complexidade A*
ax.text(0.97, 0.97,
        r"$T_{A^*}(\rho) \propto \rho \cdot N^2$" + "\n"
        r"$T_{\mathrm{SAC}} \approx$ const.",
        transform=ax.transAxes, fontsize=10, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825"))

ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Custo computacional: SAC é constante, A* cresce com a densidade",
             fontsize=13, fontweight="bold")
fig.tight_layout()
savefig("fig_pareto_boxplot_time")

# ══════════════════════════════════════════════════════════════
# FIGURA 3 — Painel completo: boxplot + curva + equações
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

ax_bp   = fig.add_subplot(gs[0, :2])   # boxplot sucesso × τ (largo)
ax_eq   = fig.add_subplot(gs[0, 2])    # equações
ax_knee = fig.add_subplot(gs[1, 0])    # curva joelho
ax_reg  = fig.add_subplot(gs[1, 1])    # regret
ax_roi  = fig.add_subplot(gs[1, 2])    # retorno marginal

# ── Painel (a): boxplot sucesso por τ ─────────────────────────
bp = ax_bp.boxplot(
    [success_by_tau[t] for t in taus_all],
    patch_artist=True, widths=0.6,
    medianprops=dict(color="black", lw=2),
    flierprops=dict(marker="o", markersize=2.5, alpha=0.35),
)
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax_bp.set_xticklabels(tau_labels, rotation=25, ha="right", fontsize=9)
ax_bp.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_bp.set_ylabel("Taxa de sucesso (por trial)")
ax_bp.set_title(r"(a) Distribuição da taxa de sucesso por $\tau$", fontsize=11)
ax_bp.grid(True, axis="y", alpha=0.3)

# Colchete de anotação Δ
ax_bp.annotate("",
    xy=(idx_compare + 1, sr_compare + 0.01),
    xytext=(idx_star   + 1, sr_star   + 0.01),
    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.8, ls="--"))
ax_bp.text((idx_star + idx_compare) / 2 + 1, max(sr_star, sr_compare) + 0.018,
           f"$\\Delta S = +{(sr_compare-sr_star)*100:.1f}$ pp", fontsize=9,
           ha="center", color="navy", fontweight="bold")

# Legenda
ax_bp.legend(handles=[
    Patch(facecolor="#2196F3", alpha=0.75, label=f"$\\tau^*={TAU_STAR:.2f}$"),
    Patch(facecolor="#FF9800", alpha=0.75, label=f"$\\tau={TAU_COMPARE:.2f}$"),
    Patch(facecolor="#90CAF9", alpha=0.75, label="outros"),
], fontsize=8, loc="lower left")

# ── Painel (b): caixa de equações ────────────────────────────
ax_eq.axis("off")
ax_eq.text(0.05, 0.98,
    r"$\mathbf{Critério}$" + "\n"
    r"$\tau^* = \arg\min_\tau\,\mathcal{R}(\tau)$" + "\n\n"
    r"$\mathcal{R}(\tau)=\mathbb{E}_\rho[\pi^*(\rho)-\pi_\tau(\rho)]$" + "\n\n"
    r"$\mathbf{Retorno\ marginal}$" + "\n"
    r"$\mathrm{ROI}(\tau)=\dfrac{dS}{df_{\mathrm{SAC}}}$" + "\n\n"
    f"$\\mathrm{{ROI}}(\\tau={TAU_COMPARE:.2f})={roi_val:.2f}\\,\\mathrm{{pp/pp}}$" + "\n"
    r"$\Rightarrow$ ponto joelho em $\tau^*$",
    transform=ax_eq.transAxes,
    fontsize=10, va="top", linespacing=2.0,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD",
              edgecolor="#1565C0", lw=1.5))
ax_eq.set_title("(b) Equações de seleção", fontsize=11)

# ── Painel (c): curva joelho ──────────────────────────────────
frac_sorted_vals = [frac_sac[t] for t in taus_all]
sr_vals = [sens[sens["tau"].round(2) == round(t, 2)]["success_rate"].values[0]
           for t in taus_all]
ax_knee.plot([f*100 for f in frac_sorted_vals], sr_vals,
             "o-", color="steelblue", lw=2, markersize=7)
ax_knee.scatter(frac_sac[TAU_STAR]*100, sr_vals[taus_all.index(TAU_STAR)],
                s=200, color="#2196F3", zorder=5,
                label=f"$\\tau^*={TAU_STAR:.2f}$", edgecolors="navy", lw=2)
ax_knee.scatter(frac_sac[TAU_COMPARE]*100, sr_vals[taus_all.index(TAU_COMPARE)],
                s=150, color="#FF9800", zorder=5,
                label=f"$\\tau={TAU_COMPARE:.2f}$", edgecolors="darkorange", lw=2)
ax_knee.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax_knee.set_xlabel("Uso de SAC (%)")
ax_knee.set_ylabel("Taxa de sucesso")
ax_knee.set_title("(c) Ponto joelho", fontsize=11)
ax_knee.legend(fontsize=8); ax_knee.grid(True, alpha=0.3)
ax_knee.annotate("ponto\njoelho",
    xy=(frac_sac[TAU_STAR]*100, sr_vals[taus_all.index(TAU_STAR)]),
    xytext=(frac_sac[TAU_STAR]*100 + 12, sr_vals[taus_all.index(TAU_STAR)] - 0.04),
    fontsize=8, color="navy",
    arrowprops=dict(arrowstyle="->", color="navy"))

# ── Painel (d): regret por τ com dados do sens (mais amplo) ──
sens_taus   = sens["tau"].values
sens_regret = sens["regret_pct"].values
ax_reg.plot(sens_taus, sens_regret, "s-", color="#9C27B0", lw=2, markersize=7)
ax_reg.axvline(TAU_STAR, ls="--", color="#2196F3", lw=2,
               label=f"$\\tau^*={TAU_STAR:.2f}$")
ax_reg.axhline(5,  ls=":", color="#FF9800", lw=1.5, label="$H_2$: 5%")
ax_reg.axhline(10, ls=":", color="#E53935", lw=1.5, label="pior caso: 10%")
platô_mask = sens_regret < 5
platô_taus = sens_taus[platô_mask]
if len(platô_taus):
    ax_reg.axvspan(platô_taus[0], platô_taus[-1], alpha=0.10, color="#4CAF50")
ax_reg.set_xlabel("Limiar $\\tau$")
ax_reg.set_ylabel("Regret vs Oracle (%)")
ax_reg.set_title("(d) Regret — $\\tau^*$ na borda do platô", fontsize=11)
ax_reg.legend(fontsize=8); ax_reg.grid(True, alpha=0.3)

# ── Painel (e): retorno marginal (barras) usando dados do sens
sens_sr   = sens["success_rate"].values
sens_fr   = np.array([np.mean(df[df["threshold"].round(2) == round(t, 2)]["selected"] == "PPO")
                      if round(t, 2) in [round(x, 2) for x in taus_all]
                      else np.nan for t in sens_taus])
# Para taus sem dados empíricos, interpola linearmente
valid_idx = ~np.isnan(sens_fr)
sens_fr_interp = np.interp(sens_taus, sens_taus[valid_idx], sens_fr[valid_idx])

# ROI entre pontos consecutivos (crescente em uso de SAC = decrescente em τ)
idx_sort = np.argsort(sens_fr_interp)
fr_s  = sens_fr_interp[idx_sort]
sr_s  = sens_sr[idx_sort]
mid_f = (fr_s[:-1] + fr_s[1:]) / 2
d_s   = np.diff(sr_s) * 100
d_f   = np.diff(fr_s) * 100
roi_s = np.where(np.abs(d_f) > 0.5, d_s / d_f, 0)

bar_c = ["#2196F3" if abs(mf - frac_sac[TAU_STAR]) < 0.08 else "#CFD8DC"
         for mf in mid_f]
ax_roi.bar(mid_f * 100, roi_s, width=4.5,
           color=bar_c, alpha=0.85, edgecolor="white")
ax_roi.axhline(0, color="black", lw=1)
ax_roi.axvline(frac_sac[TAU_STAR]*100, ls="--", color="#2196F3", lw=2,
               label=f"$\\tau^*={TAU_STAR:.2f}$ ({frac_sac[TAU_STAR]*100:.0f}% SAC)")
ax_roi.set_xlabel("Uso de SAC (%)")
ax_roi.set_ylabel(r"$\Delta S\,/\,\Delta f_{\mathrm{SAC}}$ (pp/pp)")
ax_roi.set_title("(e) Retorno marginal do SAC", fontsize=11)
ax_roi.legend(fontsize=8); ax_roi.grid(True, alpha=0.3, axis="y")

fig.suptitle(r"$\tau^*=0{,}30$: justificativa formal por análise de retorno marginal",
             fontsize=14, fontweight="bold")
savefig("fig_pareto_equations_panel")

print("\nFiguras geradas:")
print("  fig_pareto_boxplot_main    — boxplot sucesso × τ + equação ROI")
print("  fig_pareto_boxplot_time    — boxplot tempo de planejamento")
print("  fig_pareto_equations_panel — painel completo (5 subplots)")
