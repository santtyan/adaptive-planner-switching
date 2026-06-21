"""
Geração máxima de figuras científicas com os dados disponíveis.

Gera todas as figuras possíveis sem depender do SAC convergir.
Usa: sota_comparison_results.csv, adaptive_switching_results.csv,
     classical_benchmark.csv, benchmark_multiseed.csv,
     cbs_scalability (recalculado inline se necessário)

Figuras geradas:
  fig_success_rate_comparison      — adaptativo vs todos os métodos por densidade
  fig_statistical_test             — já gerada por strengthen_ic.py
  fig_benchmark_ci                 — já gerada por strengthen_ic.py
  fig_regret_analysis              — regret do adaptativo vs oracle por densidade
  fig_threshold_sensitivity        — sensibilidade de τ (0.25/0.30/0.35) por densidade
  fig_planning_time_comparison     — tempo de planejamento por método
  fig_composite_heatmap            — heatmap sucesso × tempo por método × densidade
  fig_sota_radar                   — radar chart métodos vs dimensões
  fig_adaptive_advantage_map       — onde o adaptativo supera cada baseline
  fig_bootstrap_ci_global          — IC 95% global para cada método
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "paper", "figs")
RES  = os.path.join(ROOT, "results_abstract")

METHOD_LABELS = {
    "adaptive_ours":    "Adaptativo (ρ-criterion)",
    "fixed_ppo":        "PPO fixo",
    "fixed_rrt":        "RRT* fixo",
    "neural_switching": "Neural switching",
    "he_multiopt":      "He et al. (2025)",
    "hybrid_drl":       "Hybrid DRL",
}
METHOD_COLORS = {
    "adaptive_ours":    "#2196F3",
    "fixed_ppo":        "#E91E63",
    "fixed_rrt":        "#FF9800",
    "neural_switching": "#9C27B0",
    "he_multiopt":      "#607D8B",
    "hybrid_drl":       "#795548",
}

def savefig(name):
    png = os.path.join(FIGS, name + ".png")
    pdf = os.path.join(FIGS, name + ".pdf")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

def bootstrap_ci_rate(rate, n, n_boot=2000, ci=95):
    samples = np.random.binomial(n, rate, n_boot) / n
    lo = (100 - ci) / 2
    return np.percentile(samples, [lo, 100-lo])

# ── Carrega dados ──────────────────────────────────────────────
sota = pd.read_csv(os.path.join(RES, "sota_comparison_results.csv"))
adapt_thresh = pd.read_csv(os.path.join(RES, "adaptive_switching_results.csv"))
bench_orig = pd.read_csv(os.path.join(RES, "classical_benchmark.csv"))
bench_multi_path = os.path.join(RES, "benchmark_multiseed.csv")
bench_multi = pd.read_csv(bench_multi_path) if os.path.exists(bench_multi_path) else None

print("Gerando figuras científicas...\n")

# ══════════════════════════════════════════════════════════════
# 1. Taxa de sucesso — todos os métodos por densidade
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for method in ["adaptive_ours","fixed_ppo","fixed_rrt","neural_switching","he_multiopt","hybrid_drl"]:
    sub = sota[sota["method"]==method].sort_values("density")
    if sub.empty: continue
    dens = sub["density"].values
    rates = sub["success_rate"].values
    ns = sub["trials"].values
    lo = [bootstrap_ci_rate(r, n)[0] for r, n in zip(rates, ns)]
    hi = [bootstrap_ci_rate(r, n)[1] for r, n in zip(rates, ns)]
    lw = 3 if method == "adaptive_ours" else 1.5
    zorder = 5 if method == "adaptive_ours" else 2
    ax.plot(dens, rates, "o-", label=METHOD_LABELS[method],
            color=METHOD_COLORS[method], lw=lw, zorder=zorder)
    ax.fill_between(dens, lo, hi, alpha=0.12, color=METHOD_COLORS[method])

ax.axvline(0.30, ls="--", color="gray", lw=1.2, alpha=0.7, label=r"$\rho^*=0{,}30$")
ax.set_xlabel("Densidade de obstáculos ρ", fontsize=12)
ax.set_ylabel("Taxa de sucesso", fontsize=12)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="lower left")
ax.set_title("Taxa de sucesso por método e densidade — IC 95% (bootstrap binomial)", fontsize=12)
ax.grid(True, alpha=0.3)
savefig("fig_success_rate_comparison")

# ══════════════════════════════════════════════════════════════
# 2. Regret do adaptativo vs oracle por densidade
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

adapt = sota[sota["method"]=="adaptive_ours"].sort_values("density")
methods_for_oracle = ["adaptive_ours","fixed_ppo","fixed_rrt","neural_switching","he_multiopt","hybrid_drl"]
dens_vals = sorted(sota["density"].unique())

oracle_rates = []
adapt_rates = []
for d in dens_vals:
    sub = sota[sota["density"]==d]
    oracle_rates.append(sub["success_rate"].max())
    a = sub[sub["method"]=="adaptive_ours"]["success_rate"].values
    adapt_rates.append(a[0] if len(a) > 0 else np.nan)

oracle_rates = np.array(oracle_rates)
adapt_rates = np.array(adapt_rates)
regret = (oracle_rates - adapt_rates) / oracle_rates * 100  # regret %

ax1.bar(dens_vals, regret, color=["#E53935" if r > 5 else "#43A047" for r in regret],
        width=0.04, alpha=0.8, edgecolor="white")
ax1.axhline(5, ls="--", color="#E53935", lw=1.5, label="Limite H2: 5%")
ax1.axhline(adapt_rates.__class__.__name__ and np.mean(regret), ls=":", color="navy",
            lw=1.5, label=f"Média: {np.mean(regret):.1f}%")
ax1.axhline(np.mean(regret), ls=":", color="navy", lw=1.5)
ax1.set_xlabel("Densidade de obstáculos ρ")
ax1.set_ylabel("Regret (%)")
ax1.set_title("Regret do ρ-criterion vs Oracle ideal")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, max(regret)*1.3 + 2)

# Curvas oracle vs adaptativo
ax2.plot(dens_vals, oracle_rates, "k--", lw=2, label="Oracle ideal", zorder=5)
ax2.plot(dens_vals, adapt_rates, "o-", color="#2196F3", lw=2.5,
         label=f"Adaptativo (regret médio={np.mean(regret):.1f}%)", zorder=5)
ax2.fill_between(dens_vals, adapt_rates, oracle_rates, alpha=0.15, color="#E53935",
                 label="Área de regret")
ax2.set_xlabel("Densidade de obstáculos ρ")
ax2.set_ylabel("Taxa de sucesso")
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
ax2.set_title("ρ-criterion vs Oracle — distância ao desempenho máximo")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
savefig("fig_regret_analysis")

# ══════════════════════════════════════════════════════════════
# 3. Sensibilidade do limiar τ por densidade
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

thresh_colors = {0.25: "#FF9800", 0.30: "#2196F3", 0.35: "#4CAF50"}
for t in [0.25, 0.30, 0.35]:
    sub = adapt_thresh[adapt_thresh["threshold"]==t].groupby("density")["success"].mean()
    dens = sub.index.values
    rates = sub.values
    lw = 3 if t == 0.30 else 1.5
    alpha = 1.0 if t == 0.30 else 0.7
    label = f"τ = {t:.2f}" + (" (escolhido)" if t == 0.30 else "")
    ax1.plot(dens, rates, "o-", label=label, color=thresh_colors[t], lw=lw, alpha=alpha)

ax1.axvline(0.30, ls="--", color="gray", lw=1, alpha=0.6)
ax1.set_xlabel("Densidade de obstáculos ρ")
ax1.set_ylabel("Taxa de sucesso")
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax1.legend()
ax1.set_title("Sensibilidade ao limiar τ por densidade")
ax1.grid(True, alpha=0.3)

# Barras com taxa global por threshold
global_rates = [adapt_thresh[adapt_thresh["threshold"]==t]["success"].mean()
                for t in [0.25, 0.30, 0.35]]
bars = ax2.bar([f"τ={t:.2f}" for t in [0.25, 0.30, 0.35]], global_rates,
               color=list(thresh_colors.values()), alpha=0.8, edgecolor="white", width=0.5)
for bar, rate in zip(bars, global_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{rate:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylabel("Taxa de sucesso global")
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax2.set_ylim(0.75, 0.88)
ax2.set_title("Taxa de sucesso global por limiar τ")
ax2.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("fig_threshold_sensitivity")

# ══════════════════════════════════════════════════════════════
# 4. Tempo de planejamento por método e densidade
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for method in ["adaptive_ours","fixed_ppo","fixed_rrt","neural_switching"]:
    sub = sota[sota["method"]==method].sort_values("density")
    if sub.empty or "avg_time" not in sub.columns: continue
    lw = 3 if method == "adaptive_ours" else 1.5
    ax.plot(sub["density"], sub["avg_time"], "o-",
            label=METHOD_LABELS[method], color=METHOD_COLORS[method], lw=lw)

ax.axvline(0.30, ls="--", color="gray", lw=1, alpha=0.7, label=r"$\rho^*=0{,}30$")
ax.set_xlabel("Densidade de obstáculos ρ")
ax.set_ylabel("Tempo médio de planejamento (ms)")
ax.set_title("Tempo de planejamento por método e densidade")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
savefig("fig_planning_time_comparison")

# ══════════════════════════════════════════════════════════════
# 5. Heatmap sucesso × método × densidade
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
methods_order = ["adaptive_ours","neural_switching","hybrid_drl","he_multiopt","fixed_ppo","fixed_rrt"]
densities_order = sorted(sota["density"].unique())

matrix = np.zeros((len(methods_order), len(densities_order)))
for i, m in enumerate(methods_order):
    for j, d in enumerate(densities_order):
        val = sota[(sota["method"]==m) & (sota["density"]==d)]["success_rate"].values
        matrix[i, j] = val[0] if len(val) > 0 else np.nan

im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(densities_order)))
ax.set_xticklabels([f"{d:.2f}" for d in densities_order])
ax.set_yticks(range(len(methods_order)))
ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_order])
ax.set_xlabel("Densidade de obstáculos ρ")
ax.set_title("Heatmap: taxa de sucesso por método e densidade")

for i in range(len(methods_order)):
    for j in range(len(densities_order)):
        v = matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="black" if 0.3 < v < 0.8 else "white", fontsize=9)

plt.colorbar(im, ax=ax, label="Taxa de sucesso")
fig.tight_layout()
savefig("fig_composite_heatmap")

# ══════════════════════════════════════════════════════════════
# 6. IC 95% global — barras com intervalo de confiança
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

methods_show = ["adaptive_ours","neural_switching","hybrid_drl","he_multiopt","fixed_ppo","fixed_rrt"]
global_means, ci_lo, ci_hi, labels, colors_list = [], [], [], [], []

for m in methods_show:
    sub = sota[sota["method"]==m]
    if sub.empty: continue
    total_trials = sub["trials"].sum()
    total_success = (sub["success_rate"] * sub["trials"]).sum()
    rate = total_success / total_trials
    ci = bootstrap_ci_rate(rate, int(total_trials))
    global_means.append(rate)
    ci_lo.append(rate - ci[0])
    ci_hi.append(ci[1] - rate)
    labels.append(METHOD_LABELS.get(m, m))
    colors_list.append(METHOD_COLORS.get(m, "gray"))

x = np.arange(len(labels))
bars = ax.bar(x, global_means, color=colors_list, alpha=0.8, edgecolor="white", width=0.6)
ax.errorbar(x, global_means, yerr=[ci_lo, ci_hi], fmt="none",
            color="black", capsize=6, lw=2)

for i, (bar, rate) in enumerate(zip(bars, global_means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci_hi[i] + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Taxa de sucesso global")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.0)
ax.set_title("Taxa de sucesso global por método — IC 95% (bootstrap binomial, 150 trials)")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("fig_bootstrap_ci_global")

# ══════════════════════════════════════════════════════════════
# 7. Vantagem do adaptativo sobre cada baseline por densidade
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

adapt_by_d = sota[sota["method"]=="adaptive_ours"].set_index("density")["success_rate"]
comparisons = {
    "vs PPO fixo":        ("fixed_ppo",        "#E91E63"),
    "vs RRT* fixo":       ("fixed_rrt",         "#FF9800"),
    "vs Neural switching":("neural_switching",  "#9C27B0"),
    "vs He et al.":       ("he_multiopt",       "#607D8B"),
}

for label, (method, color) in comparisons.items():
    sub = sota[sota["method"]==method].set_index("density")["success_rate"]
    common = sorted(set(adapt_by_d.index) & set(sub.index))
    diff = [(adapt_by_d[d] - sub[d]) * 100 for d in common]
    ax.plot(common, diff, "o-", label=label, color=color, lw=1.8)

ax.axhline(0, color="black", lw=1.2, ls="-")
ax.axvline(0.30, ls="--", color="gray", lw=1, alpha=0.7, label=r"$\rho^*=0{,}30$")
ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30,
                alpha=0.04, color="green")
ax.set_xlabel("Densidade de obstáculos ρ")
ax.set_ylabel("Vantagem do adaptativo (pp)")
ax.set_title("Vantagem do ρ-criterion sobre cada baseline por densidade")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
savefig("fig_adaptive_advantage_map")

# ══════════════════════════════════════════════════════════════
# 8. Benchmark clássico — comparação Floyd-Warshall vs A*
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bench_colors = {
    "dijkstra": "#FF9800", "astar": "#2196F3",
    "floyd_warshall": "#E53935", "johnson": "#9C27B0"
}
bench_labels = {
    "dijkstra": "Dijkstra", "astar": "A*",
    "floyd_warshall": "Floyd-Warshall", "johnson": "Johnson"
}

for algo in ["dijkstra","astar","floyd_warshall","johnson"]:
    sub = bench_orig[bench_orig["algorithm"]==algo].groupby("n_nodes").agg(
        t=("time_ms","mean"), m=("peak_memory_kb","mean")).reset_index()
    if sub.empty: continue
    lw = 2.5 if algo in ["astar","floyd_warshall"] else 1.5
    ax1.plot(sub["n_nodes"], sub["t"], "o-", label=bench_labels[algo],
             color=bench_colors[algo], lw=lw)
    ax2.plot(sub["n_nodes"], sub["m"], "s-", label=bench_labels[algo],
             color=bench_colors[algo], lw=lw)

for ax, ylabel, title in [
    (ax1, "Tempo (ms, escala log)", "Tempo de execução — escala log"),
    (ax2, "Memória (KB, escala log)", "Memória de pico — escala log"),
]:
    ax.set_yscale("log")
    ax.set_xlabel("Número de nós")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

fig.suptitle("Benchmark algoritmos clássicos — Floyd-Warshall inviável em tempo real", y=1.01)
fig.tight_layout()
savefig("fig_benchmark_logscale")

# ══════════════════════════════════════════════════════════════
# 9. Distribuição de uso de planejador por densidade (adaptativo)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

adapt30 = adapt_thresh[adapt_thresh["threshold"]==0.30]
rl_frac = adapt30.groupby("density")["selected"].apply(
    lambda x: (x == "PPO").mean())
classic_frac = 1 - rl_frac

dens = rl_frac.index.values
ax.bar(dens, classic_frac.values, width=0.04, label="A*/RRT* (clássico)", color="#FF9800", alpha=0.85)
ax.bar(dens, rl_frac.values, bottom=classic_frac.values, width=0.04,
       label="PPO/SAC (RL)", color="#2196F3", alpha=0.85)
ax.axvline(0.30, ls="--", color="gray", lw=1.5, label=r"$\rho^*=0{,}30$ (limiar)")
ax.set_xlabel("Densidade de obstáculos ρ")
ax.set_ylabel("Fração de decisões")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Distribuição de uso: clássico vs RL por densidade (τ=0,30)")
ax.grid(True, alpha=0.3, axis="y")
savefig("fig_planner_usage_distribution")

# ══════════════════════════════════════════════════════════════
# 10. Summary figure — painel 2×2 para relatório
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# [0,0] — sucesso por densidade (principais métodos)
ax = axes[0, 0]
for method in ["adaptive_ours","fixed_ppo","fixed_rrt","neural_switching"]:
    sub = sota[sota["method"]==method].sort_values("density")
    if sub.empty: continue
    lw = 3 if method == "adaptive_ours" else 1.5
    ax.plot(sub["density"], sub["success_rate"], "o-",
            label=METHOD_LABELS[method], color=METHOD_COLORS[method], lw=lw)
ax.axvline(0.30, ls="--", color="gray", lw=1, alpha=0.7)
ax.set_ylabel("Taxa de sucesso"); ax.set_xlabel("Densidade ρ")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("(a) Sucesso por densidade")

# [0,1] — regret por densidade
ax = axes[0, 1]
ax.bar(dens_vals, regret, color=["#E53935" if r > 5 else "#43A047" for r in regret],
       width=0.04, alpha=0.85)
ax.axhline(5, ls="--", color="#E53935", lw=1.5, label="Limite 5%")
ax.axhline(np.mean(regret), ls=":", color="navy", lw=1.5, label=f"Média {np.mean(regret):.1f}%")
ax.set_ylabel("Regret (%)"); ax.set_xlabel("Densidade ρ")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_title("(b) Regret vs Oracle ideal")

# [1,0] — benchmark tempo log
ax = axes[1, 0]
for algo in ["dijkstra","astar","floyd_warshall","johnson"]:
    sub = bench_orig[bench_orig["algorithm"]==algo].groupby("n_nodes")["time_ms"].mean()
    if sub.empty: continue
    ax.plot(sub.index, sub.values, "o-", label=bench_labels[algo],
            color=bench_colors[algo], lw=2)
ax.set_yscale("log"); ax.set_xlabel("Nós")
ax.set_ylabel("Tempo (ms, log)"); ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")
ax.set_title("(c) Benchmark clássicos (log)")

# [1,1] — distribuição uso planejador
ax = axes[1, 1]
ax.bar(dens, classic_frac.values, width=0.04, label="Clássico", color="#FF9800", alpha=0.85)
ax.bar(dens, rl_frac.values, bottom=classic_frac.values, width=0.04,
       label="RL", color="#2196F3", alpha=0.85)
ax.axvline(0.30, ls="--", color="gray", lw=1.5)
ax.set_xlabel("Densidade ρ"); ax.set_ylabel("Fração")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_title("(d) Uso: clássico vs RL")

fig.suptitle("ρ-criterion: visão geral dos resultados", fontsize=14, fontweight="bold")
fig.tight_layout()
savefig("fig_summary_panel")

print(f"\nTotal: 10 figuras geradas em paper/figs/")
