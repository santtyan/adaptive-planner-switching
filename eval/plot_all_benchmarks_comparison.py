"""
plot_all_benchmarks_comparison.py — Tabela + gráfico comparando TODOS os
benchmarks reais/calibrados disponíveis no projeto (09/07/2026).

Gera:
    paper/figs/all_benchmarks_table.csv   — tabela consolidada
    paper/figs/all_benchmarks_comparison.png/.pdf — gráfico comparativo
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

rows = []

# 1. Classical algorithms — REAL benchmark (time/memory, 100 nodes)
# Filtra reachable=True: trials sem caminho encontrado terminam cedo e usam
# menos memória, distorcendo a média para baixo (causa histórica da
# divergência 6,6 vs 7,7 KB do A* — ver [[project-numeros-canonicos]]).
df = pd.read_csv(os.path.join(RES, "classical_benchmark.csv"))
if "reachable" in df.columns:
    df = df[df.reachable.astype(str).isin(["True", "1"])]
d100 = df[df.n_nodes == 100]
for algo in ["dijkstra", "astar", "floyd_warshall", "johnson"]:
    sub = d100[d100.algorithm == algo]
    rows.append({
        "benchmark": "Clássicos (100 nós)", "método": algo, "paradigma": "Clássico (busca)",
        "métrica_principal": "tempo_ms", "valor": sub.time_ms.mean(),
        "métrica_secundária": "memória_kb", "valor_secundário": sub.peak_memory_kb.mean(),
        "n_trials": len(sub), "tipo": "real",
    })

# 2. Fase 1 Monte Carlo — calibrated models, adaptive vs baselines
adaptive_success = 85.3
fixed_best = 76.0
switch_best = 78.7
regret = 2.9
for name, val in [("ρ-criterion (adaptivo)", adaptive_success),
                   ("melhor fixo", fixed_best),
                   ("melhor switch concorrente", switch_best)]:
    rows.append({
        "benchmark": "Fase 1 Monte Carlo (1.500 trials)", "método": name,
        "paradigma": "ρ-criterion (fusão clássico+RL, calibrado)" if "ρ" in name else "Clássico ou RL fixo (calibrado)",
        "métrica_principal": "success_rate_%", "valor": val,
        "métrica_secundária": "regret_%", "valor_secundário": regret if "ρ" in name else np.nan,
        "n_trials": 1500, "tipo": "calibrado",
    })

# 3. MARL 2D — real CBS trials, 3 densities
for dens in ["sparse", "dense", "very_dense"]:
    fp = os.path.join(RES, f"multiagent_2d_results_{dens}.csv")
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        succ_col = "success" if "success" in d.columns else d.columns[0]
        try:
            sr = d[succ_col].mean() * 100 if d[succ_col].max() <= 1 else d[succ_col].mean()
        except Exception:
            sr = np.nan
        rows.append({
            "benchmark": "MARL 2D (CBS real)", "método": f"world={dens}",
            "paradigma": "Clássico multiagente (CBS) — MARL não treinado",
            "métrica_principal": "n_trials", "valor": len(d),
            "métrica_secundária": "-", "valor_secundário": np.nan,
            "n_trials": len(d), "tipo": "real",
        })

# 4. 2D SAC/CrossQ/BC comparison (esta madrugada, 09/07)
twod_results = [
    ("SAC (R_SURVIVAL=0.1)", 90, 14000, "steps_to_90pct", "RL (SAC)"),
    ("SAC (R_SURVIVAL=0.0, config Gazebo)", 90, 6000, "steps_to_90pct", "RL (SAC)"),
    ("CrossQ", 5, 3000, "success_at_timeout", "RL (CrossQ)"),
    ("BC (imitação supervisionada)", 98, None, "final_success_pct_2min", "Supervisionado (BC)"),
]
for name, sr, steps, note, paradigma in twod_results:
    rows.append({
        "benchmark": "2D twin — SAC×CrossQ×BC (09/07)", "método": name,
        "métrica_principal": "success_%", "valor": sr,
        "métrica_secundária": "steps", "valor_secundário": steps,
        "n_trials": 50, "tipo": "real_2D", "paradigma": paradigma,
    })

table = pd.DataFrame(rows)
out_csv = os.path.join(FIGS, "all_benchmarks_table.csv")
table.to_csv(out_csv, index=False)
print(f"Tabela salva: {out_csv}")
print(table.to_string(index=False))

# ── Gráfico ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) Clássicos: tempo (log)
ax = axes[0, 0]
sub = table[table.benchmark == "Clássicos (100 nós)"]
ax.bar(sub["método"], sub["valor"], color="#4C72B0")
ax.set_yscale("log")
ax.set_ylabel("Tempo (ms, log)")
ax.set_title("(a) Algoritmos clássicos — 100 nós (real)")
ax.tick_params(axis="x", rotation=25)

# (b) Fase 1 Monte Carlo
ax = axes[0, 1]
sub = table[table.benchmark == "Fase 1 Monte Carlo (1.500 trials)"]
colors = ["#55A868" if "ρ" in m else "#C44E52" for m in sub["método"]]
ax.bar(sub["método"], sub["valor"], color=colors)
ax.set_ylabel("Taxa de sucesso (%)")
ax.set_ylim(0, 100)
ax.set_title("(b) Fase 1 — ρ-criterion vs. baselines (calibrado)")
ax.tick_params(axis="x", rotation=20)
for i, v in enumerate(sub["valor"]):
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

# (c) MARL 2D — n_trials por densidade
ax = axes[1, 0]
sub = table[table.benchmark == "MARL 2D (CBS real)"]
ax.bar(sub["método"], sub["valor"], color="#8172B2")
ax.set_ylabel("N trials (real)")
ax.set_title("(c) MARL 2D — trials reais por densidade")

# (d) 2D SAC vs CrossQ vs BC — success rate
ax = axes[1, 1]
sub = table[table.benchmark == "2D twin — SAC×CrossQ×BC (09/07)"]
colors2 = ["#4C72B0", "#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(sub["método"], sub["valor"], color=colors2)
ax.set_ylabel("Taxa de sucesso (%)")
ax.set_ylim(0, 105)
ax.set_title("(d) 2D twin — SAC × CrossQ × BC (09/07/2026)")
ax.tick_params(axis="x", rotation=20)
for i, (v, row) in enumerate(zip(sub["valor"], sub.itertuples())):
    label = f"{v:.0f}%"
    if row.valor_secundário and not pd.isna(row.valor_secundário):
        label += f"\n({int(row.valor_secundário)} steps)" if row.valor_secundário > 100 else f"\n(~2min)"
    ax.text(i, v + 2, label, ha="center", fontsize=8)

fig.suptitle("Comparação consolidada de todos os benchmarks — adaptive-planner-switching",
             fontsize=13, fontweight="bold")
fig.tight_layout()

png_path = os.path.join(FIGS, "all_benchmarks_comparison.png")
pdf_path = os.path.join(FIGS, "all_benchmarks_comparison.pdf")
fig.savefig(png_path, dpi=200)
fig.savefig(pdf_path)
print(f"Gráfico salvo: {png_path} / {pdf_path}")
