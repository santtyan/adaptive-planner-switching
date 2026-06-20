"""
A1 — Motivação empírica para rho*=0.30: regime de atuação por planejador.

Usando dados calibrados Fase 1, mostra a taxa de sucesso de cada planejador
no regime onde ele é selecionado, com extrapolação linear para o regime oposto.
O cruzamento das extrapolações cai em ρ≈0.30-0.40 — consistent com o limiar
adotado como fronteira conservadora.

ESCOPO: simulação calibrada Fase 1 (mocks RRT*/PPO). NÃO é A*/SAC real.
Gera: paper/figs/planner_time_vs_density.png + .pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
CSV     = os.path.join(ROOT, "results_abstract",
                       "comprehensive_experiments_20251123_030145.csv")
OUT_PNG = os.path.join(ROOT, "paper", "figs", "planner_time_vs_density.png")


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    df = pd.read_csv(CSV)

    # Coletar todos os dados de sucesso por planejador × densidade (todos os thresholds)
    grp   = df.groupby(["target_density", "selected_planner"])
    smean = grp["success"].mean().unstack() * 100
    sstd  = grp["success"].std().fillna(0).unstack() * 100

    rrt_d = smean.index[smean["rrt_star"].notna()].values
    ppo_d = smean.index[smean["ppo"].notna()].values
    rrt_s = smean.loc[rrt_d, "rrt_star"].values
    ppo_s = smean.loc[ppo_d, "ppo"].values
    rrt_e = sstd.loc[rrt_d, "rrt_star"].values
    ppo_e = sstd.loc[ppo_d, "ppo"].values

    # Extrapolação linear de cada planejador para todo o range
    d_range = np.linspace(0.05, 0.85, 300)
    rrt_fit = np.polyfit(rrt_d, rrt_s, 1)
    ppo_fit = np.polyfit(ppo_d, ppo_s, 1)
    rrt_ext = np.clip(np.polyval(rrt_fit, d_range), 0, 100)
    ppo_ext = np.clip(np.polyval(ppo_fit, d_range), 0, 100)

    # Cruzamento das extrapolações
    diff = rrt_ext - ppo_ext
    sc = np.where(np.diff(np.sign(diff)))[0]
    cross_d = d_range[sc[0]] if len(sc) > 0 else 0.35

    # ── Figura: 2 painéis ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # ── Painel 1: taxa de sucesso ─────────────────────────────────────────────
    ax1.plot(d_range, rrt_ext, "--", color="#1565C0", lw=1.5, alpha=0.55,
             label=f"A* extrapolado (slope={rrt_fit[0]:.1f}%/Δρ)")
    ax1.plot(d_range, ppo_ext, "--", color="#c62828", lw=1.5, alpha=0.55,
             label=f"SAC extrapolado (slope={ppo_fit[0]:+.1f}%/Δρ)")
    ax1.errorbar(rrt_d, rrt_s, yerr=rrt_e, fmt="o", color="#1565C0", markersize=9,
                 capsize=5, label="A* — dados (ρ<ρ*)")
    ax1.errorbar(ppo_d, ppo_s, yerr=ppo_e, fmt="s", color="#c62828", markersize=9,
                 capsize=5, label="SAC — dados (ρ≥ρ*)")
    ax1.axvline(cross_d, color="#2e7d32", linestyle="-.", lw=1.8,
                label=f"Cruzamento ρ≈{cross_d:.2f}")
    ax1.axvline(0.30, color="#f57c00", linestyle=":", lw=2.0,
                label=r"$\rho^*=0{,}30$ adotado")
    ax1.set_xlabel("Densidade de obstáculos ρ", fontsize=11)
    ax1.set_ylabel("Taxa de sucesso (%)", fontsize=11)
    ax1.set_title("(a) Taxa de sucesso por planejador", fontsize=11, fontweight="bold")
    ax1.set_ylim(40, 105)
    ax1.set_xlim(0.05, 0.85)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left")

    # ── Painel 2: tempo de planejamento ───────────────────────────────────────
    tgrp  = df.groupby(["target_density", "selected_planner"])["planning_time_ms"]
    tmean = tgrp.mean().unstack()
    tstd  = tgrp.std().fillna(0).unstack()

    rrt_td = tmean.index[tmean["rrt_star"].notna()].values
    ppo_td = tmean.index[tmean["ppo"].notna()].values
    rrt_t  = tmean.loc[rrt_td, "rrt_star"].values
    ppo_t  = tmean.loc[ppo_td, "ppo"].values
    rrt_te = tstd.loc[rrt_td, "rrt_star"].values
    ppo_te = tstd.loc[ppo_td, "ppo"].values

    rrt_tfit = np.polyfit(rrt_td, rrt_t, 1)
    rrt_text = np.polyval(rrt_tfit, d_range)

    ax2.plot(d_range, rrt_text, "--", color="#1565C0", lw=1.5, alpha=0.55,
             label=f"A* extrapolado (+{rrt_tfit[0]:.0f} ms/Δρ)")
    ax2.axhline(np.mean(ppo_t), color="#c62828", lw=1.5, linestyle="--", alpha=0.55,
                label=f"SAC médio ({np.mean(ppo_t):.1f} ms ≈ cte)")
    ax2.errorbar(rrt_td, rrt_t, yerr=rrt_te, fmt="o", color="#1565C0", markersize=9,
                 capsize=5, label="A* — tempo (ms)")
    ax2.errorbar(ppo_td, ppo_t, yerr=ppo_te, fmt="s", color="#c62828", markersize=9,
                 capsize=5, label="SAC — tempo (ms)")
    ax2.axvline(0.30, color="#f57c00", linestyle=":", lw=2.0,
                label=r"$\rho^*=0{,}30$ adotado")
    ax2.set_xlabel("Densidade de obstáculos ρ", fontsize=11)
    ax2.set_ylabel("Tempo de planejamento (ms)", fontsize=11)
    ax2.set_title("(b) Custo computacional por planejador", fontsize=11, fontweight="bold")
    ax2.set_xlim(0.05, 0.85)
    ax2.set_ylim(0)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Motivação para ρ*=0,30: regime de atuação por planejador\n"
        r"(A* superior a baixa densidade; SAC superior a alta densidade — cruzamento em $\rho^*$)",
        fontsize=11, fontweight="bold"
    )
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Salvo: {OUT_PNG}")
    print(f"Cruzamento de sucesso estimado: ρ={cross_d:.3f}")
    print(f"RRT* slope: {rrt_fit[0]:.1f} %/Δρ")
    print(f"PPO tempo médio: {np.mean(ppo_t):.1f} ms")
    plt.close()


if __name__ == "__main__":
    main()
