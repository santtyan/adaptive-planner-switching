"""
F2 do docs/PLANO_CORRECAO.md — Chattering: one-shot (sem histerese) vs.
per-step com histerese.

Fontes:
    results_abstract/h1_oneshot_perstep_switches.csv  (SEM histerese, n=1500)
    results_abstract/h1_hysteresis_2d.csv             (COM histerese, n=1500)

Rodar: python3 eval/env2d/plot_chattering_hysteresis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs")


def main():
    no_hyst = pd.read_csv(os.path.join(RES, "h1_oneshot_perstep_switches.csv"))["switches"].values
    hyst = pd.read_csv(os.path.join(RES, "h1_hysteresis_2d.csv"))["switches"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    max_s = max(no_hyst.max(), hyst.max())
    bins = np.arange(0, max_s + 2) - 0.5
    ax.hist(no_hyst, bins=bins, alpha=0.6, label=f"Sem histerese (n={len(no_hyst)})", color="#C44E52")
    ax.hist(hyst, bins=bins, alpha=0.6, label=f"Com histerese (n={len(hyst)})", color="#55A868")
    ax.set_xlabel("Trocas de planejador por episódio")
    ax.set_ylabel("Nº de episódios")
    ax.set_title("(a) Distribuição de trocas por episódio")
    ax.legend(fontsize=9)

    ax = axes[1]
    for data, label, color in [(no_hyst, "Sem histerese", "#C44E52"), (hyst, "Com histerese", "#55A868")]:
        sorted_data = np.sort(data)
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.step(sorted_data, yvals, where="post", label=label, color=color, lw=2)
    ax.set_xlabel("Trocas de planejador por episódio")
    ax.set_ylabel("Fração acumulada de episódios (ECDF)")
    ax.set_title("(b) ECDF — fração com ≥N trocas")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    pct_no_hyst = 100 * (no_hyst >= 2).sum() / len(no_hyst)
    pct_hyst = 100 * (hyst >= 2).sum() / len(hyst)
    fig.suptitle(f"Chattering do ρ-criterion: episódios com ≥2 trocas caem de "
                 f"{pct_no_hyst:.1f}% (sem histerese) para {pct_hyst:.1f}% (com histerese) — n=1.500 cada",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    out_dir = os.path.join(FIGS, "core")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_chattering_hysteresis.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Salvo:", path)

    fig.set_size_inches(14, 6)
    slides_dir = os.path.join(FIGS, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    path = os.path.join(slides_dir, "fig_chattering_hysteresis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo (slides):", path)

    print(f"\nSem histerese: média={no_hyst.mean():.2f}, max={no_hyst.max()}, "
          f">=2 trocas: {(no_hyst>=2).sum()}/{len(no_hyst)} ({pct_no_hyst:.1f}%)")
    print(f"Com histerese: média={hyst.mean():.2f}, max={hyst.max()}, "
          f">=2 trocas: {(hyst>=2).sum()}/{len(hyst)} ({pct_hyst:.1f}%)")


if __name__ == "__main__":
    main()
