"""
F4 do docs/PLANO_CORRECAO.md — Varredura de tau com A*/BC reais, train vs. test.

Fonte: results_abstract/threshold_sweep_real.csv (Passo 4, sweep_threshold_real.py)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_abstract")
FIGS = os.path.join(ROOT, "paper", "figs")


def main():
    df = pd.read_csv(os.path.join(RES, "threshold_sweep_real.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))
    for split, color in [("train", "#4C72B0"), ("test", "#55A868")]:
        sub = df[df.split == split].sort_values("tau")
        ax.plot(sub.tau, sub.regret * 100, "o-", color=color, label=f"{split} (n={sub.n.iloc[0]})", lw=2)

    ax.axvline(0.30, color="gray", linestyle="--", alpha=0.7, label="τ=0,30 (adotado)")
    ax.set_xlabel("Limiar τ")
    ax.set_ylabel("Regret vs. oracle (%)")
    ax.set_title("Recalibração de τ com A*/BC reais\n(regret decresce monotonicamente até τ=0,60, sem mínimo interno)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(FIGS, "core")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_threshold_sweep_real.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Salvo:", path)

    fig.set_size_inches(10, 7)
    slides_dir = os.path.join(FIGS, "slides")
    os.makedirs(slides_dir, exist_ok=True)
    path = os.path.join(slides_dir, "fig_threshold_sweep_real.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Salvo (slides):", path)


if __name__ == "__main__":
    main()
