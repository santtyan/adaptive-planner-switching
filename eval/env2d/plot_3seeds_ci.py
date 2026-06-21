"""
plot_3seeds_ci.py — Curva de aprendizado 2D com IC 95% (3 seeds).

Lê os logs de treino dos 3 seeds e gera:
  fig_2d_learning_curve_ci.png — média ± IC 95% da success_rate por step
"""
import sys, os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "paper", "figs")
LOGS = os.path.join(ROOT, "ros2_ws", "logs")

SEEDS = [42, 123, 7]
SEED_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


def load_seed(seed: int) -> pd.DataFrame | None:
    """Carrega CSV de monitor do seed ou log de eval do train_2d."""
    # Tenta CSV de monitor (gerado pelo Monitor wrapper do SB3)
    patterns = [
        f"{ROOT}/ros2_ws/logs/*seed{seed}*/monitor*.csv",
        f"{ROOT}/ros2_ws/logs/sac_2d_seed{seed}*.csv",
        f"/tmp/train_seed{seed}_eval.csv",
    ]
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            df = pd.read_csv(files[0], comment="#")
            return df
    return None


def load_eval_log(seed: int) -> tuple | None:
    """Extrai steps e success_rate do log do train_2d."""
    log_path = f"/tmp/train_seed{seed}.log"
    if not os.path.exists(log_path):
        return None
    steps, rates = [], []
    with open(log_path) as f:
        for line in f:
            if "[eval @" in line:
                try:
                    parts = line.split()
                    step = int(parts[2].rstrip("]"))
                    sr_str = [p for p in parts if p.startswith("success=")][0]
                    sr = float(sr_str.replace("success=", "").replace("%", "")) / 100
                    steps.append(step)
                    rates.append(sr)
                except Exception:
                    pass
    if steps:
        return np.array(steps), np.array(rates)
    return None


def plot_ci(data: dict, out: str):
    """data: {seed: (steps_arr, rates_arr)}"""
    # Interpola para grade comum
    all_steps = sorted(set(s for _, (st, _) in data.items() for s in st))
    grid = np.array(all_steps)

    interp = {}
    for seed, (st, ra) in data.items():
        interp[seed] = np.interp(grid, st, ra)

    matrix = np.stack(list(interp.values()))   # (n_seeds, n_steps)
    mean = matrix.mean(axis=0)
    sem  = stats.sem(matrix, axis=0)
    ci95 = sem * stats.t.ppf(0.975, df=len(SEEDS)-1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Linha de cada seed
    for i, (seed, (st, ra)) in enumerate(data.items()):
        ax.plot(st, ra, color=SEED_COLORS[i], alpha=0.4, lw=1.2,
                label=f"seed={seed}")

    # Média + IC 95%
    ax.plot(grid, mean, color="#1A237E", lw=2.5, label="Média (3 seeds)")
    ax.fill_between(grid, mean - ci95, mean + ci95,
                    color="#1A237E", alpha=0.18, label="IC 95%")

    ax.axhline(0.9, color="#F44336", ls="--", lw=1.2, label="90% (limiar)")
    ax.set_xlabel("Steps de treinamento", fontsize=11)
    ax.set_ylabel("Taxa de sucesso", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.set_title("Curva de Aprendizado SAC — Env 2D (3 seeds independentes)\n"
                 "world=sparse, MAX_STEPS=200, R_APPROACH=10",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path / f"fig_2d_learning_curve_ci.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ fig_2d_learning_curve_ci.png")
    print(f"  Convergência média: {grid[mean >= 0.9][0] if (mean >= 0.9).any() else 'N/A'} steps")


def main():
    data = {}
    for seed in SEEDS:
        result = load_eval_log(seed)
        if result is None:
            print(f"  [seed {seed}] log não encontrado em /tmp/train_seed{seed}.log")
            continue
        steps, rates = result
        data[seed] = (steps, rates)
        print(f"  [seed {seed}] {len(steps)} pontos de eval, max={max(rates):.0%}")

    if len(data) < 2:
        print("Precisa de pelo menos 2 seeds concluídos para plotar IC.")
        print("Aguarde o treino de todos os seeds terminar.")
        return

    plot_ci(data, os.path.join(FIGS, "2d"))
    print(f"\nSalvo em paper/figs/2d/fig_2d_learning_curve_ci.png")


if __name__ == "__main__":
    main()
