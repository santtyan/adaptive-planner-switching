"""
Exporta curvas de aprendizado do SAC (TensorBoard → PNG/PDF).

Uso:
    python3 plot_learning_curve.py --logdir ros2_ws/logs/ --out paper/figs/
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tb(logdir: str, tag: str):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_all_runs(logs_root: str, tag: str):
    """Agrega todos os runs sac_42_* em arrays (steps, values) concatenados."""
    root = Path(logs_root)
    all_steps, all_vals = [], []
    for run_dir in sorted(root.glob("sac_42*")):
        s, v = load_tb(str(run_dir), tag)
        if len(s):
            all_steps.append(s)
            all_vals.append(v)
    if not all_steps:
        return np.array([]), np.array([])
    # usa o run com mais steps (o mais completo)
    longest = max(range(len(all_steps)), key=lambda i: all_steps[i][-1])
    return all_steps[longest], all_vals[longest]


def smooth(values, weight=0.85):
    last = values[0]
    smoothed = []
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="ros2_ws/logs/")
    p.add_argument("--out", default="paper/figs/")
    p.add_argument("--smooth", type=float, default=0.85)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SAC — Curvas de Aprendizado (TurtleBot3/Gazebo)", fontsize=13, fontweight="bold")

    metrics = [
        ("rollout/ep_rew_mean",  "Recompensa Média por Episódio",  axes[0, 0], "steelblue"),
        ("rollout/ep_len_mean",  "Duração Média do Episódio (steps)", axes[0, 1], "darkorange"),
        ("train/actor_loss",     "Actor Loss",                      axes[1, 0], "firebrick"),
        ("train/ent_coef",       "Coeficiente de Entropia",         axes[1, 1], "seagreen"),
    ]

    for tag, ylabel, ax, color in metrics:
        steps, vals = load_all_runs(args.logdir, tag)
        if len(steps) == 0:
            ax.set_title(ylabel)
            ax.text(0.5, 0.5, "sem dados", ha="center", va="center", transform=ax.transAxes)
            continue

        vals_s = smooth(vals, args.smooth)
        ax.plot(steps, vals, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(steps, vals_s, color=color, linewidth=2.0, label="suavizado")
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Timesteps", fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # anotação extra em ep_rew_mean
    steps, vals = load_all_runs(args.logdir, "rollout/ep_rew_mean")
    if len(vals):
        best = float(np.max(smooth(vals, args.smooth)))
        axes[0, 0].axhline(best, color="steelblue", linestyle="--", linewidth=1, alpha=0.6,
                           label=f"melhor ≈ {best:.1f}")
        axes[0, 0].axhline(-20, color="red", linestyle=":", linewidth=1, alpha=0.5,
                           label="linha base (apenas colisão)")
        axes[0, 0].legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(args.out, "sac_learning_curve.png")
    out_pdf = os.path.join(args.out, "sac_learning_curve.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Salvo: {out_png}")
    print(f"Salvo: {out_pdf}")


if __name__ == "__main__":
    main()
