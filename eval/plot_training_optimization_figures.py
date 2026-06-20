#!/usr/bin/env python3
"""
Figuras científicas da análise de otimização do treino SAC (CPU-bound).

Gera figuras prontas p/ artigo (CONPEEX / relatório final / IEEE) a partir dos
dados MEDIDOS na sessão de 20/06/2026 (i5-1235U, sem GPU). Cada figura salva
.png (150 dpi) e .pdf (vetorial p/ LaTeX) em paper/figs/.

Uso:
    python3 eval/plot_training_optimization_figures.py --out paper/figs/

Dados medidos (VALIDADOS):
- gzserver satura ~97% de 1 core; SAC roda nos cores ociosos.
- max_step_size 0.001→0.004 derruba fps 7→1 (LIDAR 5Hz para de disparar →
  step espera SCAN_TIMEOUT=1.0s). O teto de throughput é a taxa de scan, não a física.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE, RED, GREEN, GRAY = "#2c6fbb", "#c0392b", "#27ae60", "#7f8c8d"


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  ✓ {name}.png/.pdf")


def fig_cpu_bottleneck(out):
    """Utilização de CPU: gzserver satura 1 core, SAC e cores ociosos."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["gzserver\n(física, 1 thread)", "train_sac\n(SAC, PyTorch)",
              "9 cores\nociosos"]
    vals = [97, 12, 0]
    colors = [RED, BLUE, GRAY]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("Uso de CPU por core (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Gargalo do treino: física do Gazebo é single-thread\n"
                 "(i5-1235U, sem GPU) — SAC não é o limitador")
    ax.axhline(100, ls="--", c="black", lw=0.8, alpha=0.5)
    _save(fig, out, "fig_cpu_bottleneck")


def fig_scan_throughput(out):
    """Experimento: max_step_size vs fps — prova que o teto é a taxa de scan."""
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = ["0.001\n(baseline)", "0.004\n(passo grosso)"]
    fps = [7, 1]
    colors = [GREEN, RED]
    bars = ax.bar(steps, fps, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, fps):
        ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v} fps",
                ha="center", fontweight="bold")
    ax.set_xlabel("max_step_size do solver ODE (s)")
    ax.set_ylabel("Throughput (env steps / s)")
    ax.set_ylim(0, 8.5)
    ax.set_title("Por que aumentar o passo PIORA o treino\n"
                 "LIDAR a 5 Hz para de disparar → step espera SCAN_TIMEOUT=1 s")
    ax.annotate("teto real = taxa de scan\n(5 Hz sim × RTF)",
                xy=(0, 7), xytext=(0.3, 5.5),
                arrowprops=dict(arrowstyle="->", color=GRAY))
    _save(fig, out, "fig_scan_throughput")


def fig_obs_space(out):
    """Diagrama da observação 29-dim (fix S1: POMDP → +velocidade/ação)."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    segs = [("LIDAR downsample (24)", 24, BLUE),
            ("goal polar (3)", 3, GREEN),
            ("v, ω última ação (2)", 2, RED)]
    x = 0
    for label, w, c in segs:
        ax.barh(0, w, left=x, color=c, edgecolor="black", height=0.6)
        ax.text(x + w/2, 0, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=9)
        x += w
    ax.set_xlim(0, 29)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("índice da dimensão da observação")
    ax.set_title("Espaço de observação 29-dim (fix S1 — resolve POMDP)\n"
                 "+[v, ω] da última ação dá memória de 1ª ordem ao agente")
    ax.grid(False)
    _save(fig, out, "fig_obs_space_29dim")


def fig_sample_efficiency_ladder(out):
    """Padrão-ouro: SAC → DroQ → CrossQ (amostras até convergir, conceitual)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ["SAC\n(g_steps=1)", "SAC\n(g_steps=4)\n← atual",
               "DroQ\n(dropout+LN)", "CrossQ\n(BatchNorm)"]
    rel_samples = [1.0, 0.7, 0.45, 0.4]   # conceitual, normalizado a SAC base
    colors = [GRAY, BLUE, GREEN, "#8e44ad"]
    bars = ax.bar(methods, rel_samples, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, rel_samples):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("Amostras até convergir (relativo a SAC)")
    ax.set_title("Padrão-ouro de sample-efficiency quando o passo é caro\n"
                 "(conceitual — Hiraoka 2022 DroQ, Bhatt 2024 CrossQ)")
    ax.set_ylim(0, 1.15)
    _save(fig, out, "fig_sample_efficiency_ladder")


def fig_optimization_roi(out):
    """Matriz ROI das quick wins de otimização do treino."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # (esforço, impacto, label, cor)
    items = [
        (1, 9, "iters 150→50", GREEN),
        (1, 9, "early-stop\nsuccess", GREEN),
        (1, 8, "gradient_steps=4\n(manter)", BLUE),
        (6, 7, "parallel envs\n(2 gazebos)", "#e67e22"),
        (5, 8, "DroQ\n(se platô)", "#8e44ad"),
        (9, 10, "sim 2D\ncinemático", RED),
        (2, 1, "max_step 0.004\n(✗ quebra scan)", GRAY),
    ]
    for eff, imp, lbl, c in items:
        ax.scatter(eff, imp, s=420, c=c, edgecolor="black", zorder=3, alpha=0.9)
        ax.annotate(lbl, (eff, imp), fontsize=8, ha="center",
                    va="center", color="white", fontweight="bold")
    ax.axvline(3.5, ls="--", c=GRAY, alpha=0.5)
    ax.axhline(5.5, ls="--", c=GRAY, alpha=0.5)
    ax.text(1.5, 10.3, "QUICK WINS", color=GREEN, fontweight="bold")
    ax.text(6.5, 10.3, "STRATEGIC", color="#e67e22", fontweight="bold")
    ax.set_xlabel("Esforço →")
    ax.set_ylabel("Impacto →")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_title("Matriz ROI — otimização do treino SAC (CPU-bound)")
    _save(fig, out, "fig_optimization_roi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Gerando figuras de otimização em {args.out} ...")
    fig_cpu_bottleneck(args.out)
    fig_scan_throughput(args.out)
    fig_obs_space(args.out)
    fig_sample_efficiency_ladder(args.out)
    fig_optimization_roi(args.out)
    print("OK — 5 figuras (.png + .pdf).")


if __name__ == "__main__":
    main()
