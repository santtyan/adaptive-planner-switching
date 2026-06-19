"""
Figuras de benchmark pós-treinamento: matriz de desfecho, boxplot e barras por densidade.

Pode rodar de dois modos:
  1. --synthetic  — dados placeholder (rodar antes do benchmark real)
  2. --data results_ros2/benchmark.csv  — CSV gerado pelo benchmark_sac_vs_nav2.py

Uso:
    # Sintético:
    python3 plot_benchmark_figures.py --synthetic --out paper/figs/

    # Real:
    python3 plot_benchmark_figures.py --data results_ros2/benchmark.csv --out paper/figs/

Saídas:
    paper/figs/outcome_matrix.png/.pdf   — matriz de desfecho goal/colisão/timeout × faixa de ρ
    paper/figs/duration_boxplot.png/.pdf — boxplot tempo-até-goal por método
    paper/figs/success_by_density.png/.pdf — barras success rate por faixa de ρ
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paleta consistente com os outros scripts ─────────────────────────────────
COLORS = {
    "astar":    "#1f77b4",
    "sac":      "#d62728",
    "adaptive": "#2ca02c",
}
LABELS = {
    "astar":    "A* (Nav2)",
    "sac":      "SAC (RL)",
    "adaptive": "Adaptativo",
}

# Faixas de densidade usadas no benchmark
DENSITY_BINS = [
    (0.00, 0.15, "Baixa\n[0.00–0.15)"),
    (0.15, 0.30, "Média-baixa\n[0.15–0.30)"),
    (0.30, 0.50, "Média-alta\n[0.30–0.50)"),
    (0.50, 1.00, "Alta\n[0.50–1.00)"),
]
BIN_LABELS = [b[2] for b in DENSITY_BINS]

OUTCOMES = ["goal", "colisão", "timeout"]


# ── Gerador sintético ─────────────────────────────────────────────────────────

def _synthetic_df(n_per_bin: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    # Probabilidades plausíveis por método e faixa (sucesso, colisão, timeout)
    probs = {
        #              baixa          média-baixa    média-alta     alta
        "astar":    [(0.92, 0.05), (0.82, 0.10), (0.60, 0.20), (0.38, 0.35)],
        "sac":      [(0.55, 0.30), (0.65, 0.22), (0.78, 0.14), (0.72, 0.18)],
        "adaptive": [(0.90, 0.06), (0.85, 0.09), (0.82, 0.11), (0.70, 0.19)],
    }
    # Durations plausíveis (seg) quando há sucesso
    dur_mean = {"astar": 8.0, "sac": 12.0, "adaptive": 9.5}
    dur_std  = {"astar": 2.0, "sac":  4.0, "adaptive": 3.0}

    for method, prob_list in probs.items():
        for bi, (lo, hi, _) in enumerate(DENSITY_BINS):
            p_goal, p_col = prob_list[bi]
            p_timeout = max(0.0, 1.0 - p_goal - p_col)
            for _ in range(n_per_bin):
                density = rng.uniform(lo, hi)
                r = rng.random()
                if r < p_goal:
                    outcome = "goal"
                    duration = max(1.0, rng.gauss(dur_mean[method], dur_std[method]))
                elif r < p_goal + p_col:
                    outcome = "colisão"
                    duration = max(1.0, rng.gauss(dur_mean[method] * 0.5, 1.0))
                else:
                    outcome = "timeout"
                    duration = 30.0  # timeout fixo
                rows.append({
                    "method": method,
                    "density": density,
                    "outcome": outcome,
                    "duration_s": duration,
                })
    return pd.DataFrame(rows)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normaliza nomes esperados: success(bool) → outcome(str)
    if "success" in df.columns and "outcome" not in df.columns:
        df["outcome"] = df.apply(
            lambda r: "goal" if r["success"] else (
                "colisão" if r.get("collision", False) else "timeout"
            ), axis=1
        )
    if "duration_s" not in df.columns and "time_s" in df.columns:
        df = df.rename(columns={"time_s": "duration_s"})
    # Renomeia condition → method se necessário
    if "condition" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"condition": "method"})
    return df


def _bin_density(df: pd.DataFrame) -> pd.DataFrame:
    def assign_bin(rho):
        for lo, hi, label in DENSITY_BINS:
            if lo <= rho < hi:
                return label
        return DENSITY_BINS[-1][2]
    df = df.copy()
    df["density_bin"] = df["density"].apply(assign_bin)
    return df


# ── Figura 1: Matriz de desfecho ─────────────────────────────────────────────

def plot_outcome_matrix(df: pd.DataFrame, out: str, no_pdf: bool):
    methods = ["astar", "sac", "adaptive"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150,
                             gridspec_kw={"wspace": 0.35})

    for ax, method in zip(axes, methods):
        rows = []
        for lo, hi, label in DENSITY_BINS:
            sub = df[(df["method"] == method) & (df["density_bin"] == label)]
            total = len(sub)
            if total == 0:
                rows.append([0, 0, 0])
                continue
            counts = [
                100 * (sub["outcome"] == o).sum() / total
                for o in OUTCOMES
            ]
            rows.append(counts)

        matrix = np.array(rows)  # shape (4 bins, 3 outcomes)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=100, interpolation="nearest")
        ax.set_xticks(range(len(OUTCOMES)))
        ax.set_xticklabels(OUTCOMES, fontsize=10)
        ax.set_yticks(range(len(BIN_LABELS)))
        ax.set_yticklabels(BIN_LABELS, fontsize=8)
        ax.set_title(LABELS[method], fontsize=12, fontweight="bold",
                     color=COLORS[method])
        ax.set_xlabel("Desfecho", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Faixa de densidade ρ", fontsize=9)

        # Anotar valores
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val < 30 or val > 80 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="% dos trials" if ax is axes[-1] else "")

    fig.suptitle("Matriz de Desfecho por Método e Densidade",
                 fontsize=13, fontweight="bold")
    _save(fig, out, "outcome_matrix", no_pdf)


# ── Figura 2: Boxplot de duração por método ───────────────────────────────────

def plot_duration_boxplot(df: pd.DataFrame, out: str, no_pdf: bool):
    methods = ["astar", "sac", "adaptive"]
    data_by_bin = []
    tick_labels = []

    for lo, hi, label in DENSITY_BINS:
        bin_data = []
        for method in methods:
            sub = df[
                (df["method"] == method) &
                (df["density_bin"] == label) &
                (df["outcome"] == "goal")
            ]["duration_s"]
            bin_data.append(sub.values)
        data_by_bin.append(bin_data)
        tick_labels.append(label)

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)

    n_bins = len(DENSITY_BINS)
    n_methods = len(methods)
    group_width = 0.7
    box_width = group_width / n_methods

    positions_all = []
    colors_all = []
    data_all = []

    for bi in range(n_bins):
        center = bi * (n_methods * box_width + 0.3)
        for mi, method in enumerate(methods):
            pos = center + mi * box_width
            positions_all.append(pos)
            colors_all.append(COLORS[method])
            data_all.append(data_by_bin[bi][mi])

    bp = ax.boxplot(data_all, positions=positions_all, widths=box_width * 0.85,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", lw=2))

    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Eixo x — ticks por grupo
    group_centers = [
        bi * (n_methods * box_width + 0.3) + (n_methods - 1) * box_width / 2
        for bi in range(n_bins)
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([b[2] for b in DENSITY_BINS], fontsize=9)
    ax.set_xlabel("Faixa de densidade ρ", fontsize=11)
    ax.set_ylabel("Tempo até o goal (s)", fontsize=11)
    ax.set_title("Tempo até o Goal por Método e Faixa de Densidade\n(somente episódios com sucesso)",
                 fontsize=12, fontweight="bold")

    legend_patches = [
        plt.matplotlib.patches.Patch(facecolor=COLORS[m], alpha=0.75, label=LABELS[m])
        for m in methods
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "duration_boxplot", no_pdf)


# ── Figura 3: Barras success rate por faixa de ρ ─────────────────────────────

def plot_success_by_density(df: pd.DataFrame, out: str, no_pdf: bool):
    methods = ["astar", "sac", "adaptive"]
    n_bins = len(DENSITY_BINS)
    n_methods = len(methods)

    success_rates = np.zeros((n_bins, n_methods))
    for bi, (lo, hi, label) in enumerate(DENSITY_BINS):
        for mi, method in enumerate(methods):
            sub = df[(df["method"] == method) & (df["density_bin"] == label)]
            if len(sub) == 0:
                continue
            success_rates[bi, mi] = 100 * (sub["outcome"] == "goal").sum() / len(sub)

    x = np.arange(n_bins)
    bar_width = 0.22
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    for mi, method in enumerate(methods):
        bars = ax.bar(x + offsets[mi], success_rates[:, mi],
                      width=bar_width * 0.9,
                      color=COLORS[method], alpha=0.82,
                      label=LABELS[method], zorder=3)
        # Anotar valor
        for bar, val in zip(bars, success_rates[:, mi]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.0,
                        f"{val:.0f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    # Linha do threshold ρ*
    ax.axvline(x=1.5, color="#333333", lw=1.5, ls="--", alpha=0.6,
               label="ρ* = 0.30 (threshold)")

    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in DENSITY_BINS], fontsize=9)
    ax.set_xlabel("Faixa de densidade ρ", fontsize=11)
    ax.set_ylabel("Taxa de sucesso (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Taxa de Sucesso por Método e Faixa de Densidade",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    _save(fig, out, "success_by_density", no_pdf)


# ── Utilitário ────────────────────────────────────────────────────────────────

def _save(fig, out_dir: str, name: str, no_pdf: bool):
    path_png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    print(f"Salvo: {path_png}")
    if not no_pdf:
        path_pdf = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(path_pdf, bbox_inches="tight")
        print(f"Salvo: {path_pdf}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="Usar dados placeholder (não precisa de benchmark real)")
    p.add_argument("--data", default=None,
                   help="CSV gerado pelo benchmark_sac_vs_nav2.py")
    p.add_argument("--out", default="paper/figs")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--n-per-bin", type=int, default=30,
                   help="Trials sintéticos por faixa×método (padrão 30)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.data and not args.synthetic:
        df = _load_csv(args.data)
        print(f"[INFO] Carregado {len(df)} linhas de {args.data}")
    else:
        df = _synthetic_df(n_per_bin=args.n_per_bin)
        print(f"[INFO] Usando dados sintéticos ({len(df)} linhas placeholder).")

    df = _bin_density(df)

    plot_outcome_matrix(df, args.out, args.no_pdf)
    plot_duration_boxplot(df, args.out, args.no_pdf)
    plot_success_by_density(df, args.out, args.no_pdf)

    print("\nPronto. Substitua pelo CSV real com: --data results_ros2/benchmark.csv")


if __name__ == "__main__":
    main()
