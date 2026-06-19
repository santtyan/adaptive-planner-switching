"""
Gera slide único de apresentação CONPEEX 2026 (A4 paisagem, 300 dpi).

Uso:
    python3 paper/generate_conpeex_slide.py --out paper/figs/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

UFG_GREEN  = "#006633"
UFG_LIGHT  = "#e8f4ee"
ASTAR_BLUE = "#1f77b4"
SAC_RED    = "#d62728"
GRAY_BOX   = "#f0f0f0"
DARK       = "#1a1a1a"


def draw_criterion_diagram(ax):
    """Diagrama do ρ-criterion: sensor → ρ → decisão → A*/SAC."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel="", color=GRAY_BOX, fontsize=10):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#aaaaaa", lw=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color=DARK, zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=8, color="#555555", zorder=4)

    def arrow(x1, x2, y=2.0):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
                    zorder=5)

    # Caixas
    box(0.1, 1.2, 1.8, 1.6, "Sensor", "LIDAR 360°", color="#ddeeff")
    arrow(1.9, 2.5)
    box(2.5, 1.2, 2.0, 1.6, "Estimador", "ρ = densidade\nlocal", color="#fff3cd")
    arrow(4.5, 5.3)

    # Losango de decisão
    diamond_x, diamond_y = 5.8, 2.0
    diamond = plt.Polygon(
        [[diamond_x, diamond_y + 0.9],
         [diamond_x + 1.1, diamond_y],
         [diamond_x, diamond_y - 0.9],
         [diamond_x - 1.1, diamond_y]],
        closed=True, facecolor="#fff8e1", edgecolor="#aaaaaa", lw=1.2, zorder=3)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, "ρ < 0,30?", ha="center", va="center",
            fontsize=9, fontweight="bold", color=DARK, zorder=4)

    # Seta SIM → A*
    ax.annotate("", xy=(8.2, 3.3), xytext=(6.5, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=ASTAR_BLUE, lw=1.5), zorder=5)
    ax.text(7.1, 3.25, "Sim", fontsize=8, color=ASTAR_BLUE, fontweight="bold")
    box(8.1, 2.8, 1.7, 1.0, "A*", "Nav2", color="#ddeeff", fontsize=11)

    # Seta NÃO → SAC
    ax.annotate("", xy=(8.2, 0.8), xytext=(6.5, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=SAC_RED, lw=1.5), zorder=5)
    ax.text(7.1, 0.75, "Não", fontsize=8, color=SAC_RED, fontweight="bold")
    box(8.1, 0.2, 1.7, 1.0, "SAC", "Stable-Baselines3", color="#fde8e8", fontsize=11)

    ax.set_title("Critério de Seleção Adaptiva (ρ-criterion)",
                 fontsize=11, fontweight="bold", color=DARK, pad=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/figs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16.54, 11.69), dpi=150)  # A4 paisagem em polegadas
    fig.patch.set_facecolor("white")

    # Faixa de cabeçalho
    header = fig.add_axes([0, 0.88, 1, 0.12])
    header.set_facecolor(UFG_GREEN)
    header.axis("off")
    header.text(0.5, 0.70,
                "FRAMEWORK ADAPTIVO PARA SELEÇÃO DE PLANEJADORES DE TRAJETÓRIA EM NAVEGAÇÃO AUTÔNOMA",
                ha="center", va="center", fontsize=14, fontweight="bold", color="white",
                transform=header.transAxes)
    header.text(0.5, 0.25,
                "Yan Santos Leite  |  Orientador: Prof. Dr. Aldo André Diaz Salazar  |  EMC/UFG  |  PIBIC/FAPEG 2025-2026  |  CONPEEX 2026",
                ha="center", va="center", fontsize=10, color="#ccffcc",
                transform=header.transAxes)

    # Grid principal: 2 linhas × 3 colunas
    gs = GridSpec(2, 3, figure=fig,
                  left=0.03, right=0.97, top=0.87, bottom=0.08,
                  wspace=0.28, hspace=0.38)

    # ── Célula (0,0): Problema + Solução ─────────────────────────────────────
    ax_prob = fig.add_subplot(gs[0, 0])
    ax_prob.axis("off")
    ax_prob.set_facecolor(UFG_LIGHT)
    ax_prob.patch.set_visible(True)

    texto = (
        "PROBLEMA\n\n"
        "Algoritmos clássicos (A*) são ótimos em\n"
        "espaços abertos mas falham em alta\n"
        "densidade de obstáculos.\n\n"
        "Políticas RL (SAC) adaptam-se a ambientes\n"
        "complexos mas são custosas onde A* já\n"
        "resolve bem.\n\n"
        "SOLUÇÃO\n\n"
        "π(ρ) = { A*  se ρ < 0,30\n"
        "        { SAC se ρ ≥ 0,30\n\n"
        "Threshold ρ* = 0,30 determinado por\n"
        "1.500 experimentos Monte Carlo."
    )
    ax_prob.text(0.05, 0.97, texto, va="top", ha="left", fontsize=9.5,
                 transform=ax_prob.transAxes, linespacing=1.6,
                 fontfamily="monospace")
    ax_prob.set_title("Motivação e Critério ρ", fontsize=11, fontweight="bold",
                      color=DARK, pad=5)

    # ── Célula (0,1): Diagrama do critério ───────────────────────────────────
    ax_diag = fig.add_subplot(gs[0, 1])
    draw_criterion_diagram(ax_diag)

    # ── Célula (0,2): Heatmap de switching ───────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 2])
    img_heat = mpimg.imread(os.path.join(args.out, "switching_heatmap.png"))
    ax_heat.imshow(img_heat)
    ax_heat.axis("off")
    ax_heat.set_title("Decisão Espacial do Switcher (modelo analítico, ρ*=0,30)",
                      fontsize=10, fontweight="bold", color=DARK, pad=5)

    # ── Célula (1,0): Benchmark clássicos ────────────────────────────────────
    ax_bench = fig.add_subplot(gs[1, 0])
    ax_bench.axis("off")

    tabela = [
        ["Algoritmo", "100 nós", "2.500 nós", "Complexidade"],
        ["Dijkstra",  "0,07 ms\n3,7 KB",   "2,46 ms\n85 KB",   "O(V log V)"],
        ["A*",        "0,07 ms\n6,6 KB",   "3,09 ms\n220 KB",  "O(V log V)"],
        ["Floyd-W.",  "20,9 ms\n271 KB",   "inviável",          "O(V³)"],
        ["Johnson",   "5,5 ms\n603 KB",    "inviável",          "O(VE log V)"],
    ]
    col_widths = [0.22, 0.22, 0.28, 0.28]
    row_h = 0.155
    y0 = 0.97
    for ri, row in enumerate(tabela):
        x = 0.01
        bg = UFG_LIGHT if ri == 0 else ("white" if ri % 2 == 0 else "#f9f9f9")
        rect = FancyBboxPatch((0, y0 - row_h * (ri + 1) + 0.01), 0.98, row_h - 0.01,
                              boxstyle="square,pad=0", facecolor=bg,
                              edgecolor="#dddddd", lw=0.5,
                              transform=ax_bench.transAxes, zorder=2)
        ax_bench.add_patch(rect)
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            fw = "bold" if ri == 0 else "normal"
            color = DARK if ri > 0 and "inviável" not in cell else ("#cc0000" if "inviável" in cell else DARK)
            ax_bench.text(x + cw / 2, y0 - row_h * ri - row_h / 2, cell,
                          ha="center", va="center", fontsize=8.5,
                          fontweight=fw, color=color,
                          transform=ax_bench.transAxes, linespacing=1.3, zorder=3)
            x += cw

    ax_bench.set_title("Benchmark de Algoritmos Clássicos", fontsize=11, fontweight="bold",
                       color=DARK, pad=5)

    # ── Célula (1,1): Resultados Fase 1 ──────────────────────────────────────
    ax_res = fig.add_subplot(gs[1, 1])
    ax_res.axis("off")

    metricas = [
        ("Taxa de sucesso",    "85,3%",     "Framework adaptivo"),
        ("PPO fixo (melhor baseline)", "76,0%", "↑ 9,3 pp de diferença"),
        ("Regret vs oracle",   "2,2%",      "↓ dentro do limite 5%"),
        ("Threshold ρ*",       "0,30",      "1.500 experimentos Monte Carlo"),
        ("Generalização",      "N agentes", "CBS + ρ-criterion sem modificação"),
    ]
    y = 0.93
    for label, valor, sub in metricas:
        ax_res.text(0.03, y, label, fontsize=9, color="#555555",
                    transform=ax_res.transAxes, va="top")
        ax_res.text(0.97, y, valor, fontsize=13, fontweight="bold",
                    color=UFG_GREEN, transform=ax_res.transAxes, va="top", ha="right")
        ax_res.text(0.97, y - 0.068, sub, fontsize=7.5, color="#888888",
                    transform=ax_res.transAxes, va="top", ha="right")
        ax_res.plot([0.02, 0.98], [y - 0.11, y - 0.11], color="#eeeeee", lw=0.8,
                    transform=ax_res.transAxes)
        y -= 0.175

    ax_res.set_title("Resultados — Fase 1 (Monte Carlo)", fontsize=11, fontweight="bold",
                     color=DARK, pad=5)

    # ── Célula (1,2): painel duplo — Gazebo + progressão de densidade ──────────
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    ax_gz_outer = fig.add_subplot(gs[1, 2])
    ax_gz_outer.axis("off")
    ax_gz_outer.set_title("Implementação ROS2 + Progressão de Densidade",
                           fontsize=10, fontweight="bold", color=DARK, pad=5)

    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2],
                                    hspace=0.12)

    # Sub-painel superior: Gazebo
    ax_gz = fig.add_subplot(inner[0])
    gz_panel = os.path.join(args.out, "gazebo_screenshots", "gz_panel.png")
    gz_lidar = os.path.join(args.out, "gazebo_screenshots", "gz_02_lidar_disc.png")
    gz_path = gz_panel if os.path.exists(gz_panel) else gz_lidar
    if os.path.exists(gz_path):
        img_gz = mpimg.imread(gz_path)
        ax_gz.imshow(img_gz)
        ax_gz.axis("off")
        ax_gz.set_title("TurtleBot3 Waffle — Gazebo Classic",
                        fontsize=8, color=DARK, pad=2)
    else:
        ax_gz.axis("off")
        ax_gz.text(0.5, 0.5, "[Screenshot Gazebo — pendente convergência SAC]",
                   ha="center", va="center", fontsize=8, color="#888888",
                   transform=ax_gz.transAxes)

    # Sub-painel inferior: progressão de densidade
    ax_prog = fig.add_subplot(inner[1])
    dp_path = os.path.join(args.out, "density_progression.png")
    if os.path.exists(dp_path):
        img_dp = mpimg.imread(dp_path)
        ax_prog.imshow(img_dp)
        ax_prog.axis("off")
        ax_prog.set_title("Progressão: esparso → misto → denso (ρ*=0,30)",
                          fontsize=8, color=DARK, pad=2)
    else:
        ax_prog.axis("off")
        ax_prog.text(0.5, 0.5, "[density_progression.png]",
                     ha="center", va="center", fontsize=8, color="#888888",
                     transform=ax_prog.transAxes)

    # ── Rodapé ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.025,
             "Repositório: github.com/santtyan/adaptive-planner-switching  |  "
             "Stack: ROS2 Humble · Gazebo Classic · TurtleBot3 Waffle · SAC/Stable-Baselines3 · Nav2",
             ha="center", fontsize=8.5, color="#666666")

    # ── Salvar ────────────────────────────────────────────────────────────────
    out_png = os.path.join(args.out, "conpeex_slide.png")
    out_pdf = os.path.join(args.out, "conpeex_slide.pdf")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Salvo: {out_png}")
    print(f"Salvo: {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
