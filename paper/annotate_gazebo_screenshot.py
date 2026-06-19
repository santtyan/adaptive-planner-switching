"""
Processa e anota screenshot do Gazebo para apresentação.

Uso:
    # Depois de salvar o screenshot em paper/figs/gazebo_screenshots/gz_raw.png:
    python3 paper/annotate_gazebo_screenshot.py

    # Com arquivo específico:
    python3 paper/annotate_gazebo_screenshot.py --input paper/figs/gazebo_screenshots/gz_raw.png

Saídas:
    paper/figs/gazebo_screenshots/gz_annotated.png  — anotado para apresentação
    paper/figs/gazebo_screenshots/gz_cropped.png    — só a arena recortada
    paper/figs/gazebo_screenshots/gz_panel.png      — painel 2×2 com legendas
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

UFG_GREEN = "#006633"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="paper/figs/gazebo_screenshots/gz_raw.png")
    p.add_argument("--out", default="paper/figs/gazebo_screenshots")
    # Região da arena na imagem (ajustar conforme o screenshot real)
    # crop: left, top, right, bottom em fração da imagem total
    p.add_argument("--crop-left",   type=float, default=0.19)
    p.add_argument("--crop-top",    type=float, default=0.12)
    p.add_argument("--crop-right",  type=float, default=0.99)
    p.add_argument("--crop-bottom", type=float, default=0.97)
    return p.parse_args()


def load_and_crop(path, cl, ct, cr, cb):
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(ct * h), int(cb * h)
    x0, x1 = int(cl * w), int(cr * w)
    return img, img[y0:y1, x0:x1]


def annotate_figure(img_full, img_crop, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ── Figura 1: imagem completa com seta apontando para arena ──────────────
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.imshow(img_full)
    ax.axis("off")
    h, w = img_full.shape[:2]
    # Anotação: arena
    ax.annotate("Arena de\nnavegação\n(4m × 4m)",
                xy=(w * 0.60, h * 0.50), xytext=(w * 0.30, h * 0.20),
                fontsize=12, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white", lw=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=UFG_GREEN, alpha=0.85))
    # Anotação: obstáculos
    ax.annotate("Obstáculos\ncilíndricos",
                xy=(w * 0.55, h * 0.35), xytext=(w * 0.75, h * 0.15),
                fontsize=11, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.85))
    # Anotação: robô (linhas LIDAR visíveis)
    ax.annotate("TurtleBot3 Waffle\n(linhas: LIDAR + vetor goal)",
                xy=(w * 0.53, h * 0.53), xytext=(w * 0.10, h * 0.75),
                fontsize=11, color="white", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#003366", alpha=0.85))
    ax.set_title("TurtleBot3 Waffle navegando em Gazebo Classic\n(SAC treinando — visão top-down)",
                 fontsize=13, fontweight="bold", color=UFG_GREEN, pad=8)
    path1 = os.path.join(out_dir, "gz_annotated.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Salvo: {path1}")
    plt.close(fig)

    # ── Figura 2: só a arena recortada ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(img_crop)
    ax.axis("off")
    ax.set_title("Arena — dense_custom.world\n(7 obstáculos, TurtleBot3 Waffle)",
                 fontsize=11, fontweight="bold", pad=6)
    path2 = os.path.join(out_dir, "gz_cropped.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Salvo: {path2}")
    plt.close(fig)

    # ── Figura 3: painel 1×2 — Gazebo + heatmap switching ────────────────────
    heatmap_path = "paper/figs/switching_heatmap.png"
    if os.path.exists(heatmap_path):
        img_heat = mpimg.imread(heatmap_path)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150,
                                 gridspec_kw={"wspace": 0.05})

        axes[0].imshow(img_crop)
        axes[0].axis("off")
        axes[0].set_title("Simulação real — Gazebo Classic", fontsize=12,
                           fontweight="bold", pad=6)

        axes[1].imshow(img_heat)
        axes[1].axis("off")
        axes[1].set_title("Decisão do switcher ρ*=0,30\n(azul=A*, vermelho=SAC)",
                           fontsize=12, fontweight="bold", pad=6)

        fig.suptitle("Do simulador ao critério adaptivo: ambiente real vs decisão espacial",
                     fontsize=13, fontweight="bold", color=UFG_GREEN, y=1.01)
        path3 = os.path.join(out_dir, "gz_vs_heatmap.png")
        fig.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Salvo: {path3}")
        plt.close(fig)
    else:
        print("[WARN] switching_heatmap.png não encontrado — painel gz+heatmap não gerado")


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERRO] Screenshot não encontrado: {args.input}")
        print("Salve o screenshot do Gazebo em:", args.input)
        print("(Alt+Print Screen com a janela do Gazebo em foco → salvar como PNG)")
        sys.exit(1)

    print(f"Processando: {args.input}")
    img_full, img_crop = load_and_crop(
        args.input,
        args.crop_left, args.crop_top,
        args.crop_right, args.crop_bottom,
    )
    print(f"Imagem: {img_full.shape[1]}×{img_full.shape[0]} px → recorte {img_crop.shape[1]}×{img_crop.shape[0]} px")
    annotate_figure(img_full, img_crop, args.out)
    print("\nPronto. Ajuste --crop-* se o recorte não ficou correto.")


if __name__ == "__main__":
    main()
