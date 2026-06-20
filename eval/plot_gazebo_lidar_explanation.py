"""
Figura explicativa do screenshot Gazebo — LIDAR top-down durante treino SAC.

Recria visualmente o que aparece no gzclient:
- TurtleBot3 Waffle no centro da arena
- 360 raios LIDAR (visão Gazebo) em azul
- 7 obstáculos cilíndricos negros
- Paredes da arena 4×4m
- Anotações explicando cada elemento

Gera: paper/figs/gazebo_screenshots/gz_lidar_explanation.png + .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyArrow

OUT = "paper/figs/gazebo_screenshots/gz_lidar_explanation.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Obstáculos do dense_custom.world (posições aproximadas)
OBSTACLES = [
    ( 0.9,  0.6), (-0.5,  1.1), ( 1.3, -0.7),
    (-1.1, -0.4), ( 0.3, -1.3), (-0.8,  0.0),
    ( 0.0,  0.8),
]
OBS_R = 0.18

# Posição do robô (centro da arena, spawna em posições variadas — usar centro p/ ilustração)
ROBOT_X, ROBOT_Y = -0.3, -0.1
ROBOT_R = 0.22

ARENA = 2.0   # arena vai de -2 a +2

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.set_aspect("equal")
ax.set_facecolor("#1a1a2e")   # fundo escuro igual ao Gazebo
fig.patch.set_facecolor("#1a1a2e")

# Arena (paredes)
arena_rect = Rectangle((-ARENA, -ARENA), 2*ARENA, 2*ARENA,
                        fill=False, edgecolor="#555577", lw=3, zorder=2)
ax.add_patch(arena_rect)

# Chão
floor = Rectangle((-ARENA, -ARENA), 2*ARENA, 2*ARENA,
                  facecolor="#2a2a3e", edgecolor=None, zorder=1)
ax.add_patch(floor)

# ── Raios LIDAR ──────────────────────────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
MAX_RANGE = 3.5

def ray_hit(rx, ry, angle, obstacles, arena=ARENA, max_r=MAX_RANGE):
    """Calcula onde um raio LIDAR bate (obstáculo ou parede)."""
    dx, dy = np.cos(angle), np.sin(angle)
    t_min = max_r

    # Paredes
    for sign in [-1, 1]:
        if abs(dx) > 1e-9:
            t = (sign * arena - rx) / dx
            if 0 < t < t_min:
                y_hit = ry + t * dy
                if -arena <= y_hit <= arena:
                    t_min = t
        if abs(dy) > 1e-9:
            t = (sign * arena - ry) / dy
            if 0 < t < t_min:
                x_hit = rx + t * dx
                if -arena <= x_hit <= arena:
                    t_min = t

    # Obstáculos cilíndricos
    for (ox, oy) in obstacles:
        cx, cy = ox - rx, oy - ry
        a = dx*dx + dy*dy
        b = -2*(cx*dx + cy*dy)
        c = cx*cx + cy*cy - OBS_R**2
        disc = b*b - 4*a*c
        if disc >= 0:
            t = (-b - np.sqrt(disc)) / (2*a)
            if 0 < t < t_min:
                t_min = t

    return rx + t_min*dx, ry + t_min*dy, t_min

# Desenhar raios (subsample para não ficar pesado)
for i, ang in enumerate(angles):
    hx, hy, dist = ray_hit(ROBOT_X, ROBOT_Y, ang, OBSTACLES)
    # Gradiente de cor: azul claro perto, azul escuro longe (igual ao Gazebo)
    alpha = max(0.15, 0.9 - dist / MAX_RANGE * 0.6)
    lw = 1.2 if i % 3 == 0 else 0.5
    ax.plot([ROBOT_X, hx], [ROBOT_Y, hy],
            color="#4488ff", alpha=alpha, lw=lw, zorder=3)

# ── Obstáculos ────────────────────────────────────────────────────────────────
for (ox, oy) in OBSTACLES:
    cyl = Circle((ox, oy), OBS_R, color="#111111", zorder=5)
    cyl_edge = Circle((ox, oy), OBS_R, fill=False, edgecolor="#333333", lw=1.5, zorder=6)
    ax.add_patch(cyl)
    ax.add_patch(cyl_edge)

# ── Robô ──────────────────────────────────────────────────────────────────────
robot_body = Circle((ROBOT_X, ROBOT_Y), ROBOT_R, color="#2255aa", zorder=7)
robot_edge = Circle((ROBOT_X, ROBOT_Y), ROBOT_R, fill=False, edgecolor="#88aaff", lw=2, zorder=8)
ax.add_patch(robot_body)
ax.add_patch(robot_edge)

# Seta de direção do robô
ax.annotate("", xy=(ROBOT_X + 0.35, ROBOT_Y + 0.1),
            xytext=(ROBOT_X, ROBOT_Y),
            arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2))

# ── Anotações explicativas ────────────────────────────────────────────────────
LABEL_COLOR = "white"
ARROW_PROPS = dict(arrowstyle="->", color="#aaaaff", lw=1.2,
                   connectionstyle="arc3,rad=0.2")

# 1. Raios LIDAR
ax.annotate("360 raios LIDAR\n(Hokuyo URG-04LX-UG01)\nRange: 3.5 m",
            xy=(ROBOT_X + 1.8, ROBOT_Y + 1.4),
            xytext=(1.8, 2.1),
            color=LABEL_COLOR, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#4488ff", alpha=0.9),
            arrowprops=ARROW_PROPS)

# 2. Robô
ax.annotate("TurtleBot3 Waffle\n(chassi + LIDAR)",
            xy=(ROBOT_X - 0.22, ROBOT_Y),
            xytext=(-2.3, 0.6),
            color=LABEL_COLOR, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#88aaff", alpha=0.9),
            arrowprops=ARROW_PROPS)

# 3. Obstáculo
ax.annotate("Obstáculo cilíndrico\n(r=0.18m, estático)",
            xy=(-0.8, 0.0),
            xytext=(-2.3, -0.8),
            color=LABEL_COLOR, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#888888", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2,
                           connectionstyle="arc3,rad=-0.2"))

# 4. Raio bloqueado
bx, by, _ = ray_hit(ROBOT_X, ROBOT_Y, np.arctan2(0.6-ROBOT_Y, 0.9-ROBOT_X), OBSTACLES)
ax.annotate("Raio bloqueado\npelo obstáculo",
            xy=(bx, by),
            xytext=(1.5, -1.5),
            color=LABEL_COLOR, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#ff8844", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#ff8844", lw=1.2,
                           connectionstyle="arc3,rad=-0.3"))

# 5. Parede
ax.annotate("Parede da arena\n(4×4 m)",
            xy=(ARENA, 0.0),
            xytext=(1.2, -2.2),
            color=LABEL_COLOR, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#555577", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#7777aa", lw=1.2))

# 6. obs SAC (downsampled)
idx24 = np.linspace(0, 359, 24, dtype=int)
for i in idx24:
    ang = angles[i]
    hx2, hy2, _ = ray_hit(ROBOT_X, ROBOT_Y, ang, OBSTACLES)
    ax.plot([ROBOT_X, hx2], [ROBOT_Y, hy2],
            color="#ff4444", alpha=0.85, lw=2.0, zorder=4)

ax.annotate("24 raios usados na obs SAC\n(downsampled uniformemente)",
            xy=(ROBOT_X - 0.8, ROBOT_Y - 1.5),
            xytext=(-2.3, -2.0),
            color=LABEL_COLOR, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#001133", ec="#ff4444", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#ff6666", lw=1.2,
                           connectionstyle="arc3,rad=0.2"))

# ── Decoração ────────────────────────────────────────────────────────────────
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.tick_params(colors="white", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#444466")

ax.set_xlabel("x (m)", color="white", fontsize=10)
ax.set_ylabel("y (m)", color="white", fontsize=10)

ax.set_title(
    "Gazebo Classic — TurtleBot3 Waffle em Treino SAC\n"
    "Visão top-down: LIDAR 360° (azul = todos os raios, vermelho = obs SAC)",
    color="white", fontsize=11, fontweight="bold", pad=12
)

# Legenda
legend_elements = [
    mpatches.Patch(color="#4488ff", alpha=0.7, label="Raios LIDAR 360° (Gazebo)"),
    mpatches.Patch(color="#ff4444", alpha=0.9, label="24 raios → obs SAC"),
    mpatches.Patch(color="#111111", label="Obstáculo cilíndrico"),
    mpatches.Patch(color="#2255aa", label="TurtleBot3 Waffle"),
]
leg = ax.legend(handles=legend_elements, loc="upper right",
                fontsize=8.5, framealpha=0.8,
                facecolor="#001133", labelcolor="white",
                edgecolor="#4488ff")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Salvo: {OUT}")
plt.close()
