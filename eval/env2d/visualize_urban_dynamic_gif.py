"""
visualize_urban_dynamic_gif.py — GIF animado do ρ-criterion navegando no
cenário urbano (urban_grid) com múltiplos obstáculos dinâmicos.

Diferente de visualize_bc_gif.py (_draw_arena não desenha quarteirões nem
obstáculos móveis, só círculos estáticos), este script desenha os blocks
(quarteirões sólidos) e anima a posição dos obstáculos dinâmicos quadro a
quadro — necessário para o urban_grid, que usa blocks/walls, não circles.

Fecha a lacuna "múltiplos obstáculos dinâmicos" (sessão 25/07/2026,
docs/PLANO_CORRECAO.md, objetivo 2 do Plano de Trabalho PI08078-2024).

Uso:
    python3 eval/env2d/visualize_urban_dynamic_gif.py --condition dynamic_multi
"""
import os, sys, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d import Env2D, WORLDS, ROBOT_RADIUS, GOAL_RADIUS
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc, RHO_STAR
from eval.env2d.rerun_h1_mixed import local_rho
from eval.env2d.rerun_urban import DYNAMIC_MULTI_SPEC, DYNAMIC_FAST_SPEC, DYNAMIC_OBSTACLE_SPEC

FIGS2D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "paper", "figs", "2d")

# Variante SÓ PARA ILUSTRAÇÃO (não usada nos 2.000 trials estatísticos da
# Seção 3.6, que continuam rodando com DYNAMIC_MULTI_SPEC original). Aqui os
# 3 obstáculos nascem afastados, em ângulos diferentes, e convergem para o
# cruzamento central do urban_grid (onde o corredor em "+" força a passagem
# do robô), tornando mais provável e mais visível que os três apareçam perto
# do robô no mesmo episódio -- sem alterar a dinâmica real do experimento.
DYNAMIC_MULTI_CONVERGING_SPEC = [
    {"cx": -1.6, "cy": 0.3, "cr": 0.15, "vx": 0.55, "vy": -0.12},   # oeste -> centro
    {"cx": 0.3, "cy": -1.6, "cr": 0.15, "vx": -0.12, "vy": 0.55},   # sul -> centro
    {"cx": 1.5, "cy": 1.0, "cr": 0.12, "vx": -0.55, "vy": -0.42},   # nordeste -> centro
]

CONDITIONS = {
    "dynamic": DYNAMIC_OBSTACLE_SPEC,
    "dynamic_multi": DYNAMIC_MULTI_SPEC,
    "dynamic_fast": DYNAMIC_FAST_SPEC,
    "dynamic_multi_converging": DYNAMIC_MULTI_CONVERGING_SPEC,
}


def _draw_urban_arena(ax, world="urban_grid"):
    cfg = WORLDS[world]
    half = cfg["size"] / 2
    rect = plt.Rectangle((-half, -half), cfg["size"], cfg["size"],
                          fill=False, edgecolor="#37474F", lw=2)
    ax.add_patch(rect)
    for xmin, ymin, xmax, ymax in cfg.get("blocks", []):
        block = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor="#78909C", edgecolor="#37474F", lw=1, zorder=3)
        ax.add_patch(block)
    ax.set_xlim(-half - 0.2, half + 0.2)
    ax.set_ylim(-half - 0.2, half + 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _rollout_adaptive(dyn_spec, seed):
    """Roda o ρ-criterion (decisão per-episódio no reset, igual a
    rerun_h1_mixed.py::run_one) e retorna trajetória do robô + trajetória
    de cada obstáculo dinâmico, quadro a quadro."""
    env = Env2D(world="urban_grid", seed=seed, dynamic_obstacles=dyn_spec)
    astar_policy = AStarPolicy()
    bc_policy = load_bc("urban_grid")

    obs, _ = env.reset()
    rho0 = local_rho(env)
    use_astar = rho0 < RHO_STAR
    if use_astar:
        astar_policy.reset(env)

    frames = []
    done = goal = coll = False
    while not done:
        dyn_positions = [(o["cx"], o["cy"], o["cr"]) for o in env._dyn_state]
        frames.append((env._x, env._y, env._yaw, env._gx, env._gy, goal, coll, dyn_positions))
        if use_astar:
            a = astar_policy.act(env)
        else:
            import torch
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        goal = goal or info.get("goal_reached", False)
        coll = coll or info.get("collision", False)
        done = term or trunc
    dyn_positions = [(o["cx"], o["cy"], o["cr"]) for o in env._dyn_state]
    frames.append((env._x, env._y, env._yaw, env._gx, env._gy, goal, coll, dyn_positions))
    outcome = "goal" if goal else "collision" if coll else "timeout"
    return frames, outcome, use_astar


def _min_robot_obstacle_dist(frames):
    """Menor distância robô-obstáculo ao longo do episódio."""
    best = float("inf")
    for x, y, yaw, gx, gy, goal, coll, dyn_positions in frames:
        for ox, oy, orr in dyn_positions:
            d = np.hypot(x - ox, y - oy) - orr
            best = min(best, d)
    return best


def _deviation_score(frames, close_thresh=0.6, window=4):
    """Mede se o robô de fato DESVIOU por causa de um obstáculo, não só
    passou perto por coincidência de geometria. Para cada frame em que
    algum obstáculo está a menos de close_thresh do robô, compara a
    direção de deslocamento `window` passos antes com `window` passos
    depois e soma a variação angular -- desvio real produz uma curva
    visível nesse intervalo; passar reto perto do obstáculo (sem reagir)
    não. Retorna (score, dist_min): quanto maior o score, mais visível o
    desvio; usado para não escolher episódios "mansos" onde o obstáculo
    fica perto mas o robô não muda de rumo por causa dele."""
    xs = np.array([f[0] for f in frames])
    ys = np.array([f[1] for f in frames])
    n = len(frames)

    dist_min = float("inf")
    close_idxs = []
    for i, (x, y, yaw, gx, gy, goal, coll, dyn_positions) in enumerate(frames):
        for ox, oy, orr in dyn_positions:
            d = np.hypot(x - ox, y - oy) - orr
            dist_min = min(dist_min, d)
            if d < close_thresh:
                close_idxs.append(i)

    if not close_idxs:
        return 0.0, dist_min

    best_score = 0.0
    for i in close_idxs:
        i0, i1 = max(0, i - window), min(n - 1, i + window)
        if i1 - i <= 1 or i - i0 <= 1:
            continue
        dir_before = np.arctan2(ys[i] - ys[i0], xs[i] - xs[i0])
        dir_after = np.arctan2(ys[i1] - ys[i], xs[i1] - xs[i])
        turn = abs((dir_after - dir_before + np.pi) % (2 * np.pi) - np.pi)
        best_score = max(best_score, turn)
    return best_score, dist_min


def make_gif(condition="dynamic_multi", seed=0, fps=10, max_tries=300, min_frames=25,
             n_candidates=40):
    dyn_spec = CONDITIONS[condition]
    candidates = []  # (frames, outcome, used_astar, seed, score, min_dist)
    for try_seed in range(seed, seed + max_tries):
        cand, oc, ua = _rollout_adaptive(dyn_spec, try_seed)
        if oc == "goal" and len(cand) >= min_frames:
            score, min_dist = _deviation_score(cand)
            candidates.append((cand, oc, ua, try_seed, score, min_dist))
            if len(candidates) >= n_candidates:
                break

    if candidates:
        # escolhe o episódio com o desvio mais visível (maior mudança de
        # rumo do robô perto de um obstáculo), não só o de menor distância
        # -- um robô que passa perto sem reagir (linha reta) não ilustra
        # o mecanismo de desvio, mesmo estando fisicamente próximo
        frames_data, outcome, used_astar, best_seed, score, min_dist = max(candidates, key=lambda c: c[4])
        print(f"  Episódio vencedor: seed={best_seed} ({len(frames_data)} frames, "
              f"planejador={'A*' if used_astar else 'BC'}, "
              f"desvio={np.degrees(score):.0f}°, dist. mín.={min_dist:.2f}m, "
              f"de {len(candidates)} candidatos)")
    else:
        print(f"  Aviso: nenhum vencedor em {max_tries} seeds, usando seed={seed}")
        frames_data, outcome, used_astar = _rollout_adaptive(dyn_spec, seed)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap_trail = plt.cm.plasma
    total = len(frames_data)
    n_obstacles = len(dyn_spec)
    obs_colors = ["#E91E63", "#9C27B0", "#FF9800"][:n_obstacles]

    def draw_frame(i):
        ax.clear()
        _draw_urban_arena(ax)
        xs = [f[0] for f in frames_data[:i+1]]
        ys = [f[1] for f in frames_data[:i+1]]
        if len(xs) > 1:
            for k in range(len(xs) - 1):
                t = k / max(total - 1, 1)
                ax.plot(xs[k:k+2], ys[k:k+2], "-", color=cmap_trail(t),
                        lw=2.2, alpha=0.85, zorder=5)
        x, y, yaw, gx, gy, goal, coll, dyn_positions = frames_data[i]

        for j, (ox, oy, orr) in enumerate(dyn_positions):
            ax.add_patch(Circle((ox, oy), orr, color=obs_colors[j % len(obs_colors)],
                                 alpha=0.85, zorder=6))

        color_robot = "#F44336" if coll else "#4CAF50" if goal else "#2196F3"
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, color=color_robot, zorder=7, alpha=0.92))
        ax.annotate("", xy=(x + 0.25*np.cos(yaw), y + 0.25*np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.2))
        ax.plot(gx, gy, "*", color="#FF5722", ms=16, zorder=8,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#FF5722", alpha=0.18))

        last = (i == total - 1)
        status = ("✓ GOAL!" if goal else "✗ COLISÃO" if coll
                  else ("✗ TIMEOUT" if last else f"passo {i}/{total-1}"))
        planner_used = "A*" if used_astar else "BC"
        ax.set_title(f"ρ-criterion ({planner_used}) — urban_grid, {n_obstacles} obstáculo(s) "
                     f"dinâmico(s)\n{status}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ani = animation.FuncAnimation(fig, draw_frame, frames=total, interval=1000 // fps)
    gif_path = os.path.join(FIGS2D, f"fig_2d_urban_{condition}_episode.gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  ✓ 2d/fig_2d_urban_{condition}_episode.gif [{outcome}] ({len(frames_data)} frames)")
    make_static(frames_data, condition, outcome, used_astar, n_obstacles, obs_colors)
    return gif_path


def make_static(frames_data, condition, outcome, used_astar, n_obstacles, obs_colors):
    """Versão estática (PNG/PDF) do mesmo episódio -- necessária porque o
    relatório final vira PDF na submissão SIGAA, e GIFs não renderizam em
    PDF. Mostra a trajetória completa do robô (linha grossa e escura, sem
    concorrência visual) e, para cada obstáculo dinâmico, só a posição
    inicial (contorno tracejado) e final (cor sólida) ligadas por uma seta
    de sentido -- em vez do rastro de ~8 círculos translúcidos sobrepostos
    da versão anterior, que ficava ilegível com 3 obstáculos."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_urban_arena(ax)
    xs = [f[0] for f in frames_data]
    ys = [f[1] for f in frames_data]
    n = len(xs)
    ax.plot(xs, ys, "-", color="#1A237E", lw=3.2, alpha=0.95, zorder=5,
            solid_capstyle="round", label="Trajetória do robô")
    # marca alguns pontos ao longo do caminho para dar noção de progressão
    # temporal sem competir visualmente com a linha
    for i in range(0, n, max(1, n // 6)):
        ax.plot(xs[i], ys[i], "o", color="#1A237E", ms=4, zorder=5, alpha=0.6)

    for j in range(n_obstacles):
        ox0, oy0, orr = frames_data[0][7][j]
        oxf, oyf, _ = frames_data[-1][7][j]
        color = obs_colors[j % len(obs_colors)]
        # rótulo só no primeiro obstáculo, para a legenda não repetir
        # "Trajetória de obstáculo móvel" 3x (uma por obstáculo)
        lbl = "Trajetória de obstáculo móvel" if j == 0 else None
        ax.add_patch(Circle((ox0, oy0), orr, facecolor="none", edgecolor=color,
                             linestyle="--", lw=1.8, zorder=4))
        ax.add_patch(Circle((oxf, oyf), orr, facecolor=color, edgecolor="white",
                             alpha=0.9, lw=1, zorder=4))
        ax.plot([], [], "-", color="#9E9E9E", lw=1.8, label=lbl)
        ax.annotate("", xy=(oxf, oyf), xytext=(ox0, oy0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8,
                                    alpha=0.85, shrinkA=8, shrinkB=8),
                    zorder=4)
        # numera cada obstáculo diretamente no início da seta, para não
        # depender só de cor para diferenciar qual é qual
        ax.annotate(f"obs.{j+1}", (ox0, oy0), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color=color, fontweight="bold",
                    zorder=7)

    ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=12, zorder=6,
            label="Início do robô", markeredgecolor="white", markeredgewidth=1.5)
    gx, gy = frames_data[-1][3], frames_data[-1][4]
    ax.plot(gx, gy, "*", color="#F44336", ms=16, zorder=6,
            label="Goal", markeredgecolor="white", markeredgewidth=1)
    ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#F44336", alpha=0.15, zorder=2))

    status = "✓ Goal" if outcome == "goal" else "✗ Colisão" if outcome == "collision" else "Timeout"
    planner_used = "A*" if used_astar else "BC"
    ax.set_title(f"Trajetória ρ-criterion ({planner_used}) -- urban_grid, "
                 f"{n_obstacles} obstáculo(s) dinâmico(s)\n{n} passos | {status} | "
                 "tracejado = início do obstáculo, cheio = fim, seta = sentido do obstáculo",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS2D, f"fig_2d_urban_{condition}_trajectory.{ext}")
        plt.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_2d_urban_{condition}_trajectory.png/pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--condition", default="dynamic_multi", choices=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    make_gif(args.condition, args.seed)
