"""
plot_compare_dense_real.py — Regenera fig_2d_compare_dense.png/.pdf com
planejadores REAIS, substituindo o A* simulado por linha reta que
visualize_2d.py::plot_compare() usava (mesmo bug de "A* falso" já
corrigido em outros lugares do projeto pela auditoria de 27/07/2026 —
ver eval/env2d/astar_planner.py e rerun_h1_real.py).

Também melhora a legibilidade: figura maior, menos episódios sobrepostos
por painel (3 em vez de 5), trajetórias mais grossas, fonte maior.

Uso:
    python3 -m eval.env2d.plot_compare_dense_real --world dense
"""
import os, sys, argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d import Env2D, WORLDS, GOAL_RADIUS
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc, local_rho, RHO_STAR

FIGS2D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "paper", "figs", "2d")


def _draw_arena(ax, world):
    cfg = WORLDS[world]
    half = cfg["size"] / 2
    rect = plt.Rectangle((-half, -half), cfg["size"], cfg["size"],
                          fill=False, edgecolor="#37474F", lw=2)
    ax.add_patch(rect)
    for cx, cy, cr in cfg.get("obstacles", []):
        ax.add_patch(Circle((cx, cy), cr, color="#607D8B", alpha=0.85, zorder=3))
    ax.set_xlim(-half - 0.2, half + 0.2)
    ax.set_ylim(-half - 0.2, half + 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _rollout(env, method, astar_policy, bc_policy):
    if method in ("astar", "adaptive") and env_uses_astar(method, env):
        astar_policy.reset(env)
    obs = env._obs()
    traj = [(env._x, env._y)]
    done, steps = False, 0
    use_astar = method == "astar" or (method == "adaptive" and local_rho(env) < RHO_STAR)
    while not done and steps < 200:
        if use_astar:
            a = astar_policy.act(env)
        else:
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        traj.append((env._x, env._y))
        done = term or trunc
        steps += 1
    return traj, bool(info.get("goal_reached", False)), env._gx, env._gy


def env_uses_astar(method, env):
    return method == "astar" or (method == "adaptive" and local_rho(env) < RHO_STAR)


def plot_compare(world="dense", n_eps=3, seed0=0, seeds=None):
    astar_policy = AStarPolicy()
    bc_policy = load_bc(world)

    # seeds explícitas têm prioridade sobre seed0+i sequencial; escolhidas
    # por dispersão espacial dos pares início/goal (greedy sobre distância
    # euclidiana mínima), em vez de sequenciais, porque seed0=0 dava pontos
    # aglomerados no centro da arena, difíceis de distinguir visualmente
    if seeds is not None:
        n_eps = len(seeds)
    else:
        seeds = [seed0 + i for i in range(n_eps)]

    titles = ["A* (planejador clássico)", "BC (política aprendida)", "Adaptativo ($\\rho$-criterion)"]
    methods = ["astar", "bc", "adaptive"]
    colors = ["#1565C0", "#C2185B", "#2E7D32"]
    slugs = ["astar", "bc", "adaptive"]

    # ρ_local médio das seeds usadas (mesma métrica do ρ-criterion); reportar
    # a média em vez de uma amostra isolada, já que varia por seed conforme
    # onde o robô nasce dentro do mundo "dense"
    rho_samples = []
    for seed in seeds:
        env = Env2D(world=world, seed=seed)
        env.reset()
        rho_samples.append(local_rho(env))
    rho_mean = np.mean(rho_samples)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    for panel_idx, (title, method, color, slug) in enumerate(zip(titles, methods, colors, slugs)):
        # gera cada painel duas vezes: uma vez no eixo do combinado (para o
        # relatório, página cheia) e uma vez em figura própria dedicada,
        # maior e com fonte proporcionalmente maior — necessário porque o
        # combinado de 3 painéis fica ilegível reduzido ao tamanho de slide
        fig_solo, ax_solo = plt.subplots(figsize=(7.5, 7.5))
        for ax in (axes[panel_idx], ax_solo):
            _draw_arena(ax, world)

        successes = []
        for i, seed in enumerate(seeds):
            env = Env2D(world=world, seed=seed)
            env.reset()
            traj, goal, gx, gy = _rollout(env, method, astar_policy, bc_policy)
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            successes.append(goal)
            # rótulo só na primeira trajetória de cada painel, para a
            # legenda não repetir "Início"/"Goal" 6 vezes (1 por episódio)
            lbl_start = "Início do episódio" if i == 0 else None
            lbl_goal = "Goal do episódio" if i == 0 else None
            for ax, lw, ms_start, ms_goal in ((axes[panel_idx], 2.6, 11, 16), (ax_solo, 3.4, 14, 20)):
                ax.plot(xs, ys, color=color, lw=lw, alpha=0.8, zorder=5)
                ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=ms_start, zorder=6,
                        markeredgecolor="white", markeredgewidth=1.3, label=lbl_start)
                ax.plot(gx, gy, "*", color="#F44336", ms=ms_goal, zorder=6,
                        markeredgecolor="white", markeredgewidth=1, label=lbl_goal)
                ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#F44336", alpha=0.12, zorder=2))

        sr = np.mean(successes)
        # Sem % de sucesso no título: com n_eps pequeno (amostra ilustrativa
        # de trajetórias, não estatística) a taxa varia por ruído de seed e
        # pode contradizer visualmente o número real (n=1.500, pool misto de
        # densidades) citado no slide seguinte. A % oficial fica só lá.
        title_txt = f"{title}\n({n_eps} episódios ilustrativos)"

        axes[panel_idx].set_title(title_txt, fontsize=15, fontweight="bold")
        axes[panel_idx].set_xlabel("x (m)", fontsize=12)
        axes[panel_idx].set_ylabel("y (m)", fontsize=12)
        axes[panel_idx].tick_params(labelsize=11)
        axes[panel_idx].legend(loc="lower right", fontsize=8, framealpha=0.9)

        title_solo = (f"{title} -- env 2D ({world}, $\\rho_{{local}}\\approx{rho_mean:.2f}$)\n"
                       f"{n_eps} episódios ilustrativos")
        ax_solo.set_title(title_solo, fontsize=17, fontweight="bold")
        ax_solo.set_xlabel("x (m)", fontsize=15)
        ax_solo.set_ylabel("y (m)", fontsize=15)
        ax_solo.tick_params(labelsize=14)
        ax_solo.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax_solo.text(0.02, 0.02,
                     f"{n_eps} episódios independentes\n(cada 1 com seu par início/goal)",
                     transform=ax_solo.transAxes, fontsize=9, color="#555555",
                     verticalalignment="bottom", style="italic")
        fig_solo.tight_layout()
        for ext in ["png", "pdf"]:
            path = os.path.join(FIGS2D, f"fig_2d_compare_{world}_{slug}.{ext}")
            fig_solo.savefig(path, dpi=170 if ext == "png" else None, bbox_inches="tight")
            print("Salvo:", path)
        plt.close(fig_solo)

    fig.suptitle(f"Comparação de estratégias com planejadores reais, env 2D "
                 f"({world}, $\\rho_{{local}}$ médio $\\approx{rho_mean:.2f}$)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS2D, f"fig_2d_compare_{world}.{ext}")
        fig.savefig(path, dpi=170 if ext == "png" else None, bbox_inches="tight")
        print("Salvo:", path)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="dense", choices=["sparse", "dense", "very_dense"])
    p.add_argument("--n_eps", type=int, default=3)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="lista explícita de seeds (sobrepõe --seed0/--n_eps)")
    args = p.parse_args()
    plot_compare(args.world, args.n_eps, args.seed0, args.seeds)
