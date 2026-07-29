"""
plot_compare_mixed_pool.py — Regenera fig_2d_compare_dense.png/.pdf usando
o MESMO protocolo de amostragem do pool misto (rerun_h1_mixed.py: mundo
sorteado por trial, rng seed=42, seed0=5000), para que a % de sucesso
mostrada nesta figura venha da mesma fonte dos números 88,2%/84,3% citados
no slide seguinte ("Do plano à tese: por que um seletor..."). Antes, essa
figura rodava só no mundo "dense" fixo (n=30), o que dava uma % diferente
(90%/87%) da citada no slide vizinho — números próximos mas de amostras
distintas.

Uso:
    python3 -m eval.env2d.plot_compare_mixed_pool --n_eps 30
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

WORLDS_LIST = ["sparse", "dense", "very_dense"]
SEED0 = 5000  # mesmo seed0 de rerun_h1_mixed.py


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


def _trial_worlds(n_eps):
    """Reproduz exatamente a sequência de mundos de rerun_h1_mixed.py."""
    rng = np.random.default_rng(42)
    return [str(rng.choice(WORLDS_LIST)) for _ in range(n_eps)]


def _rollout(env, method, astar_policy, bc_policy):
    obs = env._obs()
    traj = [(env._x, env._y)]
    done, steps = False, 0
    use_astar = method == "astar" or (method == "adaptive" and local_rho(env) < RHO_STAR)
    if use_astar:
        astar_policy.reset(env)
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


def plot_compare_mixed(n_eps=30):
    astar_policy = AStarPolicy()
    bc_policies = {w: load_bc(w) for w in WORLDS_LIST}
    trial_worlds = _trial_worlds(n_eps)

    # arena de fundo: usa o mundo mais frequente na amostra, só para desenhar
    # os obstáculos de referência (cada trial pode ter um mundo distinto,
    # mas a maioria dos trials na densidade 30 cai em dense/very_dense)
    bg_world = max(set(trial_worlds), key=trial_worlds.count)

    titles = ["A* (planejador clássico)", "BC (política aprendida)", "Adaptativo ($\\rho$-criterion)"]
    methods = ["astar", "bc", "adaptive"]
    colors = ["#1565C0", "#C2185B", "#2E7D32"]
    slugs = ["astar", "bc", "adaptive"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))

    for panel_idx, (title, method, color, slug) in enumerate(zip(titles, methods, colors, slugs)):
        fig_solo, ax_solo = plt.subplots(figsize=(7.5, 7.5))
        for ax in (axes[panel_idx], ax_solo):
            _draw_arena(ax, bg_world)

        successes = []
        for i, world in enumerate(trial_worlds):
            seed = SEED0 + i
            env = Env2D(world=world, seed=seed)
            env.reset(seed=seed)
            traj, goal, gx, gy = _rollout(env, method, astar_policy, bc_policies[world])
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            successes.append(goal)
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
        title_txt = f"{title}\nSucesso: {sr:.0%} (mundos mistos, {n_eps} eps)"

        axes[panel_idx].set_title(title_txt, fontsize=15, fontweight="bold")
        axes[panel_idx].set_xlabel("x (m)", fontsize=12)
        axes[panel_idx].set_ylabel("y (m)", fontsize=12)
        axes[panel_idx].tick_params(labelsize=11)
        axes[panel_idx].legend(loc="lower right", fontsize=8, framealpha=0.9)

        title_solo = (f"{title} -- pool misto (sparse/dense/very\\_dense)\n"
                       f"Sucesso: {sr:.0%} ({n_eps} eps)")
        ax_solo.set_title(title_solo, fontsize=17, fontweight="bold")
        ax_solo.set_xlabel("x (m)", fontsize=15)
        ax_solo.set_ylabel("y (m)", fontsize=15)
        ax_solo.tick_params(labelsize=14)
        ax_solo.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax_solo.text(0.02, 0.02,
                     f"{n_eps} episódios independentes, mundo sorteado por trial\n"
                     f"(mesmo protocolo de rerun_h1_mixed.py, seed0={SEED0})",
                     transform=ax_solo.transAxes, fontsize=9, color="#555555",
                     verticalalignment="bottom", style="italic")
        fig_solo.tight_layout()
        for ext in ["png", "pdf"]:
            path = os.path.join(FIGS2D, f"fig_2d_compare_dense_{slug}.{ext}")
            fig_solo.savefig(path, dpi=170 if ext == "png" else None, bbox_inches="tight")
            print("Salvo:", path)
        plt.close(fig_solo)

    fig.suptitle(f"Comparação de estratégias com planejadores reais, pool misto de densidades "
                 f"(sparse/dense/very\\_dense, {n_eps} trials)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS2D, f"fig_2d_compare_dense.{ext}")
        fig.savefig(path, dpi=170 if ext == "png" else None, bbox_inches="tight")
        print("Salvo:", path)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_eps", type=int, default=30)
    args = p.parse_args()
    plot_compare_mixed(args.n_eps)
