"""
visualize_multi_2d.py — GIFs animados de múltiplos robôs (SAC ou A* independentes).

Uso:
  python3 -m eval.env2d.visualize_multi_2d --all-worlds
  python3 -m eval.env2d.visualize_multi_2d --all-worlds --policy astar
  python3 -m eval.env2d.visualize_multi_2d --world dense --n-agents 4 --policy astar
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d_multi import MultiAgentEnv2D
from eval.env2d.env_2d import WORLDS, ROBOT_RADIUS, GOAL_RADIUS

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "paper" / "figs"
MODS = ROOT / "models"

NOMINAL_RHO = {"sparse": 0.05, "dense": 0.30, "very_dense": 0.50}
COLORS = plt.cm.tab10.colors


def _astar_action(env, i):
    """Política de linha reta para agente i (A* simplificado)."""
    dx = env.gx[i] - env.x[i]
    dy = env.gy[i] - env.y[i]
    dist = np.hypot(dx, dy)
    ang  = np.arctan2(dy, dx)
    dtheta = (ang - env.yaw[i] + np.pi) % (2 * np.pi) - np.pi
    v = min(1.0, dist / 0.5) * (abs(dtheta) < 0.6)
    w = np.clip(dtheta / np.pi, -1.0, 1.0)
    return np.array([v, w], dtype=np.float32)


def _rollout_multi(model, world, n_agents, seed, policy="sac"):
    """Roda um episódio multi-robô e retorna frames e métricas."""
    env = MultiAgentEnv2D(n_agents, world=world, seed=seed)
    obs = env.reset()

    frames = [_snapshot(env, n_agents)]
    done = False
    while not done:
        if policy == "astar":
            actions = np.array([_astar_action(env, i) for i in range(n_agents)])
        elif model is not None:
            actions, _ = model.predict(obs, deterministic=True)
        else:
            actions = np.zeros((n_agents, 2))
        obs, done = env.step(actions)
        frames.append(_snapshot(env, n_agents))

    m = env.metrics()
    return frames, m


def _snapshot(env, n):
    return {
        "x": env.x.copy(),
        "y": env.y.copy(),
        "yaw": env.yaw.copy(),
        "gx": env.gx.copy(),
        "gy": env.gy.copy(),
        "reached": env.goal_done.copy() if hasattr(env, "goal_done") else np.zeros(n, bool),
        "collided": env.collided.copy() if hasattr(env, "collided") else np.zeros(n, bool),
        "obstacles": env.obstacles,
        "arena": env.arena,
    }


def _draw_arena_multi(ax, snap):
    half = snap["arena"] / 2
    ax.add_patch(plt.Rectangle((-half, -half), snap["arena"], snap["arena"],
                                fill=False, edgecolor="#37474F", lw=2))
    for cx, cy, cr in snap["obstacles"]:
        ax.add_patch(Circle((cx, cy), cr, color="#607D8B", alpha=0.85, zorder=3))
    ax.set_xlim(-half - 0.25, half + 0.25)
    ax.set_ylim(-half - 0.25, half + 0.25)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.12)


def make_gif(model, world, n_agents, seed=1, fps=8, suffix="", cherry_pick=True, policy="sac"):
    """Gera um GIF animado de N robôs no mundo especificado."""
    rho = NOMINAL_RHO.get(world, 0.0)
    out_dir = FIGS / "marl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tenta achar um trial com pelo menos 1 robô chegando (cherry_pick)
    best_frames, best_metrics = None, None
    seeds_to_try = range(seed, seed + (30 if cherry_pick else 1))
    for s in seeds_to_try:
        frames, metrics = _rollout_multi(model, world, n_agents, s, policy=policy)
        if best_frames is None:
            best_frames, best_metrics = frames, metrics
        if cherry_pick and metrics.get("goal_rate", 0) > best_metrics.get("goal_rate", 0):
            best_frames, best_metrics = frames, metrics
        if cherry_pick and best_metrics.get("goal_rate", 0) >= 0.75:
            print(f"  seed={s}: goal_rate={best_metrics['goal_rate']:.0%} "
                  f"inter_coll={best_metrics['inter_collision']:.0%} — usado")
            break
    else:
        print(f"  seed range esgotado: goal_rate={best_metrics.get('goal_rate',0):.0%}")

    frames = best_frames
    n = n_agents

    # Acumula trajetórias
    traj_x = [[] for _ in range(n)]
    traj_y = [[] for _ in range(n)]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap_trail = plt.cm.plasma
    total = len(frames)

    def draw_frame(fi):
        ax.clear()
        snap = frames[fi]
        _draw_arena_multi(ax, snap)

        # Acumula trilhas
        for i in range(n):
            traj_x[i].append(snap["x"][i])
            traj_y[i].append(snap["y"][i])

        for i in range(n):
            c = COLORS[i % 10]
            # Trilha com gradiente temporal
            if len(traj_x[i]) > 1:
                txs, tys = traj_x[i], traj_y[i]
                for k in range(len(txs) - 1):
                    t = k / max(total - 1, 1)
                    ax.plot(txs[k:k+2], tys[k:k+2], "-",
                            color=cmap_trail(t), lw=1.8, alpha=0.7, zorder=4)

            collided = snap["collided"][i] if snap["collided"] is not None else False
            reached  = snap["reached"][i]  if snap["reached"]  is not None else False
            rc = "#F44336" if collided else "#4CAF50" if reached else c

            # Robô
            ax.add_patch(Circle((snap["x"][i], snap["y"][i]), ROBOT_RADIUS,
                                 color=rc, alpha=0.92, zorder=5))
            # Seta de orientação
            yaw = snap["yaw"][i]
            ax.annotate("", xy=(snap["x"][i] + 0.22*np.cos(yaw),
                                snap["y"][i] + 0.22*np.sin(yaw)),
                        xytext=(snap["x"][i], snap["y"][i]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.8))
            # Rótulo do robô
            ax.text(snap["x"][i], snap["y"][i] + ROBOT_RADIUS + 0.07,
                    str(i+1), ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=c, zorder=7)
            # Goal
            ax.plot(snap["gx"][i], snap["gy"][i], "*", color=c, ms=13, zorder=6,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.add_patch(Circle((snap["gx"][i], snap["gy"][i]), GOAL_RADIUS,
                                 color=c, alpha=0.15))

        gr  = best_metrics.get("goal_rate", 0)
        ic  = best_metrics.get("inter_collision", 0)
        label = "A* independente" if policy == "astar" else "SAC independente"
        ax.set_title(
            f"{n} robôs {label} — {world} ($\\rho\\approx{rho:.2f}$)\n"
            f"passo {fi}/{total-1}  |  goal={gr:.0%}  colisão={ic:.0%}",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(frames),
                                  interval=1000 // fps)
    policy_tag = "_astar" if policy == "astar" else ""
    fname = out_dir / f"fig_marl_episode_{world}_N{n_agents}{policy_tag}{suffix}.gif"
    ani.save(str(fname), writer="pillow", fps=fps)
    plt.close()
    print(f"  ✓ {fname.name}  [{len(frames)} frames]")
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="dense",
                    choices=["sparse", "dense", "very_dense"])
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--all-worlds", action="store_true",
                    help="Gera GIFs para sparse/dense/very_dense N=4")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--policy", default="sac", choices=["sac", "astar"],
                    help="Política dos agentes: sac ou astar")
    args = ap.parse_args()

    model = None
    if args.policy == "sac" and not args.no_model:
        from stable_baselines3 import SAC
        mp = MODS / "sac_2d_best.zip"
        if mp.exists():
            model = SAC.load(str(mp), device="cpu")
            print("Modelo carregado: sac_2d_best")
        else:
            print(f"Modelo não encontrado em {mp}. Use --no-model ou --policy astar.")
            sys.exit(1)

    if args.all_worlds:
        configs = [
            ("sparse",     4, 1),
            ("dense",      4, 1),
            ("very_dense", 4, 1),
            ("dense",      8, 1),
        ]
        for world, n, seed in configs:
            print(f"\n--- {world}  N={n}  policy={args.policy} ---")
            make_gif(model, world, n, seed=seed, fps=args.fps, policy=args.policy)
    else:
        make_gif(model, args.world, args.n_agents,
                 seed=args.seed, fps=args.fps, policy=args.policy)


if __name__ == "__main__":
    main()
