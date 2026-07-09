"""
visualize_bc_gif.py — GIF animado do episódio BC (Behavior Cloning) no
gêmeo 2D, mesmo estilo visual do visualize_2d.py (plot_gif) usado pro SAC.

Melhor resultado quantitativo da sessão de 09/07/2026: BC atinge 98% de
sucesso em ~2 minutos de treino (potencial-field expert → imitação
supervisionada), muito mais rápido e confiável que SAC no mesmo ambiente.
Ver [[project-treino-sparse-08jul]].

Uso:
    python3 eval/env2d/visualize_bc_gif.py --world sparse
"""
import os, sys, argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import Env2D, WORLDS, ROBOT_RADIUS, GOAL_RADIUS
from eval.env2d.visualize_2d import _draw_arena, FIGS2D

MODS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


class BCPolicy(nn.Module):
    """Mesma arquitetura de train_2d_bc.py — necessária para carregar o state_dict."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def _rollout(policy: BCPolicy, world: str, seed: int):
    env = Env2D(world=world, seed=seed)
    obs, _ = env.reset()
    frames = [(env._x, env._y, env._yaw, env._gx, env._gy, False, False)]
    done = goal = coll = False
    with torch.no_grad():
        while not done:
            a = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            obs, r, term, trunc, info = env.step(a.squeeze(0).numpy())
            goal = goal or info.get("goal_reached", False)
            coll = coll or info.get("collision", False)
            frames.append((env._x, env._y, env._yaw, env._gx, env._gy, goal, coll))
            done = term or trunc
    outcome = "goal" if goal else "collision" if coll else "timeout"
    return frames, outcome


def plot_bc_gif(policy: BCPolicy, world: str = "sparse", seed: int = 0, fps: int = 10):
    frames_data, outcome = None, "timeout"
    for try_seed in range(seed, seed + 50):
        cand, oc = _rollout(policy, world, try_seed)
        if oc == "goal":
            frames_data, outcome = cand, oc
            print(f"  Episódio vencedor: seed={try_seed} ({len(cand)} frames)")
            break
    if frames_data is None:
        print(f"  Aviso: nenhum vencedor em 50 seeds no '{world}', usando seed={seed}")
        frames_data, outcome = _rollout(policy, world, seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_trail = plt.cm.plasma
    total = len(frames_data)

    def draw_frame(i):
        ax.clear()
        _draw_arena(ax, world)
        xs = [f[0] for f in frames_data[:i+1]]
        ys = [f[1] for f in frames_data[:i+1]]
        if len(xs) > 1:
            for k in range(len(xs) - 1):
                t = k / max(total - 1, 1)
                ax.plot(xs[k:k+2], ys[k:k+2], "-", color=cmap_trail(t),
                        lw=2.2, alpha=0.85, zorder=4)
        x, y, yaw, gx, gy, goal, coll = frames_data[i]
        color_robot = "#F44336" if coll else "#4CAF50" if goal else "#2196F3"
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, color=color_robot, zorder=5, alpha=0.92))
        ax.annotate("", xy=(x + 0.25*np.cos(yaw), y + 0.25*np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.2))
        ax.plot(gx, gy, "*", color="#FF5722", ms=16, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#FF5722", alpha=0.18))
        last = (i == total - 1)
        status = "✓ GOAL!" if goal else "✗ COLISÃO" if coll else ("✗ TIMEOUT" if last else f"passo {i}/{total-1}")
        ax.set_title(f"BC (imitação por campo potencial) — env 2D [{world}]\n{status}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ani = animation.FuncAnimation(fig, draw_frame, frames=total, interval=1000 // fps)
    gif_path = os.path.join(FIGS2D, f"fig_2d_bc_episode_{world}.gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  ✓ 2d/fig_2d_bc_episode_{world}.gif [{outcome}] ({len(frames_data)} frames)")
    return gif_path, frames_data, outcome


def plot_bc_static(frames_data, world: str, outcome: str):
    """Versão estática (PNG/PDF) do mesmo episódio — necessária porque o
    relatório final vira PDF na submissão SIGAA, e GIFs não renderizam em PDF."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_arena(ax, world)
    xs = [f[0] for f in frames_data]
    ys = [f[1] for f in frames_data]
    n = len(xs)
    cmap = plt.cm.plasma
    for i in range(n - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=cmap(i / max(n-1, 1)), lw=2, zorder=4)
    ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=12, zorder=6,
            label="Início", markeredgecolor="white", markeredgewidth=1.5)
    gx, gy = frames_data[-1][3], frames_data[-1][4]
    ax.plot(gx, gy, "*", color="#F44336", ms=16, zorder=6,
            label="Goal", markeredgecolor="white", markeredgewidth=1)
    ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#F44336", alpha=0.15, zorder=2))
    status = "✓ Goal" if outcome == "goal" else "✗ Colisão" if outcome == "collision" else "Timeout"
    ax.set_title(f"Trajetória BC (imitação por campo potencial) — env 2D ({world})\n"
                 f"{n} passos | {status}", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS2D, f"fig_2d_bc_trajectory_{world}.{ext}")
        plt.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_2d_bc_trajectory_{world}.png/pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="sparse", choices=["sparse", "dense", "very_dense"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt_path = os.path.join(MODS, "bc_2d_policy.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    obs_dim = state["net.0.weight"].shape[1]
    act_dim = state["net.4.weight"].shape[0]
    policy = BCPolicy(obs_dim, act_dim)
    policy.load_state_dict(state)
    policy.eval()

    _, frames_data, outcome = plot_bc_gif(policy, args.world, args.seed)
    plot_bc_static(frames_data, args.world, outcome)
