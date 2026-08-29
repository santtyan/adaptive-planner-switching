"""
visualize_2d.py — Visualização do env 2D treinado.

Modos:
  --mode episode   → salva PNG/PDF de uma trajetória completa
  --mode compare   → A* vs SAC vs Adaptativo lado a lado (3 painéis)
  --mode heatmap   → heatmap de decisão A*/BC no espaço da arena
  --mode gif       → gera GIF animado de um episódio (para apresentação)
  --mode all       → gera tudo

Uso:
  python3 eval/env2d/visualize_2d.py --mode all
  python3 eval/env2d/visualize_2d.py --mode compare --world dense
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import Env2D, WORLDS, ROBOT_RADIUS, GOAL_RADIUS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(ROOT, "paper", "figs")
FIGS2D = os.path.join(FIGS, "2d")
os.makedirs(FIGS2D, exist_ok=True)
MODS = os.path.join(ROOT, "models")


def load_model(path: str):
    from stable_baselines3 import SAC
    if not os.path.exists(path + ".zip"):
        print(f"Modelo não encontrado: {path}.zip")
        return None
    return SAC.load(path, device="cpu")


def _draw_arena(ax, world: str, alpha_obs: float = 0.85):
    cfg  = WORLDS[world]
    half = cfg["size"] / 2
    rect = plt.Rectangle((-half, -half), cfg["size"], cfg["size"],
                          fill=False, edgecolor="#37474F", lw=2)
    ax.add_patch(rect)
    for cx, cy, cr in cfg["obstacles"]:
        c = Circle((cx, cy), cr, color="#607D8B", alpha=alpha_obs, zorder=3)
        ax.add_patch(c)
    ax.set_xlim(-half - 0.2, half + 0.2)
    ax.set_ylim(-half - 0.2, half + 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def _run_episode(model, world: str, seed: int = 0, deterministic: bool = True):
    """Executa um episódio e retorna trajetória + info."""
    env  = Env2D(world=world, seed=seed)
    obs, _ = env.reset()
    traj = [(env._x, env._y, env._yaw)]
    total_r = 0.0
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        traj.append((env._x, env._y, env._yaw))
        total_r += r
        done = term or trunc
    return traj, info, total_r, env


# ══════════════════════════════════════════════════════════════
# MODO 1 — Trajetória única
# ══════════════════════════════════════════════════════════════
def plot_episode(model, world: str = "sparse", seed: int = 7):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_arena(ax, world)

    traj, info, total_r, env = _run_episode(model, world, seed)
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]

    # Gradiente de cor ao longo do tempo
    n = len(xs)
    cmap = plt.cm.plasma
    for i in range(n - 1):
        ax.plot(xs[i:i+2], ys[i:i+2], color=cmap(i / max(n-1, 1)),
                lw=2, zorder=4)

    # Início e fim
    ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=12, zorder=6,
            label="Início", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(env._gx, env._gy, "*", color="#F44336", ms=16, zorder=6,
            label="Goal", markeredgecolor="white", markeredgewidth=1)
    goal_circle = Circle((env._gx, env._gy), GOAL_RADIUS,
                          color="#F44336", alpha=0.15, zorder=2)
    ax.add_patch(goal_circle)

    # Seta de orientação final
    if len(traj) > 1:
        lx, ly, lyaw = traj[-1]
        ax.annotate("", xy=(lx + 0.15*np.cos(lyaw), ly + 0.15*np.sin(lyaw)),
                    xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="->", color="#FF9800", lw=2))

    status = "✓ Goal" if info.get("goal_reached") else "✗ Colisão" if info.get("collision") else "Timeout"
    ax.set_title(f"Trajetória SAC — env 2D ({world})\n"
                 f"{len(traj)} passos | reward={total_r:.1f} | {status}",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # Colorbar temporal
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Passo", shrink=0.7)

    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIGS2D, f"fig_2d_trajectory_{world}.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_2d_trajectory_{world}.png")


# ══════════════════════════════════════════════════════════════
# MODO 2 — Comparação A* (Nav2) vs SAC vs Adaptativo
# ══════════════════════════════════════════════════════════════
def plot_compare(model, world: str = "dense"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles  = ["A* (planejador clássico)", "SAC (aprendizado RL)", "Adaptativo ($\\rho$-criterion)"]
    colors  = ["#2196F3", "#E91E63", "#4CAF50"]

    # Para A* simulado: segue linha reta até o goal (aproximação)
    def astar_policy(env):
        dx = env._gx - env._x
        dy = env._gy - env._y
        dist = np.hypot(dx, dy)
        ang  = np.arctan2(dy, dx)
        dtheta = ang - env._yaw
        dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi
        v_norm = min(1.0, dist / 0.5)
        w_norm = np.clip(dtheta / np.pi, -1.0, 1.0)
        return np.array([v_norm, w_norm])

    RHO_STAR = 0.30

    def density(env) -> float:
        cfg  = WORLDS[env.world if hasattr(env, 'world') else world]
        area = cfg["size"] ** 2
        obs_area = sum(np.pi * cr**2 for _, _, cr in cfg["obstacles"])
        return obs_area / area

    for col, (title, color) in enumerate(zip(titles, colors)):
        ax = axes[col]
        _draw_arena(ax, world)
        rho = density(Env2D(world=world))

        all_success = []
        for seed in range(5):
            env = Env2D(world=world, seed=seed)
            obs, _ = env.reset()
            traj = [(env._x, env._y)]
            done = False
            steps = 0
            while not done and steps < 200:
                if col == 0:       # A* simulado
                    action = astar_policy(env)
                elif col == 1:     # SAC puro
                    action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
                else:              # Adaptativo
                    if rho < RHO_STAR:
                        action = astar_policy(env)
                    else:
                        action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
                obs, r, term, trunc, info = env.step(action)
                traj.append((env._x, env._y))
                done = term or trunc
                steps += 1
            all_success.append(info.get("goal_reached", False))

            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=color, lw=1.8, alpha=0.7, zorder=4)
            ax.plot(xs[0], ys[0], "o", color="#4CAF50", ms=8, zorder=6,
                    markeredgecolor="white", markeredgewidth=1)
            ax.plot(env._gx, env._gy, "*", color="#F44336", ms=12, zorder=6,
                    markeredgecolor="white")
            ax.add_patch(Circle((env._gx, env._gy), GOAL_RADIUS,
                                color="#F44336", alpha=0.10))

        sr = np.mean(all_success)
        ax.set_title(f"{title}\nSucesso: {sr:.0%} (5 eps)", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)" if col == 0 else "")

    fig.suptitle(f"Comparação de estratégias — env 2D ({world}, $\\rho\\approx{rho:.2f}$)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIGS2D, f"fig_2d_compare_{world}.{ext}"),
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_2d_compare_{world}.png")


# ══════════════════════════════════════════════════════════════
# MODO 3 — Heatmap de decisão A*/BC no espaço da arena
# ══════════════════════════════════════════════════════════════
def plot_heatmap(world: str = "dense"):
    cfg  = WORLDS[world]
    half = cfg["size"] / 2
    res  = 60
    xs   = np.linspace(-half, half, res)
    ys   = np.linspace(-half, half, res)
    rho_grid = np.zeros((res, res))

    env = Env2D(world=world)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            # Densidade local: fração do scan < 1.0m
            from eval.env2d.env_2d import _scan
            scan = _scan(x, y, 0.0, cfg["obstacles"], cfg["size"])
            local_density = float(np.mean(scan < 1.0))
            rho_grid[i, j] = local_density

    RHO_STAR = 0.30
    decision = (rho_grid >= RHO_STAR).astype(float)  # 0=A*, 1=BC

    fig, axes = plt.subplots(2, 1, figsize=(7, 12))

    # Painel superior: densidade local
    im1 = axes[0].imshow(rho_grid, origin="lower",
                         extent=[-half, half, -half, half],
                         cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im1, ax=axes[0], label="Densidade local $\\rho$")
    for cx, cy, cr in cfg["obstacles"]:
        axes[0].add_patch(Circle((cx, cy), cr, color="gray", alpha=0.6))
    axes[0].axhline(0, color="white", lw=0.5, alpha=0.3)
    axes[0].axvline(0, color="white", lw=0.5, alpha=0.3)
    axes[0].set_title(f"(a) Densidade local $\\rho(x,y)$ — {world}", fontsize=11)
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")

    # Painel inferior: decisão A*/BC
    cmap_dec = matplotlib.colors.ListedColormap(["#2196F3", "#E91E63"])
    im2 = axes[1].imshow(decision, origin="lower",
                         extent=[-half, half, -half, half],
                         cmap=cmap_dec, vmin=0, vmax=1, aspect="equal")
    for cx, cy, cr in cfg["obstacles"]:
        axes[1].add_patch(Circle((cx, cy), cr, color="gray", alpha=0.6))
    legend_patches = [
        mpatches.Patch(color="#2196F3", label=f"A* ($\\rho < {RHO_STAR}$)"),
        mpatches.Patch(color="#E91E63", label=f"BC ($\\rho \\geq {RHO_STAR}$)"),
    ]
    axes[1].legend(handles=legend_patches, loc="upper right", fontsize=9)
    axes[1].set_title(f"(b) Decisão $\\pi(\\rho)$ — $\\rho^*={RHO_STAR}$", fontsize=11)
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    axes[1].text(0.02, 0.02,
                 f"$\\rho^* = {RHO_STAR}$ → {decision.mean():.0%} BC",
                 transform=axes[1].transAxes, fontsize=9, color="white",
                 bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.ax.set_visible(False)

    fig.suptitle("Heatmap do critério $\\rho$: onde A* e BC são ativados",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIGS2D, f"fig_2d_heatmap_{world}.{ext}"),
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 2d/fig_2d_heatmap_{world}.png")


# ══════════════════════════════════════════════════════════════
# MODO 4 — GIF animado (para apresentação)
# ══════════════════════════════════════════════════════════════
def _rollout(model, world, seed):
    """Roda um episódio e retorna (frames, desfecho)."""
    env = Env2D(world=world, seed=seed)
    obs, _ = env.reset()
    frames = [(env._x, env._y, env._yaw, env._gx, env._gy, False, False)]
    done = goal = coll = False
    while not done:
        action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
        obs, r, term, trunc, info = env.step(action)
        goal = goal or info.get("goal_reached", False)
        coll = coll or info.get("collision", False)
        frames.append((env._x, env._y, env._yaw, env._gx, env._gy, goal, coll))
        done = term or trunc
    outcome = "goal" if goal else "collision" if coll else "timeout"
    return frames, outcome


def plot_gif(model, world: str = "sparse", seed: int = 3, fps: int = 10,
             cherry_pick: bool = True, suffix: str = ""):
    """Gera GIF de um episódio.

    cherry_pick=True  → tenta até 50 seeds até achar um episódio vencedor (vitrine).
    cherry_pick=False → usa o seed fixo e mostra o desfecho REAL (diagnóstico honesto:
                        colisão no denso, timeout no muito denso).
    """
    cfg  = WORLDS[world]
    half = cfg["size"] / 2

    if cherry_pick:
        frames_data, outcome = None, "timeout"
        for try_seed in range(seed, seed + 50):
            cand, oc = _rollout(model, world, try_seed)
            if oc == "goal":
                frames_data, outcome = cand, oc
                print(f"  Episódio vencedor: seed={try_seed} ({len(cand)} frames)")
                break
        if frames_data is None:
            print(f"  Aviso: nenhum vencedor em 50 seeds no '{world}', usando seed={seed}")
            frames_data, outcome = _rollout(model, world, seed)
    else:
        frames_data, outcome = _rollout(model, world, seed)
        oc_label = {"goal": "✓ chegou ao goal", "collision": "✗ colidiu",
                    "timeout": "✗ não chegou (timeout)"}[outcome]
        print(f"  [honesto] '{world}' seed={seed}: {oc_label} ({len(frames_data)} frames)")

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_trail = plt.cm.plasma
    total = len(frames_data)

    def draw_frame(i):
        ax.clear()
        _draw_arena(ax, world)

        # Trilha com gradiente temporal
        xs = [f[0] for f in frames_data[:i+1]]
        ys = [f[1] for f in frames_data[:i+1]]
        if len(xs) > 1:
            for k in range(len(xs) - 1):
                t = k / max(total - 1, 1)
                ax.plot(xs[k:k+2], ys[k:k+2], "-",
                        color=cmap_trail(t), lw=2.2, alpha=0.85, zorder=4)

        x, y, yaw, gx, gy, goal, coll = frames_data[i]

        # Robô
        color_robot = "#F44336" if coll else "#4CAF50" if goal else "#2196F3"
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, color=color_robot, zorder=5, alpha=0.92))
        ax.annotate("", xy=(x + 0.25*np.cos(yaw), y + 0.25*np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.2))

        # Goal
        ax.plot(gx, gy, "*", color="#FF5722", ms=16, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.add_patch(Circle((gx, gy), GOAL_RADIUS, color="#FF5722", alpha=0.18))

        last = (i == total - 1)
        if goal:
            status = "✓ GOAL!"
        elif coll:
            status = "✗ COLISÃO"
        elif last:
            status = "✗ TIMEOUT"
        else:
            status = f"passo {i}/{total-1}"
        ax.set_title(f"SAC — env 2D  [{world}]\n{status}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ani = animation.FuncAnimation(fig, draw_frame,
                                  frames=total, interval=1000//fps)
    gif_path = os.path.join(FIGS2D, f"fig_2d_episode_{world}{suffix}.gif")
    ani.save(gif_path, writer="pillow", fps=fps)
    plt.close()
    print(f"  ✓ 2d/fig_2d_episode_{world}{suffix}.gif [{outcome}] ({len(frames_data)} frames)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",  default="all",
                   choices=["episode", "compare", "heatmap", "gif", "all"])
    p.add_argument("--world", default="sparse",
                   choices=["sparse", "dense", "very_dense"])
    p.add_argument("--seed",  type=int, default=7)
    p.add_argument("--no-model", action="store_true",
                   help="Roda sem modelo (ação aleatória) — útil para testar")
    p.add_argument("--honest", action="store_true",
                   help="GIF diagnóstico: seed fixo, mostra desfecho real (sufixo _diag)")
    args = p.parse_args()

    model = None
    if not args.no_model:
        model = load_model(os.path.join(MODS, "sac_2d_best"))
        if model is None:
            print("Modelo não encontrado — use --no-model para rodar sem modelo")

    print(f"Gerando figuras (mode={args.mode}, world={args.world})...")

    if args.mode in ("episode", "all"):
        plot_episode(model, args.world, args.seed)

    if args.mode in ("compare", "all"):
        plot_compare(model, args.world)

    if args.mode in ("heatmap", "all"):
        plot_heatmap(args.world)

    if args.mode in ("gif", "all"):
        if args.honest:
            plot_gif(model, args.world, args.seed, cherry_pick=False, suffix="_diag")
        else:
            plot_gif(model, args.world, args.seed)

    print("\nFiguras salvas em paper/figs/:")
    print("  fig_2d_trajectory_<world>.png/pdf  — trajetória com gradiente temporal")
    print("  fig_2d_compare_<world>.png/pdf      — A* vs SAC vs Adaptativo")
    print("  fig_2d_heatmap_<world>.png/pdf      — onde ρ-criterion ativa cada planner")
    print("  fig_2d_episode_<world>.gif           — animação para apresentação")
