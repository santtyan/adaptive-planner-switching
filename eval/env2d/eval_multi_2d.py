"""
eval_multi_2d.py — Avaliação REAL de RL independente multi-agente no env 2D.

Roda N robôs com a política SAC treinada (sac_2d_best), de forma INDEPENDENTE
(sem comunicação), e mede a degradação conforme N cresce:
  - inter_collision : robôs colidem entre si (RL não treinado p/ evitar outros robôs)
  - goal_rate       : fração que atinge o goal (cai com N)
  - deadlock        : bloqueio mútuo
  - time_to_goal

Estratégias REAIS comparadas:
  "sac"   — política SAC independente por robô  (foco do experimento)
  "astar" — política analítica reta independente (baseline)

Substitui os dados SINTÉTICOS de eval/eval_multiagent_independent_rl.py por dados reais.

Uso:
  python3 -m eval.env2d.eval_multi_2d --n-agents 1 2 3 4 5 6 8 --trials 20
"""

import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.env2d.env_2d_multi import MultiAgentEnv2D
from eval.env2d.env_2d import WORLDS, ROBOT_RADIUS, GOAL_RADIUS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "paper", "figs")
MODS = os.path.join(ROOT, "models")

STRATEGY_LABELS = {"astar": "A* fixo\n(independente)", "sac": "SAC\n(independente)"}
STRATEGY_COLORS = {"astar": "#4CAF50", "sac": "#2196F3"}


def astar_policy(env: MultiAgentEnv2D, i: int) -> np.ndarray:
    """Política analítica: aponta e avança em linha reta para o goal."""
    dx = env.gx[i] - env.x[i]
    dy = env.gy[i] - env.y[i]
    dist = np.hypot(dx, dy)
    ang  = np.arctan2(dy, dx)
    dtheta = (ang - env.yaw[i] + np.pi) % (2 * np.pi) - np.pi
    v_norm = min(1.0, dist / 0.5) * (abs(dtheta) < 0.6)  # só avança alinhado
    w_norm = np.clip(dtheta / np.pi, -1.0, 1.0)
    return np.array([v_norm, w_norm])


def run_trial(strategy, model, n_agents, world, seed):
    env = MultiAgentEnv2D(n_agents, world=world, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        if strategy == "sac":
            actions, _ = model.predict(obs, deterministic=True)
        else:  # astar
            actions = np.stack([astar_policy(env, i) for i in range(n_agents)])
        obs, done = env.step(actions)
    return env.metrics()


def run_experiment(strategies, n_agents_list, n_trials, world, model):
    rows = []
    for strategy in strategies:
        for n in n_agents_list:
            for t in range(n_trials):
                m = run_trial(strategy, model, n, world, seed=1000 + t)
                m["strategy"] = strategy
                rows.append(m)
            sub = [r for r in rows if r["strategy"] == strategy and r["n_agents"] == n]
            gr = np.mean([r["goal_rate"] for r in sub])
            ic = np.mean([r["inter_collision"] for r in sub])
            print(f"  {strategy:5s} N={n}: goal={gr:.0%}  inter_coll={ic:.0%}")
    return pd.DataFrame(rows)


# ── Figuras ────────────────────────────────────────────────────
def plot_degradation(df, out):
    ns = sorted(df["n_agents"].unique())
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax2 = ax1.twinx()

    sac = df[df.strategy == "sac"]
    goal = [sac[sac.n_agents == n]["goal_rate"].mean() for n in ns]
    gstd = [sac[sac.n_agents == n]["goal_rate"].std() for n in ns]
    coll = [sac[sac.n_agents == n]["inter_collision"].mean() for n in ns]

    ns_a = np.array(ns)
    ax1.plot(ns_a, goal, "o-", color="#2196F3", lw=2.2, ms=7, label="Taxa de goal (esq.)")
    ax1.fill_between(ns_a, np.array(goal)-np.array(gstd), np.array(goal)+np.array(gstd),
                     alpha=0.15, color="#2196F3")
    ax2.plot(ns_a, coll, "s--", color="#F44336", lw=2.2, ms=7, label="Colisão inter-robô (dir.)")

    ax1.set_xlabel("Número de robôs independentes (N)", fontsize=11)
    ax1.set_ylabel("Taxa de goal média por robô", color="#2196F3", fontsize=11)
    ax2.set_ylabel("Taxa de colisão inter-robô", color="#F44336", fontsize=11)
    ax1.set_ylim(0, 1.05); ax2.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xticks(ns)
    ax1.set_title("Degradação do SAC independente com N robôs (dados REAIS, env 2D)\n"
                  f"arena densa ρ≈{_density(df):.2f}, sem coordenação central",
                  fontsize=10, fontweight="bold")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=9, loc="center left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"fig_marl_motivation_degradation_2d.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_marl_motivation_degradation_2d.png")


def plot_summary(df, out):
    ns = sorted(df["n_agents"].unique())
    strategies = [s for s in ["astar", "sac"] if s in df.strategy.unique()]
    metrics = [("inter_collision", "Colisão inter-robô"),
               ("goal_rate", "Taxa de goal"),
               ("deadlock", "Deadlock")]
    x = np.arange(len(ns)); width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        for k, s in enumerate(strategies):
            vals = [df[(df.strategy == s) & (df.n_agents == n)][col].mean() for n in ns]
            ax.bar(x + k*width, vals, width, label=STRATEGY_LABELS[s],
                   color=STRATEGY_COLORS[s], edgecolor="k", lw=0.6)
        ax.set_xticks(x + width/2); ax.set_xticklabels([f"N={n}" for n in ns])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend(fontsize=8)
    fig.suptitle("RL independente vs A* independente — dados REAIS (env 2D, sem coordenação)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"fig_marl_motivation_summary_2d.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_marl_motivation_summary_2d.png")


def plot_snapshot(model, world, n_agents, out, seed=2):
    env = MultiAgentEnv2D(n_agents, world=world, seed=seed)
    obs = env.reset()
    trajs = [[(env.x[i], env.y[i])] for i in range(n_agents)]
    done = False
    while not done:
        actions, _ = model.predict(obs, deterministic=True)
        obs, done = env.step(actions)
        for i in range(n_agents):
            trajs[i].append((env.x[i], env.y[i]))

    half = env.arena / 2
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Rectangle((-half, -half), env.arena, env.arena,
                               fill=False, edgecolor="#37474F", lw=2))
    for cx, cy, cr in env.obstacles:
        ax.add_patch(Circle((cx, cy), cr, color="#607D8B", alpha=0.85))
    cmap = plt.cm.tab10
    for i in range(n_agents):
        xs = [p[0] for p in trajs[i]]; ys = [p[1] for p in trajs[i]]
        c = cmap(i % 10)
        ax.plot(xs, ys, "-", color=c, lw=2, alpha=0.8)
        ax.plot(xs[0], ys[0], "o", color=c, ms=10, markeredgecolor="white")
        ax.plot(env.gx[i], env.gy[i], "*", color=c, ms=15, markeredgecolor="black")
    ax.set_xlim(-half-.2, half+.2); ax.set_ylim(-half-.2, half+.2)
    ax.set_aspect("equal"); ax.grid(alpha=0.15)
    ax.set_title(f"{n_agents} robôs SAC independentes (env 2D, {world})\n"
                 f"★=goal  ●=início  | colisão inter-robô: {env.inter_collision}",
                 fontsize=11)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"fig_2d_multiagent_snapshot.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_2d_multiagent_snapshot.png")


def plot_degradation_by_density(dfs_by_world, out):
    """Figura comparativa: goal/inter_collision/deadlock vs N, 1 curva por densidade."""
    metrics = [("goal_rate", "Taxa de goal (SAC)"),
               ("inter_collision", "Colisão inter-robô (SAC)"),
               ("deadlock", "Deadlock (SAC)")]
    colors = {"sparse": "#2E7D32", "dense": "#F9A825", "very_dense": "#C62828"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, title) in zip(axes, metrics):
        for world, (df, rho) in dfs_by_world.items():
            sac = df[df.strategy == "sac"]
            ns = sorted(sac.n_agents.unique())
            vals = [sac[sac.n_agents == n][col].mean() for n in ns]
            ax.plot(ns, vals, "o-", color=colors.get(world, "gray"), lw=2.2, ms=7,
                    label=f"{world} (ρ≈{rho:.2f})")
        ax.set_xlabel("Número de robôs (N)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.03, 1.05); ax.set_xticks(ns)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("RL independente: degradação por densidade × N robôs (dados REAIS, env 2D)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / "marl" / f"fig_marl_degradation_by_density_2d.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ marl/fig_marl_degradation_by_density_2d.png")


def _density(df):
    return 0.0  # placeholder; sobrescrito no main com densidade real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--world", default="dense")
    ap.add_argument("--worlds", nargs="+", default=None,
                    help="Varre múltiplas densidades (ex.: sparse dense very_dense) "
                         "e gera figura comparativa. Sobrepõe --world.")
    ap.add_argument("--out", default=FIGS)
    args = ap.parse_args()

    from stable_baselines3 import SAC
    model_path = os.path.join(MODS, "sac_2d_best.zip")
    if not os.path.exists(model_path):
        print(f"ERRO: modelo não encontrado em {model_path}")
        print("Treine antes: python3 -m eval.env2d.train_2d --world sparse")
        sys.exit(1)
    model = SAC.load(model_path, device="cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "marl").mkdir(parents=True, exist_ok=True)

    # ρ nominal da tese (densidade LOCAL vista pelo LIDAR; alinhado a ρ*=0.30),
    # consistente com env_2d.py e diagnose_density.py. NÃO é a fração de área global.
    NOMINAL_RHO = {"sparse": 0.05, "dense": 0.30, "very_dense": 0.50}

    def rho_of(world):
        return NOMINAL_RHO.get(world, 0.0)

    # ── Modo progressivo: varre densidades ────────────────────
    if args.worlds:
        dfs = {}
        for world in args.worlds:
            rho = rho_of(world)
            print(f"\n=== world={world}  ρ≈{rho:.2f} ===")
            df = run_experiment(["astar", "sac"], args.n_agents, args.trials, world, model)
            df.to_csv(Path(args.out) / f"multiagent_2d_results_{world}.csv", index=False)
            dfs[world] = (df, rho)
        print("\nGerando figura comparativa...")
        plot_degradation_by_density(dfs, args.out)
        print("\nResumo por densidade (SAC, deadlock/goal):")
        for world, (df, rho) in dfs.items():
            sac = df[df.strategy == "sac"]
            nmax = max(args.n_agents)
            s = sac[sac.n_agents == nmax]
            print(f"  {world:11s} N={nmax}: goal={s.goal_rate.mean():.0%}  "
                  f"inter_coll={s.inter_collision.mean():.0%}  "
                  f"deadlock={s.deadlock.mean():.0%}")
        return

    # ── Modo single-world (comportamento original) ────────────
    rho = rho_of(args.world)
    global _density
    _density = lambda df: rho

    print(f"Avaliando RL independente multi-agente (REAL) — world={args.world} ρ≈{rho:.2f}")
    df = run_experiment(["astar", "sac"], args.n_agents, args.trials, args.world, model)

    csv = Path(args.out) / "multiagent_2d_results.csv"
    df.to_csv(csv, index=False)
    print(f"\nCSV: {csv}")

    print("\nGerando figuras...")
    plot_degradation(df, args.out)
    plot_summary(df, args.out)
    plot_snapshot(model, args.world, max(args.n_agents), args.out)

    print("\nResumo (SAC independente):")
    sac = df[df.strategy == "sac"]
    for n in sorted(df.n_agents.unique()):
        s = sac[sac.n_agents == n]
        print(f"  N={n}: goal={s.goal_rate.mean():.0%}  "
              f"inter_coll={s.inter_collision.mean():.0%}  "
              f"deadlock={s.deadlock.mean():.0%}")


if __name__ == "__main__":
    main()
