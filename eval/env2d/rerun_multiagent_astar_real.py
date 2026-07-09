"""
rerun_multiagent_astar_real.py — Refaz o achado multiagente "A* independente
vs SAC independente" (21/06/2026) com A* REAL em vez da política de linha
reta usada em eval_multi_2d.py (`astar_policy()`, documentada no próprio
código como "aponta e avança em linha reta para o goal" — não é busca A*).

Cada agente planeja seu próprio caminho A* (busca em grade, heapq,
astar_planner.py) contra os obstáculos ESTÁTICOS do mundo, ignorando os
outros robôs no planejamento (mesmo desenho experimental do achado
original: robôs "independentes", sem coordenação, cada um só percebe os
vizinhos via LIDAR durante a execução, não no planejamento). Isso isola
exatamente a mesma pergunta do achado original — "planejador clássico
independente vs política aprendida independente, sem coordenação" — agora
com busca real.

Uso:
    python3 -m eval.env2d.rerun_multiagent_astar_real --n-agents 4 --trials 20
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d_multi import MultiAgentEnv2D
from eval.env2d.astar_planner import plan_astar

ROBOT_RADIUS = 0.17


def real_astar_actions(env: MultiAgentEnv2D, paths: dict, idxs: dict) -> np.ndarray:
    """Perseguição de waypoint (pure-pursuit) do caminho A* real de cada agente."""
    actions = []
    for i in range(env.N):
        path = paths[i]
        idx = idxs[i]
        x, y, yaw = env.x[i], env.y[i], env.yaw[i]
        while idx < len(path) - 1 and np.hypot(path[idx][0] - x, path[idx][1] - y) < 0.18:
            idx += 1
        idxs[i] = idx
        tx, ty = path[idx]
        dx, dy = tx - x, ty - y
        desired_yaw = np.arctan2(dy, dx)
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        w_norm = float(np.clip(yaw_err / (np.pi / 2), -1.0, 1.0))
        v_norm = float(np.clip(1.0 - abs(yaw_err) / (np.pi / 2), 0.15, 1.0))
        actions.append([v_norm, w_norm])
    return np.array(actions, dtype=np.float32)


def run_trial_real_astar(n_agents, world, seed):
    env = MultiAgentEnv2D(n_agents, world=world, seed=seed)
    env.reset()
    paths, idxs = {}, {}
    for i in range(n_agents):
        path = plan_astar((env.x[i], env.y[i]), (env.gx[i], env.gy[i]),
                           env.obstacles, env.arena / 2.0, ROBOT_RADIUS + 0.08)
        paths[i] = path if path is not None else [(env.x[i], env.y[i]), (env.gx[i], env.gy[i])]
        idxs[i] = 0
    done = False
    while not done:
        actions = real_astar_actions(env, paths, idxs)
        obs, done = env.step(actions)
    return env.metrics()


def run_trial_sac(model, n_agents, world, seed):
    env = MultiAgentEnv2D(n_agents, world=world, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        actions, _ = model.predict(obs, deterministic=True)
        obs, done = env.step(actions)
    return env.metrics()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--worlds", nargs="+", default=["sparse", "dense", "very_dense"])
    args = ap.parse_args()

    from stable_baselines3 import SAC
    MODS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    model = SAC.load(os.path.join(MODS, "sac_2d_best.zip"), device="cpu")

    print(f"{'mundo':>12} {'estratégia':>12} {'goal_rate':>10} {'inter_coll':>11}")
    rows = []
    for world in args.worlds:
        for strategy in ["astar_real", "sac"]:
            goals, colls = [], []
            for t in range(args.trials):
                if strategy == "astar_real":
                    m = run_trial_real_astar(args.n_agents, world, seed=2000 + t)
                else:
                    m = run_trial_sac(model, args.n_agents, world, seed=2000 + t)
                goals.append(m["goal_rate"])
                colls.append(m["inter_collision"])
                rows.append({"world": world, "strategy": strategy, "trial": t,
                              "goal_rate": m["goal_rate"], "inter_collision": m["inter_collision"]})
            print(f"{world:>12} {strategy:>12} {np.mean(goals):>9.0%} {np.mean(colls):>10.0%}")

    import pandas as pd
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results_abstract", "multiagent_astar_real_vs_sac.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
