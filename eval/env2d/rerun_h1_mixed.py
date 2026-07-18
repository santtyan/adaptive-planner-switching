"""
rerun_h1_mixed.py — Revalida H1 com o protocolo CORRETO: pool misto de
densidades no mesmo lote de trials (replicando a estrutura do Monte Carlo
original — 1.500 trials cobrindo ρ de 0,05 a 0,60), não mundos fixos em
lotes separados (rerun_h1_real.py testou isso e a comutação nunca era
exercitada dentro de dense/very_dense, onde ρ_local já start >= ρ*).

Cada trial: sorteia um mundo (sparse/dense/very_dense — proxy discreto do
espectro contínuo de densidade), roda A* real, BC real (casado com o mundo)
e o ρ-criterion adaptativo NO MESMO trial (mesmo start/goal, mesma seed) —
pareado, para computar regret por trial como no protocolo original:
  regret(π) = E[R_oracle] - E[R_π],  oracle = melhor entre A*/BC nesse trial.

Uso:
    python3 eval/env2d/rerun_h1_mixed.py --trials 500
"""
import os, sys, argparse, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import Env2D, _scan
from eval.env2d.astar_planner import AStarPolicy, plan_astar
from eval.env2d.rerun_h1_real import BCPolicy, load_bc, RHO_STAR

WORLDS_LIST = ["sparse", "dense", "very_dense"]


def local_rho(env: Env2D) -> float:
    ranges = _scan(env._x, env._y, env._yaw, env.obstacles, env.arena,
                    walls=getattr(env, "walls", None))
    return float(np.mean(ranges < 1.0))


def _path_length(waypoints) -> float:
    """Comprimento euclidiano acumulado de uma sequência de (x, y)."""
    if waypoints is None or len(waypoints) < 2:
        return 0.0
    pts = np.asarray(waypoints, dtype=np.float64)
    return float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))


def optimal_path_length(env: Env2D, start_xy, goal_xy) -> float:
    """Comprimento do caminho ótimo A* entre start_xy e goal_xy (posições do
    RESET do episódio, não a posição atual do robô), mesma grade/margem
    usada por AStarPolicy — referência para route_efficiency.
    Retorna None se o par start/goal for inalcançável na grade rasterizada
    (mesmo critério de fallback já usado em AStarPolicy.reset)."""
    path = plan_astar(
        start_xy, goal_xy, env.obstacles,
        env.arena / 2.0, 0.17 + 0.08,  # ROBOT_RADIUS + safety_margin (AStarPolicy default)
        walls=getattr(env, "walls", None),
        blocks=getattr(env, "blocks", None),
    )
    if path is None:
        return None
    return _path_length(path)


def _run_episode(env: Env2D, seed: int, act_fn, astar_policy=None) -> dict:
    """Roda um episódio, acumulando trajetória passo a passo (mesmo padrão
    de visualize_2d.py::_run_episode), e retorna sucesso + métricas de rota."""
    env.reset(seed=seed)
    start_xy = (env._x, env._y)
    goal_xy = (env._gx, env._gy)
    if astar_policy is not None:
        astar_policy.reset(env)
    traj = [(env._x, env._y)]
    obs = env._obs()
    done, steps = False, 0
    while not done and steps < 200:
        a = act_fn(env, obs)
        obs, r, term, trunc, info = env.step(a)
        traj.append((env._x, env._y))
        done = term or trunc
        steps += 1
    # `optimal` é o caminho A* até o CENTRO do goal; a política real para
    # dentro de GOAL_RADIUS=0,25m, então route_efficiency < 1.0 é esperado
    # em trials curtos (a política "economiza" a última fração do caminho
    # dentro do raio de tolerância) — não é um bug de medição.
    traveled = _path_length(traj)
    optimal = optimal_path_length(env, start_xy, goal_xy)
    route_efficiency = (traveled / optimal) if (optimal and optimal > 1e-6) else None
    return {
        "goal_reached": bool(info.get("goal_reached", False)),
        "traveled": traveled,
        "optimal": optimal,
        "route_efficiency": route_efficiency,
    }


def run_one(env: Env2D, seed: int, astar_policy: AStarPolicy, bc_policy) -> dict:
    """Roda astar, bc e adaptive no MESMO trial (start/goal pareados via seed)."""
    out = {}

    # astar
    def astar_act(env, obs):
        return astar_policy.act(env)
    res = _run_episode(env, seed, astar_act, astar_policy=astar_policy)
    out["astar"] = res["goal_reached"]
    out["astar_traveled"] = res["traveled"]
    out["astar_optimal"] = res["optimal"]
    out["astar_route_efficiency"] = res["route_efficiency"]

    # bc
    def bc_act(env, obs):
        with torch.no_grad():
            return bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    res = _run_episode(env, seed, bc_act)
    out["bc"] = res["goal_reached"]
    out["bc_traveled"] = res["traveled"]
    out["bc_optimal"] = res["optimal"]
    out["bc_route_efficiency"] = res["route_efficiency"]

    # adaptive (ρ-criterion): decide no reset, com o mesmo par start/goal
    env.reset(seed=seed)
    rho0 = local_rho(env)
    use_astar = rho0 < RHO_STAR

    def adaptive_act(env, obs):
        if use_astar:
            return astar_policy.act(env)
        with torch.no_grad():
            return bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    res = _run_episode(env, seed, adaptive_act, astar_policy=(astar_policy if use_astar else None))
    out["adaptive"] = res["goal_reached"]
    out["adaptive_traveled"] = res["traveled"]
    out["adaptive_optimal"] = res["optimal"]
    out["adaptive_route_efficiency"] = res["route_efficiency"]
    out["rho0"] = rho0
    out["used_astar"] = use_astar
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed0", type=int, default=5000)
    args = p.parse_args()

    astar_policy = AStarPolicy()
    bc_policies = {w: load_bc(w) for w in WORLDS_LIST}
    rng = np.random.default_rng(42)

    rows = []
    t0 = time.time()
    for trial in range(args.trials):
        world = rng.choice(WORLDS_LIST)
        env = Env2D(world=world, seed=args.seed0 + trial)
        seed = args.seed0 + trial
        res = run_one(env, seed, astar_policy, bc_policies[world])
        res["world"] = world
        res["trial"] = trial
        rows.append(res)
        if (trial + 1) % 100 == 0:
            print(f"  {trial+1}/{args.trials} trials...", flush=True)

    dt = time.time() - t0
    print(f"\nTempo total: {dt:.1f}s ({dt/args.trials*1000:.0f} ms/trial)")

    astar_sr = np.mean([r["astar"] for r in rows])
    bc_sr = np.mean([r["bc"] for r in rows])
    adaptive_sr = np.mean([r["adaptive"] for r in rows])
    oracle_sr = np.mean([max(r["astar"], r["bc"]) for r in rows])
    regret_adaptive = oracle_sr - adaptive_sr
    used_astar_frac = np.mean([r["used_astar"] for r in rows])

    print("\n=== H1 — validação com planejadores REAIS, pool misto de densidades ===")
    print(f"  A* real (fixo):        {astar_sr:.1%}")
    print(f"  BC real (fixo, casado): {bc_sr:.1%}")
    print(f"  ρ-criterion (real):     {adaptive_sr:.1%}")
    print(f"  Oracle (melhor por trial): {oracle_sr:.1%}")
    print(f"  Regret do ρ-criterion vs oracle: {regret_adaptive:.1%}")
    print(f"  Fração de trials roteados p/ A* (ρ<0,30): {used_astar_frac:.1%}")

    def _mean_route_eff(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return np.mean(vals) if vals else float("nan")

    print("\n  Eficiência de rota (percorrido/ótimo, 1.0=ideal, apenas trials alcançáveis na grade):")
    print(f"    A* real:      {_mean_route_eff('astar_route_efficiency'):.3f}")
    print(f"    BC real:      {_mean_route_eff('bc_route_efficiency'):.3f}")
    print(f"    Adaptativo:   {_mean_route_eff('adaptive_route_efficiency'):.3f}")

    per_world = {}
    for w in WORLDS_LIST:
        sub = [r for r in rows if r["world"] == w]
        per_world[w] = {
            "astar": np.mean([r["astar"] for r in sub]),
            "bc": np.mean([r["bc"] for r in sub]),
            "adaptive": np.mean([r["adaptive"] for r in sub]),
            "n": len(sub),
        }
    print("\nPor mundo:")
    for w, d in per_world.items():
        print(f"  {w:>10}: astar={d['astar']:.1%} bc={d['bc']:.1%} adaptive={d['adaptive']:.1%} (n={d['n']})")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "results_abstract", "h1_real_2d_mixed_pool.csv")

    def _fmt_eff(v):
        return f"{v:.4f}" if v is not None else ""

    with open(out_path, "w") as f:
        f.write("trial,world,rho0,used_astar,astar,bc,adaptive,"
                "astar_route_efficiency,bc_route_efficiency,adaptive_route_efficiency\n")
        for r in rows:
            f.write(f"{r['trial']},{r['world']},{r['rho0']:.4f},{r['used_astar']},"
                    f"{int(r['astar'])},{int(r['bc'])},{int(r['adaptive'])},"
                    f"{_fmt_eff(r['astar_route_efficiency'])},"
                    f"{_fmt_eff(r['bc_route_efficiency'])},"
                    f"{_fmt_eff(r['adaptive_route_efficiency'])}\n")
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
