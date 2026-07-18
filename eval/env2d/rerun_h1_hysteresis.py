"""
rerun_h1_hysteresis.py — Revalida H1 com o ρ-criterion CORRIGIDO: decisão
per-step (não mais decide-uma-vez-no-reset, como rerun_h1_mixed.py fazia
sem perceber) COM HISTERESE, para eliminar o chattering descoberto em
09/07/2026 (46/300 episódios com ≥2 trocas de planejador, até 48 trocas
num único episódio de 200 passos — ver [[project-treino-sparse-08jul]]).

Histerese: dois limiares em vez de um.
  - ρ_low  = 0,28 → só volta pro A* se ρ cair ABAIXO deste valor
  - ρ_high = 0,32 → só vai pro BC se ρ subir ACIMA deste valor
  - Entre os dois, mantém o planejador atual (zona morta).
Isso é o padrão-ouro de controle supervisório chaveado (Hespanha & Morse)
para eliminar oscilação perto do limiar de decisão.

Uso:
    python3 eval/env2d/rerun_h1_hysteresis.py --trials 1500
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d import Env2D
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc
from eval.env2d.rerun_h1_mixed import local_rho, WORLDS_LIST, _path_length, optimal_path_length

RHO_LOW = 0.28
RHO_HIGH = 0.32


def run_adaptive_hysteresis(env, seed, astar_policy, bc_policy):
    env.reset(seed=seed)
    start_xy = (env._x, env._y)
    goal_xy = (env._gx, env._gy)
    rho0 = local_rho(env)
    current = "astar" if rho0 < RHO_LOW else "bc"  # zona morta no primeiro passo: usa RHO_HIGH como desempate
    if RHO_LOW <= rho0 <= RHO_HIGH:
        current = "astar" if rho0 < 0.30 else "bc"  # fallback ao limiar original só no instante inicial
    if current == "astar":
        astar_policy.reset(env)
    traj = [(env._x, env._y)]
    obs = env._obs()
    done, steps = False, 0
    switches = 0
    while not done and steps < 200:
        rho = local_rho(env)
        new_choice = current
        if current == "astar" and rho > RHO_HIGH:
            new_choice = "bc"
        elif current == "bc" and rho < RHO_LOW:
            new_choice = "astar"
        if new_choice != current:
            switches += 1
            current = new_choice
            if current == "astar":
                astar_policy.reset(env)  # replaneja a partir da posição atual
        if current == "astar":
            a = astar_policy.act(env)
        else:
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        traj.append((env._x, env._y))
        done = term or trunc
        steps += 1
    traveled = _path_length(traj)
    optimal = optimal_path_length(env, start_xy, goal_xy)
    route_efficiency = (traveled / optimal) if (optimal and optimal > 1e-6) else None
    return bool(info.get("goal_reached", False)), switches, route_efficiency


def run_fixed(env, seed, policy_type, astar_policy, bc_policy):
    """A* real ou BC real, fixo o episódio inteiro (baseline, sem mudança)."""
    env.reset(seed=seed)
    start_xy = (env._x, env._y)
    goal_xy = (env._gx, env._gy)
    if policy_type == "astar":
        astar_policy.reset(env)
    traj = [(env._x, env._y)]
    obs = env._obs()
    done, steps = False, 0
    while not done and steps < 200:
        if policy_type == "astar":
            a = astar_policy.act(env)
        else:
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        traj.append((env._x, env._y))
        done = term or trunc
        steps += 1
    traveled = _path_length(traj)
    optimal = optimal_path_length(env, start_xy, goal_xy)
    route_efficiency = (traveled / optimal) if (optimal and optimal > 1e-6) else None
    return bool(info.get("goal_reached", False)), route_efficiency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1500)
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
        bc = bc_policies[world]

        astar_ok, astar_eff = run_fixed(env, seed, "astar", astar_policy, bc)
        bc_ok, bc_eff = run_fixed(env, seed, "bc", astar_policy, bc)
        adaptive_ok, switches, adaptive_eff = run_adaptive_hysteresis(env, seed, astar_policy, bc)

        rows.append({"trial": trial, "world": world, "astar": int(astar_ok),
                      "bc": int(bc_ok), "adaptive": int(adaptive_ok), "switches": switches,
                      "astar_route_efficiency": astar_eff, "bc_route_efficiency": bc_eff,
                      "adaptive_route_efficiency": adaptive_eff})
        if (trial + 1) % 200 == 0:
            print(f"  {trial+1}/{args.trials} trials...", flush=True)

    dt = time.time() - t0
    print(f"\nTempo total: {dt:.1f}s")

    astar_sr = np.mean([r["astar"] for r in rows])
    bc_sr = np.mean([r["bc"] for r in rows])
    adaptive_sr = np.mean([r["adaptive"] for r in rows])
    oracle_sr = np.mean([max(r["astar"], r["bc"]) for r in rows])
    regret = oracle_sr - adaptive_sr
    switches_arr = np.array([r["switches"] for r in rows])

    print("\n=== H1 com histerese (per-step, RHO_LOW=0.28, RHO_HIGH=0.32) ===")
    print(f"  A* real:      {astar_sr:.1%}")
    print(f"  BC real:      {bc_sr:.1%}")
    print(f"  Adaptativo:   {adaptive_sr:.1%}")
    print(f"  Oracle:       {oracle_sr:.1%}")
    print(f"  Regret:       {regret:.1%}")
    print(f"\n  Trocas/episódio: média={switches_arr.mean():.2f}, max={switches_arr.max()}, "
          f"episódios com >=2 trocas={(switches_arr>=2).sum()}/{args.trials}")

    def _mean_route_eff(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return np.mean(vals) if vals else float("nan")

    print("\n  Eficiência de rota (percorrido/ótimo, apenas trials alcançáveis na grade):")
    print(f"    A* real:      {_mean_route_eff('astar_route_efficiency'):.3f}")
    print(f"    BC real:      {_mean_route_eff('bc_route_efficiency'):.3f}")
    print(f"    Adaptativo:   {_mean_route_eff('adaptive_route_efficiency'):.3f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results_abstract", "h1_hysteresis_2d.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "world", "astar", "bc", "adaptive", "switches",
                                                "astar_route_efficiency", "bc_route_efficiency",
                                                "adaptive_route_efficiency"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
