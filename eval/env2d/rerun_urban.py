"""
rerun_urban.py — Fecha os itens "cenário urbano", "obstáculos móveis" e
"eficiência de rota em cenários urbanos simulados" do plano de trabalho
oficial (PI08078-2024), usando a mesma infraestrutura A*/BC/ρ-criterion já
validada em rerun_h1_mixed.py (reaproveita run_one, _run_episode,
route_efficiency), sem depender do Gazebo.

Compara três condições no mesmo protocolo pareado (mesmo start/goal via
seed) usado em rerun_h1_mixed.py:
  1. static   — world="urban_grid", sem obstáculos móveis (baseline)
  2. dynamic  — world="urban_grid", com 1 obstáculo móvel programado
               (trajetória linear com bounce elástico, ver env_2d.py)

Uso:
    python3 eval/env2d/rerun_urban.py --trials 500
"""
import os, sys, argparse, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import Env2D
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc
from eval.env2d.rerun_h1_mixed import run_one

# Obstáculo móvel: cruza o corredor horizontal (y perto de 0), trajetória
# linear com bounce elástico nas bordas da arena — ver env_2d.py::Env2D.
DYNAMIC_OBSTACLE_SPEC = [
    {"cx": -1.5, "cy": 0.3, "cr": 0.15, "vx": 0.7, "vy": -0.25},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed0", type=int, default=9000)
    args = p.parse_args()

    astar_policy = AStarPolicy()
    bc_policy = load_bc("urban_grid")

    conditions = {
        "static": None,
        "dynamic": DYNAMIC_OBSTACLE_SPEC,
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "results_abstract", "urban_grid_results.csv")

    def _fmt_eff(v):
        return f"{v:.4f}" if v is not None else ""

    all_rows = []
    for cond_name, dyn_spec in conditions.items():
        print(f"\n=== Condição: {cond_name} ===")
        rows = []
        t0 = time.time()
        for trial in range(args.trials):
            seed = args.seed0 + trial
            env = Env2D(world="urban_grid", seed=seed, dynamic_obstacles=dyn_spec)
            res = run_one(env, seed, astar_policy, bc_policy)
            res["condition"] = cond_name
            res["trial"] = trial
            rows.append(res)
            if (trial + 1) % 100 == 0:
                print(f"  {trial+1}/{args.trials} trials...", flush=True)
        dt = time.time() - t0
        print(f"Tempo total: {dt:.1f}s ({dt/args.trials*1000:.0f} ms/trial)")

        astar_sr = np.mean([r["astar"] for r in rows])
        bc_sr = np.mean([r["bc"] for r in rows])
        adaptive_sr = np.mean([r["adaptive"] for r in rows])
        oracle_sr = np.mean([max(r["astar"], r["bc"]) for r in rows])
        regret = oracle_sr - adaptive_sr

        def _mean_eff(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return np.mean(vals) if vals else float("nan")

        print(f"  A* real:      {astar_sr:.1%}  (route_efficiency={_mean_eff('astar_route_efficiency'):.3f})")
        print(f"  BC real:      {bc_sr:.1%}  (route_efficiency={_mean_eff('bc_route_efficiency'):.3f})")
        print(f"  Adaptativo:   {adaptive_sr:.1%}  (route_efficiency={_mean_eff('adaptive_route_efficiency'):.3f})")
        print(f"  Oracle:       {oracle_sr:.1%}")
        print(f"  Regret:       {regret:.1%}")

        all_rows.extend(rows)

    with open(out_path, "w") as f:
        f.write("condition,trial,rho0,used_astar,astar,bc,adaptive,"
                "astar_route_efficiency,bc_route_efficiency,adaptive_route_efficiency\n")
        for r in all_rows:
            f.write(f"{r['condition']},{r['trial']},{r['rho0']:.4f},{r['used_astar']},"
                    f"{int(r['astar'])},{int(r['bc'])},{int(r['adaptive'])},"
                    f"{_fmt_eff(r['astar_route_efficiency'])},"
                    f"{_fmt_eff(r['bc_route_efficiency'])},"
                    f"{_fmt_eff(r['adaptive_route_efficiency'])}\n")
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
