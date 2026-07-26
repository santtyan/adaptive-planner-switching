"""
rerun_h1_oneshot_perstep.py — Gera o comparativo para a Figura F2 (chattering
one-shot vs. histerese, docs/PLANO_CORRECAO.md). Reproduz a política per-step
SEM histerese (limiar único rho*=0.30) para poder comparar diretamente com a
versão COM histerese já medida em h1_hysteresis_2d.csv.

Esse dado (per-step sem histerese, n=1500) nunca foi rodado em escala completa
-- o achado original de chattering (DEVELOPMENT_LOG.md, Fase 10, 09/07/2026)
veio de um piloto de 100/300 trials. Esta rodada fecha essa lacuna.

Uso:
    python3 eval/env2d/rerun_h1_oneshot_perstep.py --trials 1500 --seed0 5000
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d import Env2D
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc
from eval.env2d.rerun_h1_mixed import local_rho, WORLDS_LIST

RHO_STAR = 0.30


def run_perstep_no_hysteresis(env, seed, astar_policy, bc_policy):
    env.reset(seed=seed)
    rho0 = local_rho(env)
    current = "astar" if rho0 < RHO_STAR else "bc"
    if current == "astar":
        astar_policy.reset(env)
    obs = env._obs()
    done, steps = False, 0
    switches = 0
    while not done and steps < 200:
        rho = local_rho(env)
        new_choice = "astar" if rho < RHO_STAR else "bc"
        if new_choice != current:
            switches += 1
            current = new_choice
            if current == "astar":
                astar_policy.reset(env)
        if current == "astar":
            a = astar_policy.act(env)
        else:
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
    return switches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=5000)
    args = p.parse_args()

    astar_policy = AStarPolicy()
    bc_policies = {w: load_bc(w) for w in WORLDS_LIST}
    rng = np.random.default_rng(42)

    switches_list = []
    t0 = time.time()
    for trial in range(args.trials):
        world = rng.choice(WORLDS_LIST)
        env = Env2D(world=world, seed=args.seed0 + trial)
        bc = bc_policies[world]
        switches = run_perstep_no_hysteresis(env, args.seed0 + trial, astar_policy, bc)
        switches_list.append(switches)
        if (trial + 1) % 300 == 0:
            print(f"  {trial+1}/{args.trials}...", flush=True)

    dt = time.time() - t0
    switches_arr = np.array(switches_list)
    print(f"\nTempo total: {dt:.1f}s")
    print(f"Switches/episódio (SEM histerese): média={switches_arr.mean():.2f}, "
          f"max={switches_arr.max()}, episódios com >=2 trocas={(switches_arr>=2).sum()}/{args.trials} "
          f"({100*(switches_arr>=2).sum()/args.trials:.1f}%)")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results_abstract", "h1_oneshot_perstep_switches.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "switches"])
        for i, s in enumerate(switches_list):
            writer.writerow([i, s])
    print(f"Salvo em {out_path}")


if __name__ == "__main__":
    main()
