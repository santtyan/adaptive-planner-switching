"""
sweep_threshold_real.py — Recalibração de tau com planejadores REAIS (A* e BC,
não os mocks RRT*/PPO da Fase 1), com train/test split (Etapa 1.1/Passo 4,
docs/PLANO_CORRECAO.md).

Motivação: o relatório usa tau=0.30 "por continuidade", mas o unico dado que
existia (threshold_sensitivity.csv, sobre os mocks lineares da Fase 1) minimiza
regret em tau=0.20, nao em 0.30 -- e o proprio limiar foi calibrado e avaliado
no MESMO conjunto de dados (overfitting de selecao). Este script fecha as duas
lacunas: usa A*/BC reais, e separa calibracao (split "train") de avaliacao
(split "test") por seed par/impar.

Reaproveita a mesma amostragem pareada de eval/env2d/rerun_h1_mixed.py: cada
trial sorteia um mundo, roda A* real e BC real (casado por densidade) no MESMO
par start/goal, registra rho0 no reset. O criterio adaptivo, para um dado tau,
usa A* se rho0 < tau, BC caso contrario -- exatamente a mesma regra de
rerun_h1_mixed.py, so que parametrizada em tau em vez de fixa em RHO_STAR=0.30.

Uso:
    python3 eval/env2d/sweep_threshold_real.py --trials 1500 --seed0 5000
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d import Env2D
from eval.env2d.astar_planner import AStarPolicy
from eval.env2d.rerun_h1_real import load_bc
from eval.env2d.rerun_h1_mixed import local_rho, WORLDS_LIST, _run_episode

TAUS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def run_trial(env, seed, astar_policy, bc_policy):
    """Roda A* e BC reais no MESMO par start/goal (pareado), retorna sucesso
    dos dois e rho0 -- suficiente para reconstruir o adaptativo para
    QUALQUER tau depois, sem precisar re-simular por valor de tau."""
    env.reset(seed=seed)
    rho0 = local_rho(env)

    def astar_act(e, obs):
        return astar_policy.act(e)
    res_astar = _run_episode(env, seed, astar_act, astar_policy=astar_policy)

    def bc_act(e, obs):
        with torch.no_grad():
            return bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    res_bc = _run_episode(env, seed, bc_act)

    return {
        "rho0": rho0,
        "astar": res_astar["goal_reached"],
        "bc": res_bc["goal_reached"],
    }


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
        seed = args.seed0 + trial
        env = Env2D(world=world, seed=seed)
        r = run_trial(env, seed, astar_policy, bc_policies[world])
        r["trial"] = trial
        r["world"] = world
        r["split"] = "train" if trial % 2 == 0 else "test"
        rows.append(r)
        if (trial + 1) % 300 == 0:
            print(f"  {trial+1}/{args.trials} trials...", flush=True)

    dt = time.time() - t0
    print(f"\nTempo total: {dt:.1f}s ({dt/args.trials*1000:.0f} ms/trial)")

    def eval_tau(subset, tau):
        astar_sr = np.mean([r["astar"] for r in subset])
        bc_sr = np.mean([r["bc"] for r in subset])
        oracle_sr = np.mean([max(r["astar"], r["bc"]) for r in subset])
        adaptive_ok = [r["astar"] if r["rho0"] < tau else r["bc"] for r in subset]
        adaptive_sr = np.mean(adaptive_ok)
        regret = oracle_sr - adaptive_sr
        return adaptive_sr, regret, len(subset)

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print(f"\n=== Calibração no split TRAIN (n={len(train_rows)}) ===")
    print(f"{'tau':>6} {'sucesso':>9} {'regret':>8}")
    train_results = []
    for tau in TAUS:
        sr, regret, n = eval_tau(train_rows, tau)
        train_results.append((tau, sr, regret))
        print(f"{tau:>6.2f} {sr:>9.1%} {regret:>8.1%}")

    best_tau, best_sr_train, best_regret_train = min(train_results, key=lambda x: x[2])
    print(f"\ntau que MINIMIZA REGRET no split train: {best_tau:.2f} (regret={best_regret_train:.1%})")

    print(f"\n=== Avaliação no split TEST (n={len(test_rows)}), tau={best_tau:.2f} calibrado no train ===")
    sr_test, regret_test, n_test = eval_tau(test_rows, best_tau)
    print(f"  sucesso={sr_test:.1%}  regret={regret_test:.1%}")

    print(f"\n=== Comparação: tau=0.30 (valor atual do relatório) no split TEST ===")
    sr_030, regret_030, _ = eval_tau(test_rows, 0.30)
    print(f"  sucesso={sr_030:.1%}  regret={regret_030:.1%}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results_abstract", "threshold_sweep_real.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "split", "n", "success", "regret"])
        for split_name, subset in [("train", train_rows), ("test", test_rows)]:
            for tau in TAUS:
                sr, regret, n = eval_tau(subset, tau)
                writer.writerow([tau, split_name, n, f"{sr:.4f}", f"{regret:.4f}"])
    print(f"\nSalvo em {out_path}")

    print(f"\n=== RESUMO PARA O RELATÓRIO ===")
    print(f"tau recalibrado (A*/BC reais, split train/test): {best_tau:.2f}")
    print(f"tau=0.30 (valor atual): regret no test = {regret_030:.1%}")
    print(f"tau={best_tau:.2f} (recalibrado): regret no test = {regret_test:.1%}")
    if abs(best_tau - 0.30) <= 0.03:
        print(">>> CONFIRMA tau=0.30 dentro de faixa razoável (+/-0.03)")
    else:
        print(f">>> DIVERGE de tau=0.30 -- considerar revisão em cascata (ver Passo 4 do plano)")


if __name__ == "__main__":
    main()
