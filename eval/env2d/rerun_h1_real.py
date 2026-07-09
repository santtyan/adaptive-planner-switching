"""
rerun_h1_real.py — Revalida H1 (o ρ-criterion supera qualquer planejador
fixo) usando planejadores REAIS no gêmeo 2D, substituindo o Monte Carlo
calibrado (MockRRTStarPlanner + PPOPlannerMockScientific) usado na Fase 1.

Compara, nos 3 mundos (sparse/dense/very_dense), em N trials cada:
  - A* real (astar_planner.py — busca em grade, heapq, sem mock)
  - SAC real (models/sac_2d_best.zip)
  - BC real (models/bc_2d_policy.pt)
  - ρ-criterion real: A* se ρ_local < 0,30, senão o melhor dos dois
    aprendizados (SAC ou BC, escolhido por --learned)

ρ_local é a mesma métrica usada no heatmap (visualize_2d.py):
  fração dos raios do LIDAR inicial com leitura < 1,0 m.

Uso:
    python3 eval/env2d/rerun_h1_real.py --trials 100 --learned bc
"""
import os, sys, argparse, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import Env2D, WORLDS, _scan
from eval.env2d.astar_planner import AStarPolicy

MODS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
RHO_STAR = 0.30


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def load_bc(world: str = None):
    """Carrega o BC específico do mundo (bc_2d_policy_<world>.pt) se existir
    — teste de capacidade (treino casado com a densidade de teste) em vez de
    generalização entre densidades. Cai para bc_2d_policy.pt (treinado só em
    sparse) se o arquivo específico não existir."""
    name = f"bc_2d_policy_{world}.pt" if world else "bc_2d_policy.pt"
    path = os.path.join(MODS, name)
    if not os.path.exists(path):
        path = os.path.join(MODS, "bc_2d_policy.pt")
    state = torch.load(path, map_location="cpu")
    obs_dim = state["net.0.weight"].shape[1]
    act_dim = state["net.4.weight"].shape[0]
    policy = BCPolicy(obs_dim, act_dim)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def load_sac():
    from stable_baselines3 import SAC
    path = os.path.join(MODS, "sac_2d_best")
    if not os.path.exists(path + ".zip"):
        return None
    return SAC.load(path, device="cpu")


def local_rho(env: Env2D) -> float:
    """Mesma definição usada no heatmap de switching (visualize_2d.py)."""
    ranges = _scan(env._x, env._y, env._yaw, env.obstacles, env.arena)
    return float(np.mean(ranges < 1.0))


def rollout(env: Env2D, method: str, astar_policy: AStarPolicy,
            sac_model, bc_policy, learned: str) -> bool:
    """Roda 1 episódio com o método indicado; retorna se atingiu o goal."""
    if method == "astar":
        astar_policy.reset(env)
    elif method == "adaptive":
        rho0 = local_rho(env)
        use_astar = rho0 < RHO_STAR
        if use_astar:
            astar_policy.reset(env)

    obs = env._obs()
    done = False
    steps = 0
    while not done and steps < 200:
        if method == "astar":
            a = astar_policy.act(env)
        elif method == "sac":
            a, _ = sac_model.predict(obs, deterministic=True)
        elif method == "bc":
            with torch.no_grad():
                a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        elif method == "adaptive":
            if use_astar:
                a = astar_policy.act(env)
            elif learned == "sac":
                a, _ = sac_model.predict(obs, deterministic=True)
            else:
                with torch.no_grad():
                    a = bc_policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
    return bool(info.get("goal_reached", False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100, help="trials por mundo")
    p.add_argument("--learned", choices=["sac", "bc"], default="bc",
                    help="qual política aprendida usar no ρ-criterion e no baseline 'learned fixo'")
    p.add_argument("--seed0", type=int, default=1000)
    args = p.parse_args()

    astar_policy = AStarPolicy()
    sac_model = load_sac()

    if sac_model is None:
        print("Aviso: models/sac_2d_best.zip não encontrado — pulando método 'sac' isolado.")

    methods = ["astar", "bc", "adaptive"]
    if sac_model is not None:
        methods.insert(1, "sac")

    results = {m: {} for m in methods}
    t0 = time.time()
    for world in ["sparse", "dense", "very_dense"]:
        bc_policy = load_bc(world)  # BC casado com a densidade do mundo testado
        for method in methods:
            env = Env2D(world=world, seed=args.seed0)
            successes = 0
            for trial in range(args.trials):
                env.reset(seed=args.seed0 + trial)
                ok = rollout(env, method, astar_policy, sac_model, bc_policy, args.learned)
                successes += int(ok)
            sr = successes / args.trials
            results[method][world] = sr
            print(f"[{world:>10}] {method:>9}: {successes:>4}/{args.trials} = {sr:.1%}")

    print(f"\nTempo total: {time.time()-t0:.1f}s")
    print("\n=== Resumo (média entre os 3 mundos) ===")
    for m in methods:
        vals = list(results[m].values())
        print(f"  {m:>9}: {np.mean(vals):.1%}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "results_abstract", "h1_real_2d_validation.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("method,world,success_rate,n_trials\n")
        for m in methods:
            for w, sr in results[m].items():
                f.write(f"{m},{w},{sr:.4f},{args.trials}\n")
    print(f"\nSalvo em {out_path}")


if __name__ == "__main__":
    main()
