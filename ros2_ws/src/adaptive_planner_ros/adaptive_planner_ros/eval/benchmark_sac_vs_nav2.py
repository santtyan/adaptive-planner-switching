"""
Benchmark comparativo: SAC vs Nav2/SmacPlanner2D no dense_custom.world.

Métricas por condição (N=30 trials, seeds pareadas):
  - Taxa de sucesso (goal reached)
  - Tempo médio até o goal (s)
  - Comprimento médio do path (m)
  - Taxa de colisão
  - Distância mínima a obstáculos (clearance)

Condições:
  - sac_only     : agente SAC puro
  - nav2_only    : SmacPlanner2D puro (requer Nav2 rodando)
  - adaptive     : switcher adaptativo (requer adaptive_switcher_node)

Uso (após treino convergir):
    python3 benchmark_sac_vs_nav2.py --model models/best_model.zip \\
        --condition sac_only --trials 30 --out results_ros2/benchmark.csv

Análise estatística (Wilcoxon + Holm-Bonferroni):
    python3 benchmark_sac_vs_nav2.py --analyze results_ros2/benchmark.csv
"""

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import rclpy

# ---------------------------------------------------------------------------
# Estrutura de resultado por trial
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_id: int
    seed: int
    condition: str
    success: bool
    duration_s: float
    path_length_m: float
    min_clearance_m: float
    collision: bool
    timeout: bool
    dist_to_goal_final: float


# ---------------------------------------------------------------------------
# Runner SAC
# ---------------------------------------------------------------------------

def run_sac_trial(model, env, seed: int, trial_id: int) -> TrialResult:
    from turtlebot3_gym_env.gazebo_gym_env import _GazeboEnvNode
    obs, info = env.reset(seed=seed)
    goal = info["goal"]

    xs, ys = [], []
    min_clearance = float("inf")
    t0 = time.time()
    done = False
    success = collision = timeout = False

    node = env.unwrapped._node

    while not done:
        x, y, _ = node.get_robot_pose()
        xs.append(x)
        ys.append(y)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        min_clearance = min(min_clearance, step_info.get("min_scan", float("inf")))
        if step_info.get("goal_reached"):
            success = True
        if step_info.get("collision"):
            collision = True
        if truncated and not terminated:
            timeout = True

    duration = time.time() - t0
    path_len = sum(math.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) for i in range(len(xs)-1))
    gx, gy = goal
    dist_final = math.hypot(gx - xs[-1], gy - ys[-1]) if xs else float("inf")

    return TrialResult(
        trial_id=trial_id, seed=seed, condition="sac_only",
        success=success, duration_s=duration, path_length_m=path_len,
        min_clearance_m=min_clearance, collision=collision, timeout=timeout,
        dist_to_goal_final=dist_final,
    )


# ---------------------------------------------------------------------------
# Análise estatística
# ---------------------------------------------------------------------------

def analyze(csv_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    conditions = sorted(set(r["condition"] for r in rows))
    print(f"\n{'='*60}")
    print(f"Benchmark: {csv_path}")
    print(f"Condições: {conditions}")
    print(f"{'='*60}\n")

    success_by_cond = {}
    duration_by_cond = {}
    for cond in conditions:
        subset = [r for r in rows if r["condition"] == cond]
        successes = [int(r["success"] == "True") for r in subset]
        durations = [float(r["duration_s"]) for r in subset if r["success"] == "True"]
        success_by_cond[cond] = successes
        duration_by_cond[cond] = durations
        sr = np.mean(successes) * 100
        dt = np.mean(durations) if durations else float("nan")
        print(f"{cond:20s} | success={sr:.1f}%  |  mean_time={dt:.1f}s  |  N={len(subset)}")

    # Wilcoxon signed-rank entre pares (se houver seeds pareadas)
    if len(conditions) >= 2:
        print("\nTeste Wilcoxon signed-rank (pares de seeds):")
        conds = list(conditions)
        p_values = []
        pairs = []
        for i in range(len(conds)):
            for j in range(i+1, len(conds)):
                a = success_by_cond[conds[i]]
                b = success_by_cond[conds[j]]
                n = min(len(a), len(b))
                if n > 0 and not np.all(np.array(a[:n]) == np.array(b[:n])):
                    try:
                        stat, p = stats.wilcoxon(a[:n], b[:n])
                        p_values.append(p)
                        pairs.append((conds[i], conds[j], p))
                    except Exception:
                        pass
        # Correção Holm-Bonferroni
        if p_values:
            sorted_pairs = sorted(pairs, key=lambda x: x[2])
            m = len(sorted_pairs)
            for k, (c1, c2, p) in enumerate(sorted_pairs):
                p_adj = min(p * (m - k), 1.0)
                sig = "✓ sig." if p_adj < 0.05 else "n.s."
                print(f"  {c1} vs {c2}: p={p:.4f}  p_adj={p_adj:.4f}  {sig}")

    # Figura: barras de taxa de sucesso
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Benchmark SAC vs Nav2 — dense_custom.world", fontweight="bold")

    srs = [np.mean(success_by_cond[c]) * 100 for c in conditions]
    axes[0].bar(conditions, srs, color=["steelblue", "darkorange", "seagreen"][:len(conditions)])
    axes[0].set_ylabel("Taxa de Sucesso (%)")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(srs):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    dts = [np.mean(duration_by_cond[c]) if duration_by_cond[c] else 0 for c in conditions]
    axes[1].bar(conditions, dts, color=["steelblue", "darkorange", "seagreen"][:len(conditions)])
    axes[1].set_ylabel("Tempo médio até goal (s)")

    out_fig = csv_path.replace(".csv", "_figure.png")
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"\nFigura salva: {out_fig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best_model.zip")
    p.add_argument("--condition", default="sac_only",
                   choices=["sac_only", "nav2_only", "adaptive"])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--out", default="results_ros2/benchmark.csv")
    p.add_argument("--analyze", metavar="CSV", help="Só analisa CSV existente")
    p.add_argument("--seed-start", type=int, default=0)
    args = p.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)

    rclpy.init()

    from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    node = _GazeboEnvNode()
    env = Monitor(TurtleBot3GazeboEnv(node=node))
    model = SAC.load(args.model, env=env)
    print(f"Modelo: {args.model} | condição: {args.condition} | trials: {args.trials}")

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialResult.__dataclass_fields__.keys()))
        if write_header:
            writer.writeheader()

        for i in range(args.trials):
            seed = args.seed_start + i
            print(f"Trial {i+1}/{args.trials} (seed={seed})...", end=" ", flush=True)
            result = run_sac_trial(model, env, seed, trial_id=i)
            writer.writerow(asdict(result))
            f.flush()
            status = "✓" if result.success else "✗"
            print(f"{status}  {result.duration_s:.1f}s  path={result.path_length_m:.2f}m")

    env.close()
    rclpy.shutdown()
    print(f"\nResultados salvos: {args.out}")
    analyze(args.out)


if __name__ == "__main__":
    main()
