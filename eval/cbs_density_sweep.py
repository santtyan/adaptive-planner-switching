"""
Varredura sistemática: densidade de obstáculos × % SAC/PPO ativado pelo ρ-criterion.

Gera cenários CBS sintéticos com densidades crescentes (ρ ∈ [0.05, 0.55]),
roda CBS + adaptive_annotator e plota a curva de transição de regime.

Gera: paper/figs/cbs_density_sweep.png
"""
import os
import sys
import subprocess
import tempfile
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

CBS_DIR = (
    "/home/yan/Documentos/Projetos/multi_agent_path_planning/centralized/cbs"
)
OUT_PNG = (
    "/home/yan/Documentos/Projetos/adaptive-planner-switching"
    "/paper/figs/cbs_density_sweep.png"
)
THRESHOLD = 0.30
GRID      = 10      # grid 10×10
N_AGENTS  = [2, 3, 5]
N_TRIALS  = 8       # tentativas por ponto de densidade
random.seed(42)

def make_scenario(n_obs, n_agents, grid=GRID):
    """Gera cenário aleatório com n_obs obstáculos e n_agents agentes."""
    all_cells = [(x, y) for x in range(grid) for y in range(grid)]
    random.shuffle(all_cells)

    obs_cells = all_cells[:n_obs]
    free_cells = [c for c in all_cells if c not in obs_cells]

    if len(free_cells) < n_agents * 2:
        return None

    random.shuffle(free_cells)
    agents = []
    for i in range(n_agents):
        s = free_cells[i * 2]
        g = free_cells[i * 2 + 1]
        if s == g:
            continue
        agents.append({"name": f"agent{i}", "start": list(s), "goal": list(g)})

    if len(agents) < n_agents:
        return None

    return {
        "agents": agents,
        "map": {
            "dimensions": [grid, grid],
            "obstacles": [list(o) for o in obs_cells]
        }
    }

def run_cbs_and_annotate(scenario):
    """Roda CBS + annotator em cenário temporário. Retorna ppo_ratio ou None."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fin:
        # converter para formato com !!python/tuple nos obstacles
        d = dict(scenario)
        d["map"] = dict(scenario["map"])
        d["map"]["obstacles"] = [
            f"!!python/tuple {o}" for o in scenario["map"]["obstacles"]
        ]
        fin_path = fin.name
        yaml.dump(scenario, fin)

    fout_path = fin_path.replace(".yaml", "_out.yaml")
    fann_path = fin_path.replace(".yaml", "_ann.yaml")

    try:
        # CBS
        r = subprocess.run(
            ["python3", "cbs.py", fin_path, fout_path],
            cwd=CBS_DIR, capture_output=True, timeout=30
        )
        if r.returncode != 0 or b"solution found" not in r.stdout:
            return None

        # Annotator
        r2 = subprocess.run(
            ["python3", "adaptive_annotator.py", fin_path, fout_path, fann_path],
            cwd=CBS_DIR, capture_output=True, timeout=10
        )
        if r2.returncode != 0:
            return None

        with open(fann_path) as f:
            ann = yaml.full_load(f)

        stats = ann.get("planner_stats", {})
        rrt   = stats.get("rrt_star", 0)
        ppo   = stats.get("ppo", 0)
        total = rrt + ppo
        return ppo / total if total > 0 else 0.0

    except Exception:
        return None
    finally:
        for p in [fin_path, fout_path, fann_path]:
            try:
                os.remove(p)
            except Exception:
                pass

def main():
    max_obs   = GRID * GRID - 4          # deixar células livres
    densities = np.linspace(0.04, 0.52, 13)
    n_obs_arr = [max(1, int(d * GRID * GRID)) for d in densities]

    results = {n: {"rho": [], "ppo_mean": [], "ppo_std": []} for n in N_AGENTS}

    for n_agents in N_AGENTS:
        print(f"\n── {n_agents} agentes ──")
        for rho_target, n_obs in zip(densities, n_obs_arr):
            ratios = []
            for _ in range(N_TRIALS):
                sc = make_scenario(n_obs, n_agents)
                if sc is None:
                    continue
                ratio = run_cbs_and_annotate(sc)
                if ratio is not None:
                    ratios.append(ratio)

            if ratios:
                mean = np.mean(ratios) * 100
                std  = np.std(ratios)  * 100
            else:
                mean = std = np.nan

            results[n_agents]["rho"].append(rho_target)
            results[n_agents]["ppo_mean"].append(mean)
            results[n_agents]["ppo_std"].append(std)
            print(f"  ρ={rho_target:.2f} ({n_obs} obs, {len(ratios)} trials): "
                  f"PPO={mean:.1f}±{std:.1f}%")

    # ── Figura ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {2: "#1565C0", 3: "#2e7d32", 5: "#c62828"}
    markers = {2: "o", 3: "s", 5: "^"}

    for n in N_AGENTS:
        rho  = np.array(results[n]["rho"])
        mean = np.array(results[n]["ppo_mean"])
        std  = np.array(results[n]["ppo_std"])
        mask = ~np.isnan(mean)
        ax.plot(rho[mask], mean[mask], f"{markers[n]}-",
                color=colors[n], linewidth=2, markersize=7,
                label=f"{n} agentes")
        ax.fill_between(rho[mask],
                        np.clip(mean[mask] - std[mask], 0, 100),
                        np.clip(mean[mask] + std[mask], 0, 100),
                        color=colors[n], alpha=0.12)

    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Limiar ρ* = {THRESHOLD}")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1,
               label="50% (regime balanceado)")

    ax.set_xlabel("Densidade de obstáculos ρ (fração do grid)", fontsize=11)
    ax.set_ylabel("% passos com SAC/PPO ativado", fontsize=11)
    ax.set_title(
        "Transição de Regime: densidade × ativação do planejador RL\n"
        f"(grid {GRID}×{GRID}, {N_TRIALS} trials por ponto, ρ* = {THRESHOLD})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.56)
    ax.set_ylim(-2, 80)
    ax.grid(alpha=0.3)

    ax.annotate("Regime A*\n(clássico domina)",
                xy=(0.10, 5), fontsize=9, color="#1565C0",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#1565C0", alpha=0.8))
    ax.annotate("Regime SAC\n(RL domina)",
                xy=(0.45, 55), fontsize=9, color="#c62828",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c62828", alpha=0.8))
    ax.annotate("Fronteira\nde fase",
                xy=(THRESHOLD, 38), xytext=(THRESHOLD + 0.05, 60),
                fontsize=8.5, color="black",
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.9))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSalvo: {OUT_PNG}")
    plt.close()

if __name__ == "__main__":
    main()
