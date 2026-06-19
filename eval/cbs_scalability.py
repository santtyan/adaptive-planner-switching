"""
Curva de escalabilidade do CBS: tempo de solução vs número de agentes N.

Gera cenários aleatórios num grid 10×10 com ρ≈0.30 de obstáculos e mede
o tempo de solução do CBS para N ∈ {2,3,4,5,6,7,8} agentes.

O crescimento super-linear (exponencial no pior caso) justifica a decisão
local O(1) do ρ-criterion por agente e motiva MARL como trabalho futuro.

Timeouts (>TIMEOUT_S segundos) são registrados como ponto censurado, não
como falha do script — marcados com ✗ no gráfico e excluídos da média.

Gera: paper/figs/cbs_scalability.png + .pdf
"""

import os
import sys
import time
import random
import signal
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CBS_DIR = "/home/yan/Documentos/Projetos/multi_agent_path_planning/centralized"
sys.path.insert(0, CBS_DIR)

from cbs.cbs import CBS, Environment

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUT_PNG = os.path.join(ROOT, "paper", "figs", "cbs_scalability.png")

GRID_SIZE  = 10
N_AGENTS   = [2, 3, 4, 5, 6, 7, 8]
N_TRIALS   = 5
TIMEOUT_S  = 30
OBSTACLE_DENSITY = 0.20   # ρ≈0.20 → ~20 obstáculos em grid 10×10
SEED       = 42


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError()


def generate_scenario(n_agents, grid_size, obstacle_density, rng):
    """Gera cenário CBS aleatório com starts/goals distintos e sem obstáculos nas posições."""
    all_cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    n_obs = int(grid_size * grid_size * obstacle_density)

    # Posicionar obstáculos
    obs_cells = set()
    shuffled = all_cells[:]
    rng.shuffle(shuffled)
    for cell in shuffled:
        if len(obs_cells) >= n_obs:
            break
        obs_cells.add(cell)

    free_cells = [c for c in all_cells if c not in obs_cells]
    if len(free_cells) < 2 * n_agents:
        return None

    rng.shuffle(free_cells)
    positions = free_cells[: 2 * n_agents]
    starts = positions[:n_agents]
    goals  = positions[n_agents:]

    agents = [{"name": f"agent{i}", "start": list(starts[i]), "goal": list(goals[i])}
               for i in range(n_agents)]
    obstacles = [{"x": x, "y": y} for (x, y) in obs_cells]

    return {"agents": agents, "obstacles": obstacles}


def run_cbs(scenario, grid_size):
    """Executa CBS num cenário. Retorna (tempo_s, success)."""
    env = Environment(
        [grid_size, grid_size],
        scenario["agents"],
        scenario["obstacles"],
    )
    cbs = CBS(env)

    t0 = time.perf_counter()
    solution = cbs.search()
    elapsed = time.perf_counter() - t0

    return elapsed, solution is not None and len(solution) > 0


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    results = {}   # n → list of (time_s, censored)

    for n in N_AGENTS:
        times = []
        print(f"\nN={n} agentes:")
        for trial in range(N_TRIALS):
            scenario = None
            for _ in range(20):
                scenario = generate_scenario(n, GRID_SIZE, OBSTACLE_DENSITY, rng)
                if scenario is not None:
                    break
            if scenario is None:
                print(f"  trial {trial+1}: sem cenário válido — pulando")
                continue

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_S)
            try:
                elapsed, ok = run_cbs(scenario, GRID_SIZE)
                signal.alarm(0)
                status = "OK" if ok else "sem solução"
                print(f"  trial {trial+1}: {elapsed:.3f}s ({status})")
                times.append((elapsed, False))
            except TimeoutError:
                signal.alarm(0)
                print(f"  trial {trial+1}: TIMEOUT (>{TIMEOUT_S}s) — censurado")
                times.append((TIMEOUT_S, True))
            except Exception as e:
                signal.alarm(0)
                print(f"  trial {trial+1}: erro — {e}")

        results[n] = times

    # ── Estatísticas ──────────────────────────────────────────────────────────
    ns_plot     = []
    mean_times  = []
    std_times   = []
    timeout_ns  = []

    for n in N_AGENTS:
        valid = [t for (t, c) in results.get(n, []) if not c]
        censored = [t for (t, c) in results.get(n, []) if c]
        if censored:
            timeout_ns.append(n)
        if valid:
            ns_plot.append(n)
            mean_times.append(np.mean(valid))
            std_times.append(np.std(valid))
            print(f"N={n}: {np.mean(valid):.3f}s ±{np.std(valid):.3f}s  "
                  f"({len(censored)} timeout(s))")
        else:
            print(f"N={n}: todos censurados")

    ns_plot    = np.array(ns_plot)
    mean_times = np.array(mean_times)
    std_times  = np.array(std_times)

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(ns_plot, mean_times, yerr=std_times,
                fmt="o-", color="#1565C0", lw=2, markersize=8,
                capsize=5, label="CBS (média ±1σ, trials válidos)")

    # Linha de tendência exponencial (se pontos suficientes)
    if len(ns_plot) >= 3:
        try:
            log_t = np.log(mean_times + 1e-6)
            coeffs = np.polyfit(ns_plot, log_t, 1)
            n_fit = np.linspace(ns_plot[0], ns_plot[-1], 100)
            t_fit = np.exp(np.polyval(coeffs, n_fit))
            ax.plot(n_fit, t_fit, "--", color="#c62828", lw=1.5, alpha=0.7,
                    label=f"Tendência exp. (e^{{{coeffs[0]:.2f}·N}})")
        except Exception:
            pass

    # Marcar timeouts
    for n in timeout_ns:
        ax.axvline(n, color="#f57c00", linestyle=":", lw=1.2, alpha=0.7)
        ax.text(n + 0.05, ax.get_ylim()[1] * 0.85, "✗ timeout",
                color="#f57c00", fontsize=8, rotation=90, va="top")

    # Linha de referência: custo O(1) do ρ-criterion
    ax.axhline(0.001, color="#2e7d32", linestyle="--", lw=1.5, alpha=0.8,
               label="ρ-criterion por agente: O(1) ≈ 1 ms")

    ax.set_xlabel("Número de agentes N", fontsize=11)
    ax.set_ylabel("Tempo de solução (s)", fontsize=11)
    ax.set_title(
        f"Escalabilidade do CBS: crescimento exponencial com N agentes\n"
        f"(grid {GRID_SIZE}×{GRID_SIZE}, ρ={OBSTACLE_DENSITY}, timeout={TIMEOUT_S}s, "
        f"{N_TRIALS} trials/ponto)",
        fontsize=11, fontweight="bold"
    )
    ax.set_yscale("log")
    ax.set_xticks(N_AGENTS)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSalvo: {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
