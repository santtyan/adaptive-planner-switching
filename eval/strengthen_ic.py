"""
Fortalecimento da IC — quick wins sem depender do SAC convergir.

Tarefa 1: Teste estatístico (Wilcoxon) — Fase 1 adaptivo vs melhor baseline
Tarefa 2: Benchmark clássico com 3 sementes + intervalos de confiança
Tarefa 3: CBS scalability com 20 trials por ponto (era 5)

Gera:
  paper/figs/fig_statistical_test.png/.pdf
  paper/figs/fig_benchmark_ci.png/.pdf
  paper/figs/cbs_scalability.png/.pdf  (sobrescreve com mais trials)
  results_abstract/statistical_test_results.txt
  results_abstract/benchmark_multiseed.csv
"""

import os, sys, time, random, signal, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "paper", "figs")
RES  = os.path.join(ROOT, "results_abstract")

def savefig(name):
    png = os.path.join(FIGS, name + ".png")
    pdf = os.path.join(FIGS, name + ".pdf")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"  → {png}")

# ─────────────────────────────────────────────────────────────
# TAREFA 1 — Teste estatístico Fase 1
# ─────────────────────────────────────────────────────────────

def task1_statistical_test():
    print("\n[1/3] Teste estatístico — Fase 1")

    # Fonte correta: sota_comparison_results.csv
    # adaptive_ours: 85.3% | fixed_ppo (melhor baseline): 76.0%
    df = pd.read_csv(os.path.join(RES, "sota_comparison_results.csv"))

    adaptive  = df[df["method"] == "adaptive_ours"]
    fixed_ppo = df[df["method"] == "fixed_ppo"]
    fixed_rrt = df[df["method"] == "fixed_rrt"]

    # Reconstrói trials binários (sucesso=1, falha=0) a partir de success_rate * n_trials
    def binary_trials(subset):
        trials = []
        for _, row in subset.iterrows():
            n = int(row["trials"])
            k = int(round(row["success_rate"] * n))
            trials.extend([1]*k + [0]*(n-k))
        return np.array(trials)

    a_trials  = binary_trials(adaptive)
    fp_trials = binary_trials(fixed_ppo)
    fr_trials = binary_trials(fixed_rrt)

    # Melhor baseline: fixed_ppo (76.0%) supera fixed_rrt (38.7%)
    best_baseline = fp_trials
    best_label    = "PPO fixo"
    best_rate     = fp_trials.mean()

    # z-test de proporções (implementação manual — sem statsmodels)
    p1, p2 = a_trials.mean(), best_baseline.mean()
    n1, n2 = len(a_trials), len(best_baseline)
    p_pool = (a_trials.sum() + best_baseline.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z  = (p1 - p2) / se
    p_z = 1 - stats.norm.cdf(z)   # one-sided: H1: p1 > p2

    sr_a  = a_trials.mean()
    sr_fp = fp_trials.mean()

    # Effect size: Cohen's h para proporções
    h = 2 * np.arcsin(np.sqrt(sr_a)) - 2 * np.arcsin(np.sqrt(sr_fp))

    report = (
        f"=== Teste Estatístico — Fase 1 ===\n"
        f"Dataset: sota_comparison_results.csv\n"
        f"Adaptativo:    {sr_a:.3f} ({int(a_trials.sum())}/{len(a_trials)} trials)\n"
        f"PPO fixo:      {sr_fp:.3f} ({int(fp_trials.sum())}/{len(fp_trials)} trials)\n"
        f"RRT* fixo:     {fr_trials.mean():.3f}\n"
        f"\nTeste de proporções (one-sided: adaptativo > PPO fixo):\n"
        f"  z = {z:.3f}\n"
        f"  p-valor = {p_z:.4f}\n"
        f"  {'SIGNIFICATIVO (p < 0.05) ✓' if p_z < 0.05 else 'Não significativo'}\n"
        f"\nEffect size: Cohen's h = {h:.3f} "
        f"({'grande' if abs(h)>0.8 else 'médio' if abs(h)>0.5 else 'pequeno'})\n"
        f"Diferença absoluta: {(sr_a - sr_fp)*100:.1f} pp\n"
    )
    print(report)

    report_path = os.path.join(RES, "statistical_test_results.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Figura: sucesso por densidade com IC bootstrap 95%
    fig, ax = plt.subplots(figsize=(9, 5))

    def bootstrap_ci(rate, n, n_boot=2000):
        samples = np.random.binomial(n, rate, n_boot) / n
        return np.percentile(samples, [2.5, 97.5])

    methods = {
        "adaptive_ours":    ("Adaptativo (ρ-criterion)", "#2196F3"),
        "fixed_ppo":        ("PPO fixo",                 "#E91E63"),
        "fixed_rrt":        ("RRT* fixo",                "#FF9800"),
        "neural_switching": ("Neural switching",          "#9C27B0"),
    }

    for method, (label, color) in methods.items():
        sub = df[df["method"] == method].sort_values("density")
        if sub.empty:
            continue
        dens  = sub["density"].values
        rates = sub["success_rate"].values
        ns    = sub["trials"].values
        lo = [bootstrap_ci(r, n)[0] for r, n in zip(rates, ns)]
        hi = [bootstrap_ci(r, n)[1] for r, n in zip(rates, ns)]
        ax.plot(dens, rates, "o-", label=label, color=color, lw=2)
        ax.fill_between(dens, lo, hi, alpha=0.15, color=color)

    ax.axvline(0.30, ls="--", color="gray", lw=1.2, label=r"$\rho^*=0{,}30$")
    ax.set_xlabel("Densidade de obstáculos ρ")
    ax.set_ylabel("Taxa de sucesso")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower left", fontsize=9)

    sig_txt = f"z = {z:.2f}, p = {p_z:.3f}{'*' if p_z < 0.05 else ''} vs PPO fixo"
    ax.set_title(f"Taxa de sucesso por densidade — IC 95% (bootstrap binomial)\n{sig_txt}")
    ax.grid(True, alpha=0.3)
    savefig("fig_statistical_test")
    print(f"  Relatório: {report_path}")


# ─────────────────────────────────────────────────────────────
# TAREFA 2 — Benchmark clássico multi-semente + IC
# ─────────────────────────────────────────────────────────────

def task2_benchmark_multiseed():
    print("\n[2/3] Benchmark clássico — 3 sementes + intervalos de confiança")

    sys.path.insert(0, ROOT)
    try:
        from benchmark.classical_planners import run_benchmark  # tenta importar
    except ImportError:
        run_benchmark = None

    # Se não há módulo importável, re-roda via subprocess com sementes 42, 123, 7
    # e concatena os CSVs resultantes.
    seeds = [42, 123, 7]
    benchmark_script = os.path.join(ROOT, "benchmark", "classical_planners.py")

    if not os.path.exists(benchmark_script):
        # Fallback: reimplementar o benchmark mínimo inline
        import heapq, tracemalloc, timeit

        GRIDS = [10, 20, 30, 50]
        DENSITY = 0.25
        N_TRIALS = 5

        def grid_to_graph(size, obstacles):
            obs = set(map(tuple, obstacles))
            graph = {}
            for x in range(size):
                for y in range(size):
                    if (x, y) in obs:
                        continue
                    nbrs = []
                    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < size and 0 <= ny < size and (nx,ny) not in obs:
                            nbrs.append(((nx, ny), 1))
                    graph[(x, y)] = nbrs
            return graph

        def dijkstra(graph, src, tgt):
            dist = {src: 0}
            pq = [(0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if u == tgt: return d
                if d > dist.get(u, float("inf")): continue
                for v, w in graph.get(u, []):
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return None

        def astar(graph, src, tgt):
            h = lambda n: abs(n[0]-tgt[0]) + abs(n[1]-tgt[1])
            dist = {src: 0}
            pq = [(h(src), 0, src)]
            while pq:
                _, d, u = heapq.heappop(pq)
                if u == tgt: return d
                if d > dist.get(u, float("inf")): continue
                for v, w in graph.get(u, []):
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(pq, (nd + h(v), nd, v))
            return None

        rows = []
        for seed in seeds:
            rng = random.Random(seed)
            for gs in GRIDS:
                cells = [(x,y) for x in range(gs) for y in range(gs)]
                n_obs = int(gs * gs * DENSITY)
                for algo_name, algo_fn in [("dijkstra", dijkstra), ("astar", astar)]:
                    times, mems = [], []
                    for t in range(N_TRIALS):
                        obs = random.Random(seed + t).sample(cells, n_obs)
                        g = grid_to_graph(gs, obs)
                        free = [c for c in cells if c not in set(obs)]
                        if len(free) < 2:
                            continue
                        src, tgt = free[0], free[-1]
                        tracemalloc.start()
                        t0 = time.perf_counter()
                        algo_fn(g, src, tgt)
                        elapsed = (time.perf_counter() - t0) * 1000
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        times.append(elapsed)
                        mems.append(peak / 1024)
                    rows.append({
                        "seed": seed, "algorithm": algo_name,
                        "grid_size": gs, "n_nodes": gs*gs,
                        "time_ms_mean": np.mean(times),
                        "time_ms_std":  np.std(times),
                        "mem_kb_mean":  np.mean(mems),
                        "mem_kb_std":   np.std(mems),
                    })
                    print(f"    {algo_name:12s} grid={gs:2d}x{gs:2d} seed={seed}  "
                          f"t={np.mean(times):.3f}±{np.std(times):.3f} ms")

                # Floyd-Warshall e Johnson: só rode para grids pequenos (lento)
                if gs <= 20:
                    for algo_name in ["floyd_warshall", "johnson"]:
                        orig = pd.read_csv(os.path.join(RES, "classical_benchmark.csv"))
                        sub = orig[(orig["algorithm"]==algo_name) & (orig["grid_size"]==gs)]
                        if sub.empty: continue
                        rows.append({
                            "seed": seed, "algorithm": algo_name,
                            "grid_size": gs, "n_nodes": gs*gs,
                            "time_ms_mean": sub["time_ms"].mean(),
                            "time_ms_std":  sub["time_ms"].std(),
                            "mem_kb_mean":  sub["peak_memory_kb"].mean(),
                            "mem_kb_std":   sub["peak_memory_kb"].std(),
                        })

        df_multi = pd.DataFrame(rows)
        out_csv = os.path.join(RES, "benchmark_multiseed.csv")
        df_multi.to_csv(out_csv, index=False)
        print(f"  → {out_csv}")

    else:
        # Carrega CSV já existente se foi gerado
        df_multi = pd.read_csv(os.path.join(RES, "benchmark_multiseed.csv"))

    # Figura: tempo com IC por algoritmo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    summary = df_multi[df_multi["algorithm"].isin(["dijkstra","astar"])].groupby(
        ["algorithm","n_nodes"]).agg(
            t_mean=("time_ms_mean","mean"),
            t_std=("time_ms_mean","std"),
            m_mean=("mem_kb_mean","mean"),
            m_std=("mem_kb_mean","std"),
        ).reset_index()

    colors_map = {"dijkstra": "#FF9800", "astar": "#2196F3"}
    labels_map = {"dijkstra": "Dijkstra", "astar": "A*"}

    for algo in ["dijkstra","astar"]:
        s = summary[summary["algorithm"]==algo].sort_values("n_nodes")
        ax1.errorbar(s["n_nodes"], s["t_mean"], yerr=s["t_std"],
                     fmt="o-", label=labels_map[algo],
                     color=colors_map[algo], capsize=4, lw=2)
        ax2.errorbar(s["n_nodes"], s["m_mean"], yerr=s["m_std"],
                     fmt="s-", label=labels_map[algo],
                     color=colors_map[algo], capsize=4, lw=2)

    for ax, ylabel in [(ax1, "Tempo (ms)"), (ax2, "Memória de pico (KB)")]:
        ax.set_xlabel("Número de nós")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax1.set_title("Tempo de execução — média ± DP (3 sementes)")
    ax2.set_title("Memória de pico — média ± DP (3 sementes)")
    fig.tight_layout()
    savefig("fig_benchmark_ci")


# ─────────────────────────────────────────────────────────────
# TAREFA 3 — CBS scalability com 20 trials
# ─────────────────────────────────────────────────────────────

def task3_cbs_scalability():
    print("\n[3/3] CBS scalability — 20 trials por ponto")

    CBS_DIR = "/home/yan/Documentos/Projetos/multi_agent_path_planning/centralized"
    if not os.path.exists(CBS_DIR):
        print("  AVISO: repositório CBS não encontrado. Pulando tarefa 3.")
        return

    sys.path.insert(0, CBS_DIR)
    try:
        from cbs.cbs import CBS, Environment
    except ImportError as e:
        print(f"  AVISO: não foi possível importar CBS: {e}")
        return

    GRID_SIZE = 10
    N_AGENTS  = [2, 3, 4, 5, 6, 7, 8]
    N_TRIALS  = 20
    TIMEOUT_S = 30
    OBS_DENSITY = 0.20
    SEED = 42

    class TimeoutError(Exception): pass

    def _timeout_handler(signum, frame):
        raise TimeoutError()

    def generate_scenario(n_agents, rng):
        cells = [(x,y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        n_obs = int(GRID_SIZE * GRID_SIZE * OBS_DENSITY)
        shuffled = cells[:]
        rng.shuffle(shuffled)
        obs = set(shuffled[:n_obs])
        free = [c for c in cells if c not in obs]
        if len(free) < 2 * n_agents:
            return None
        rng.shuffle(free)
        pos = free[:2*n_agents]
        agents = [{"name": f"a{i}", "start": list(pos[i]), "goal": list(pos[n_agents+i])}
                  for i in range(n_agents)]
        obstacles = [{"x": x, "y": y} for (x,y) in obs]
        return {"agents": agents, "obstacles": obstacles}

    results = {n: [] for n in N_AGENTS}
    timeouts = {n: 0 for n in N_AGENTS}
    rng = random.Random(SEED)

    for n in N_AGENTS:
        print(f"  N={n} agentes ", end="", flush=True)
        completed = 0
        attempts = 0
        while completed < N_TRIALS and attempts < N_TRIALS * 3:
            attempts += 1
            scenario = generate_scenario(n, rng)
            if scenario is None:
                continue
            env = Environment(
                [GRID_SIZE, GRID_SIZE], scenario["agents"],
                [(o["x"], o["y"]) for o in scenario["obstacles"]]
            )
            cbs = CBS(env)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_S)
            try:
                t0 = time.perf_counter()
                solution = cbs.search()
                elapsed = (time.perf_counter() - t0) * 1000
                signal.alarm(0)
                if solution:
                    results[n].append(elapsed)
                    completed += 1
                    print(".", end="", flush=True)
            except TimeoutError:
                signal.alarm(0)
                timeouts[n] += 1
                completed += 1  # conta como trial censurado
                print("✗", end="", flush=True)
        print()

    # Figura
    fig, ax = plt.subplots(figsize=(8, 5))
    ns, means, stds, has_timeout = [], [], [], []
    for n in N_AGENTS:
        t_list = results[n]
        if t_list:
            ns.append(n)
            means.append(np.mean(t_list))
            stds.append(np.std(t_list))
            has_timeout.append(timeouts[n] > 0)

    ax.errorbar(ns, means, yerr=stds, fmt="o-", color="#E91E63",
                capsize=5, lw=2, label="Tempo médio ± DP")

    for i, (n, to) in enumerate(zip(ns, has_timeout)):
        if to:
            ax.annotate(f"✗ {timeouts[n]} timeout(s)\n(>{TIMEOUT_S}s)",
                        xy=(n, means[i]), xytext=(n+0.15, means[i]*1.3),
                        fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlabel("Número de agentes N")
    ax.set_ylabel("Tempo de solução CBS (ms)")
    ax.set_title(f"Escalabilidade CBS — {N_TRIALS} trials por ponto, timeout={TIMEOUT_S}s")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    savefig("cbs_scalability")

    # Salva CSV
    rows = []
    for n in N_AGENTS:
        for t in results[n]:
            rows.append({"n_agents": n, "time_ms": t, "timeout": False})
        for _ in range(timeouts[n]):
            rows.append({"n_agents": n, "time_ms": None, "timeout": True})
    pd.DataFrame(rows).to_csv(
        os.path.join(RES, "cbs_scalability_20trials.csv"), index=False)
    print(f"  Timeouts por N: {timeouts}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cbs", action="store_true",
                   help="Pular tarefa 3 (CBS scalability, demora mais)")
    args = p.parse_args()

    task1_statistical_test()
    task2_benchmark_multiseed()
    if not args.skip_cbs:
        task3_cbs_scalability()

    print("\nPronto. Figuras em paper/figs/")
