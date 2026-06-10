"""
benchmark_classical.py — micro-benchmark dos algoritmos clássicos.

Mede tempo de execução (ms) e uso de memória (MB) para Dijkstra, A*,
Floyd-Warshall e Johnson sobre grids de tamanhos crescentes.

Exigência do Plano de Trabalho SIGAA PI08078-2024/1:
  "Realizar comparações entre os métodos clássicos e modernos,
   com base em métricas como tempo de execução e eficiência computacional."

Parecer do consultor (13/06/2025):
  "seria pertinente a comparação com implementações otimizadas de tais métodos"

Saída: results_abstract/classical_benchmark.csv + tabela no terminal.

Usage:
    python3 validation_abstract/benchmark_classical.py
    python3 validation_abstract/benchmark_classical.py --sizes 10 20 30 50
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import random
import time
import tracemalloc
from dataclasses import dataclass, asdict, fields
from typing import List

import sys
sys.path.insert(0, os.path.dirname(__file__))

from algorithms.classical import (
    dijkstra,
    astar,
    build_grid_heuristics,
    floyd_warshall,
    johnson,
    grid_to_graph,
)

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    algorithm:       str
    grid_size:       int          # N for N×N grid
    n_nodes:         int
    density:         float        # obstacle density [0,1]
    trial:           int
    time_ms:         float        # wall-clock
    peak_memory_kb:  float        # tracemalloc peak
    path_length:     float        # optimal path length (0 if not applicable)
    reachable:       bool         # whether source→target had a path


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------

def make_random_grid(size: int, density: float, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    grid = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if (r, c) in ((0, 0), (size - 1, size - 1)):
                continue  # keep source and target free
            if rng.random() < density:
                grid[r][c] = 1
    return grid


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_dijkstra(graph, n_nodes: int, source: int, target: int) -> tuple:
    tracemalloc.start()
    t0 = time.perf_counter()
    dist, prev = dijkstra(graph, source, n_nodes)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    path_len = dist[target] if dist[target] < math.inf else 0.0
    return elapsed_ms, peak / 1024, path_len, dist[target] < math.inf


def bench_astar(graph, n_nodes: int, source: int, target: int,
                width: int) -> tuple:
    h = build_grid_heuristics(n_nodes, target, width)
    tracemalloc.start()
    t0 = time.perf_counter()
    cost, path = astar(graph, source, target, h)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    path_len = cost if cost < math.inf else 0.0
    return elapsed_ms, peak / 1024, path_len, cost < math.inf


def bench_floyd_warshall(grid: List[List[int]], n_nodes: int,
                         source: int, target: int) -> tuple:
    size = len(grid)
    # Build dense weight matrix (required by FW)
    INF = float("inf")
    W = [[INF] * n_nodes for _ in range(n_nodes)]
    for r in range(size):
        for c in range(size):
            u = r * size + c
            W[u][u] = 0.0
            if grid[r][c] == 1:
                continue
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                    v = nr * size + nc
                    W[u][v] = math.hypot(dr, dc)

    tracemalloc.start()
    t0 = time.perf_counter()
    dist, _ = floyd_warshall(W)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    path_len = dist[source][target] if dist[source][target] < math.inf else 0.0
    return elapsed_ms, peak / 1024, path_len, dist[source][target] < math.inf


def bench_johnson(graph, n_nodes: int, source: int, target: int) -> tuple:
    tracemalloc.start()
    t0 = time.perf_counter()
    result = johnson(graph, n_nodes)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if result is None:
        return elapsed_ms, peak / 1024, 0.0, False
    d = result[source].get(target, math.inf)
    path_len = d if d < math.inf else 0.0
    return elapsed_ms, peak / 1024, path_len, d < math.inf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes",    nargs="+", type=int,
                   default=[10, 20, 30, 50],
                   help="Grid side lengths to benchmark (N×N)")
    p.add_argument("--density",  type=float, default=0.25,
                   help="Obstacle density [0,1]")
    p.add_argument("--trials",   type=int, default=5,
                   help="Trials per (algorithm, size) for averaging")
    p.add_argument("--skip-fw-above", type=int, default=30,
                   help="Skip Floyd-Warshall for grids larger than this "
                        "(O(n^3) becomes very slow for large sizes)")
    p.add_argument("--skip-johnson-above", type=int, default=30,
                   help="Skip Johnson for grids larger than this")
    p.add_argument("--output",   default="results_abstract/classical_benchmark.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results: List[BenchResult] = []

    for size in args.sizes:
        n_nodes = size * size
        source = 0
        target = n_nodes - 1
        print(f"\nGrid {size}×{size}  ({n_nodes} nodes, density={args.density})")

        for trial in range(args.trials):
            seed = trial * 1000 + size
            grid = make_random_grid(size, args.density, seed)
            graph, _ = grid_to_graph(grid, four_connected=True)
            gc.collect()

            # Dijkstra
            t, mem, plen, reach = bench_dijkstra(graph, n_nodes, source, target)
            results.append(BenchResult("dijkstra", size, n_nodes, args.density,
                                       trial, t, mem, plen, reach))
            print(f"  dijkstra   t={t:7.3f}ms  mem={mem:7.1f}KB  "
                  f"path={plen:.2f}  reach={reach}")

            # A*
            t, mem, plen, reach = bench_astar(graph, n_nodes, source, target, size)
            results.append(BenchResult("astar", size, n_nodes, args.density,
                                       trial, t, mem, plen, reach))
            print(f"  a_star     t={t:7.3f}ms  mem={mem:7.1f}KB  "
                  f"path={plen:.2f}  reach={reach}")

            # Floyd-Warshall (skip for large grids)
            if size <= args.skip_fw_above:
                t, mem, plen, reach = bench_floyd_warshall(grid, n_nodes,
                                                            source, target)
                results.append(BenchResult("floyd_warshall", size, n_nodes,
                                           args.density, trial, t, mem, plen, reach))
                print(f"  floyd_wsh  t={t:7.3f}ms  mem={mem:7.1f}KB  "
                      f"path={plen:.2f}  reach={reach}")
            else:
                print(f"  floyd_wsh  SKIPPED (size > {args.skip_fw_above})")

            # Johnson (skip for large grids)
            if size <= args.skip_johnson_above:
                t, mem, plen, reach = bench_johnson(graph, n_nodes, source, target)
                results.append(BenchResult("johnson", size, n_nodes, args.density,
                                           trial, t, mem, plen, reach))
                print(f"  johnson    t={t:7.3f}ms  mem={mem:7.1f}KB  "
                      f"path={plen:.2f}  reach={reach}")
            else:
                print(f"  johnson    SKIPPED (size > {args.skip_johnson_above})")

    # Write CSV
    fieldnames = [f.name for f in fields(BenchResult)]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nSaved {len(results)} rows → {args.output}")

    # Summary table: mean time per (algorithm, grid_size)
    import pandas as pd
    df = pd.read_csv(args.output)
    summary = (df.groupby(["algorithm", "grid_size"])
                 .agg(time_ms_mean=("time_ms", "mean"),
                      time_ms_std=("time_ms", "std"),
                      peak_memory_kb_mean=("peak_memory_kb", "mean"),
                      reach_rate=("reachable", "mean"))
                 .reset_index())
    print("\nSummary (mean ± std over trials):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
