"""
Classical graph search algorithms for trajectory planning.

These implementations serve as theoretical baselines in the IC evaluation.
All algorithms operate on adjacency representations of the same grid
environments used in validation_abstract/environment.py.

References:
  - Cormen et al., "Introduction to Algorithms", 4th ed. (A*, Dijkstra, FW, Johnson)
  - Hart et al. 1968 (A*)
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

INF = float("inf")


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------

def dijkstra(
    graph: Dict[int, List[Tuple[int, float]]],
    source: int,
    n_nodes: int,
) -> Tuple[List[float], List[Optional[int]]]:
    """Single-source shortest paths (non-negative weights).

    Returns (dist, prev) where dist[v] is shortest distance from source to v,
    and prev[v] is the predecessor on the shortest path (None if unreachable).
    """
    dist = [INF] * n_nodes
    prev: List[Optional[int]] = [None] * n_nodes
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev: List[Optional[int]], source: int, target: int) -> List[int]:
    """Reconstruct node path from predecessor array."""
    path: List[int] = []
    cur: Optional[int] = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]
    if not path or path[-1] != source:
        return []  # unreachable
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def astar(
    graph: Dict[int, List[Tuple[int, float]]],
    source: int,
    target: int,
    heuristic: Dict[int, float],
) -> Tuple[float, List[int]]:
    """A* shortest path.

    Args:
        heuristic: admissible h(v) estimate of cost from v to target.

    Returns (cost, path). cost=INF and path=[] if unreachable.
    """
    g: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    # heap entries: (f, g, node)
    pq: List[Tuple[float, float, int]] = [(heuristic.get(source, 0.0), 0.0, source)]

    while pq:
        f, cost, u = heapq.heappop(pq)
        if u == target:
            # reconstruct
            path: List[int] = []
            cur: Optional[int] = target
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return cost, path
        if cost > g.get(u, INF):
            continue
        for v, w in graph.get(u, []):
            ng = cost + w
            if ng < g.get(v, INF):
                g[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + heuristic.get(v, 0.0), ng, v))

    return INF, []


def grid_heuristic(
    node: int, target: int, width: int, euclidean: bool = True
) -> float:
    """Admissible heuristic for 2-D grid (node index = row*width + col)."""
    r1, c1 = divmod(node, width)
    r2, c2 = divmod(target, width)
    if euclidean:
        return math.hypot(r1 - r2, c1 - c2)
    return abs(r1 - r2) + abs(c1 - c2)  # Manhattan (4-connected)


def build_grid_heuristics(
    n_nodes: int, target: int, width: int, euclidean: bool = True
) -> Dict[int, float]:
    return {v: grid_heuristic(v, target, width, euclidean) for v in range(n_nodes)}


# ---------------------------------------------------------------------------
# Floyd-Warshall
# ---------------------------------------------------------------------------

def floyd_warshall(
    weight: List[List[float]],
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """All-pairs shortest paths — O(n³).

    Args:
        weight: n×n weight matrix; weight[i][j]=INF means no edge.

    Returns (dist, next) where next[i][j] is the first hop from i to j
    on a shortest path (None if i==j or unreachable).
    """
    n = len(weight)
    dist = [row[:] for row in weight]
    nxt: List[List[Optional[int]]] = [
        [j if weight[i][j] < INF else None for j in range(n)] for i in range(n)
    ]
    for i in range(n):
        dist[i][i] = 0.0
        nxt[i][i] = None

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                candidate = dist[i][k] + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate
                    nxt[i][j] = nxt[i][k]

    return dist, nxt


def fw_path(
    nxt: List[List[Optional[int]]], source: int, target: int
) -> List[int]:
    """Reconstruct path from Floyd-Warshall next-hop table."""
    if nxt[source][target] is None:
        return [] if source != target else [source]
    path = [source]
    cur = source
    while cur != target:
        hop = nxt[cur][target]
        if hop is None:
            return []  # unreachable
        path.append(hop)
        cur = hop
    return path


# ---------------------------------------------------------------------------
# Johnson's algorithm
# ---------------------------------------------------------------------------

def johnson(
    adj: Dict[int, List[Tuple[int, float]]],
    n_nodes: int,
) -> Optional[Dict[int, Dict[int, float]]]:
    """All-pairs shortest paths with negative weights — O(V·E·log V).

    Uses Bellman-Ford to compute re-weighting potentials, then runs
    Dijkstra from every source on the re-weighted graph.

    Returns dict dist[u][v] = shortest path length, or None if a
    negative-weight cycle is detected.
    """
    # Step 1: add virtual source q with zero-weight edges to all nodes
    virtual = n_nodes
    adj_ext = dict(adj)
    adj_ext[virtual] = [(v, 0.0) for v in range(n_nodes)]

    # Step 2: Bellman-Ford from virtual source
    h = [INF] * (n_nodes + 1)
    h[virtual] = 0.0
    for _ in range(n_nodes):
        updated = False
        for u in adj_ext:
            if h[u] == INF:
                continue
            for v, w in adj_ext[u]:
                if h[u] + w < h[v]:
                    h[v] = h[u] + w
                    updated = True
        if not updated:
            break
    # Check negative cycle
    for u in adj_ext:
        for v, w in adj_ext[u]:
            if h[u] + w < h[v]:
                return None  # negative cycle

    # Step 3: re-weight edges w'(u,v) = w(u,v) + h[u] - h[v]  (≥ 0)
    adj_rw: Dict[int, List[Tuple[int, float]]] = {}
    for u in range(n_nodes):
        adj_rw[u] = [(v, w + h[u] - h[v]) for v, w in adj.get(u, [])]

    # Step 4: Dijkstra from each source, un-reweight result
    result: Dict[int, Dict[int, float]] = {}
    for s in range(n_nodes):
        d_rw, _ = dijkstra(adj_rw, s, n_nodes)
        result[s] = {}
        for t in range(n_nodes):
            if d_rw[t] < INF:
                result[s][t] = d_rw[t] - h[s] + h[t]
            else:
                result[s][t] = INF

    return result


# ---------------------------------------------------------------------------
# Grid graph builder (shared with experiment scripts)
# ---------------------------------------------------------------------------

def grid_to_graph(
    grid: List[List[int]],
    four_connected: bool = True,
) -> Tuple[Dict[int, List[Tuple[int, float]]], int]:
    """Convert a 0/1 obstacle grid to an adjacency list.

    grid[r][c] == 1 means obstacle (impassable).
    Returns (graph, n_nodes) where n_nodes = rows*cols.
    """
    rows, cols = len(grid), len(grid[0])
    graph: Dict[int, List[Tuple[int, float]]] = {}

    if four_connected:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        moves = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                continue
            u = r * cols + c
            graph[u] = []
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    w = math.hypot(dr, dc)  # 1.0 or √2
                    graph[u].append((nr * cols + nc, w))

    return graph, rows * cols
