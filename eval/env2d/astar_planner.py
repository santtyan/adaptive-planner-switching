"""
astar_planner.py — A* REAL (busca em grade, heapq) para o Env2D, substituindo
a política de linha reta que estava incorretamente rotulada como "A*" em
eval_multi_2d.py / visualize_2d.py.

Rasteriza os obstáculos circulares do mundo numa grade, roda A* 8-conectado
com heurística octile (mesma estrutura de dados — heap binário — usada no
benchmark clássico de validation_abstract/algorithms/classical.py, atendendo
à mesma exigência do parecer do consultor SIGAA de "implementações
otimizadas"), e converte o caminho resultante numa política de perseguição
de waypoints (pure pursuit simples) compatível com a action space contínua
do Env2D.

Ver [[project-treino-sparse-08jul]] — motivado pela descoberta de que o
"A*" usado até 09/07/2026 nas comparações multi-agente era, na verdade,
uma política analítica de linha reta, não busca real.
"""
import heapq
import numpy as np


def _rasterize(obstacles, arena_half: float, robot_radius: float,
                resolution: float = 0.08):
    """Marca células bloqueadas (True) numa grade quadrada [-half, half]^2."""
    n = int(2 * arena_half / resolution) + 1
    blocked = np.zeros((n, n), dtype=bool)
    xs = np.linspace(-arena_half, arena_half, n)
    ys = np.linspace(-arena_half, arena_half, n)
    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            for cx, cy, cr in obstacles:
                if np.hypot(gx - cx, gy - cy) < cr + robot_radius:
                    blocked[i, j] = True
                    break
    return blocked, xs, ys


def _to_cell(x, y, xs, ys):
    i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
    j = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
    return i, j


_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _astar_search(blocked, start_cell, goal_cell):
    """A* 8-conectado, heap binário, heurística octile — busca REAL, não proxy."""
    n = blocked.shape[0]

    def octile(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    open_heap = [(octile(start_cell, goal_cell), 0.0, start_cell, None)]
    came_from = {}
    g_score = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, g, current, parent = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal_cell:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        for dx, dy in _NEIGHBORS:
            ni, nj = current[0] + dx, current[1] + dy
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            if blocked[ni, nj]:
                continue
            step_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1.0
            ng = g + step_cost
            if (ni, nj) not in g_score or ng < g_score[(ni, nj)]:
                g_score[(ni, nj)] = ng
                f = ng + octile((ni, nj), goal_cell)
                heapq.heappush(open_heap, (f, ng, (ni, nj), current))
    return None  # sem caminho


def plan_astar(start_xy, goal_xy, obstacles, arena_half: float,
                robot_radius: float, resolution: float = 0.08):
    """Retorna lista de waypoints (x, y) do start ao goal, ou None se
    inalcançável na grade rasterizada."""
    blocked, xs, ys = _rasterize(obstacles, arena_half, robot_radius, resolution)
    start_cell = _to_cell(start_xy[0], start_xy[1], xs, ys)
    goal_cell = _to_cell(goal_xy[0], goal_xy[1], xs, ys)
    if blocked[start_cell] or blocked[goal_cell]:
        return None
    cells = _astar_search(blocked, start_cell, goal_cell)
    if cells is None:
        return None
    return [(float(xs[i]), float(ys[j])) for i, j in cells]


class AStarPolicy:
    """Política que segue o caminho A* real via pure-pursuit simples.
    Recalcula o caminho uma vez por episódio (obstáculos estáticos —
    replanning contínuo não é necessário, igual a um A* real em Nav2
    rodando sobre um costmap estático)."""

    def __init__(self, lookahead: float = 0.18, safety_margin: float = 0.08):
        self.lookahead = lookahead
        self.safety_margin = safety_margin
        self.path = None
        self.idx = 0

    def reset(self, env):
        # Margem de segurança extra além do raio do robô: o controlador
        # pure-pursuit corta cantos entre waypoints discretos, então o
        # caminho precisa de clearance maior que o robô sozinho exigiria.
        self.path = plan_astar(
            (env._x, env._y), (env._gx, env._gy), env.obstacles,
            env.arena / 2.0, 0.17 + self.safety_margin,  # ROBOT_RADIUS + margem
        )
        self.idx = 0
        if self.path is None:
            # sem caminho encontrado na grade — fallback: linha reta
            # (mesma degradação que um A* real teria sobre grade sem solução)
            self.path = [(env._x, env._y), (env._gx, env._gy)]

    def act(self, env) -> np.ndarray:
        x, y, yaw = env._x, env._y, env._yaw
        # avança o índice do waypoint enquanto estiver perto o bastante
        while (self.idx < len(self.path) - 1 and
               np.hypot(self.path[self.idx][0] - x, self.path[self.idx][1] - y) < self.lookahead):
            self.idx += 1
        tx, ty = self.path[self.idx]
        dx, dy = tx - x, ty - y
        desired_yaw = np.arctan2(dy, dx)
        yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        w_norm = float(np.clip(yaw_err / (np.pi / 2), -1.0, 1.0))
        v_norm = float(np.clip(1.0 - abs(yaw_err) / (np.pi / 2), 0.15, 1.0))
        return np.array([v_norm, w_norm], dtype=np.float32)
