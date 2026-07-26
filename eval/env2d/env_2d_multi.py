"""
env_2d_multi.py — Multi-agente independente no env 2D leve.

N robôs unicycle, cada um com política RL INDEPENDENTE (sem comunicação).
Cada robô "vê" os outros via LIDAR (vizinhos viram obstáculos circulares dinâmicos).

Objetivo: medir onde o RL independente quebra (colisão inter-robô, deadlock,
queda de goal-rate) conforme N cresce → evidência REAL que motiva o MARL.

Reutiliza _scan, constantes e make_observation de env_2d.py (sem duplicar lógica).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.env2d.env_2d import (
    _scan, WORLDS, make_observation,
    ROBOT_RADIUS, GOAL_RADIUS, MAX_STEPS, DT,
    LINEAR_VEL_MAX, ANGULAR_VEL_MAX,
    R_SURVIVAL, R_COLLISION, R_GOAL, R_APPROACH,
)

# ── Parâmetros multi-agente (alinhados ao eval_multiagent_independent_rl.py) ──
INTER_COLLISION_DIST = 0.40    # m — dois robôs colidem se dist < este valor
DEADLOCK_VEL_THRESH  = 0.03    # m/s — "parado" abaixo disso
DEADLOCK_STEPS       = int(8.0 / DT)   # 8 s sem progresso → deadlock


class MultiAgentEnv2D:
    """N robôs independentes em arena densa. Execução descentralizada."""

    def __init__(self, n_agents: int, world: str = "dense", seed: int = 42,
                 max_steps: int = MAX_STEPS, terminate_on_any: bool = False):
        assert world in WORLDS
        self.N         = n_agents
        self.cfg       = WORLDS[world]
        self.obstacles = self.cfg["obstacles"]
        self.arena     = self.cfg["size"]
        self.max_steps = max_steps
        self.terminate_on_any = terminate_on_any
        self._rng      = np.random.default_rng(seed)

    # ── Spawn ─────────────────────────────────────────────────
    def _sample_free(self, others: list) -> tuple:
        """Posição livre de obstáculos, paredes e dos robôs já posicionados."""
        half = self.arena / 2.0 - 0.3
        for _ in range(400):
            x = self._rng.uniform(-half, half)
            y = self._rng.uniform(-half, half)
            ok_obs = all(np.hypot(x - cx, y - cy) > cr + ROBOT_RADIUS + 0.15
                         for cx, cy, cr in self.obstacles)
            ok_rob = all(np.hypot(x - ox, y - oy) > 2 * ROBOT_RADIUS + 0.30
                         for ox, oy in others)
            if ok_obs and ok_rob:
                return float(x), float(y)
        return 0.0, 0.0

    def reset(self):
        self.x   = np.zeros(self.N); self.y = np.zeros(self.N)
        self.yaw = np.zeros(self.N)
        self.gx  = np.zeros(self.N); self.gy = np.zeros(self.N)
        self.v_norm = np.zeros(self.N); self.w_norm = np.zeros(self.N)
        self.done       = np.zeros(self.N, dtype=bool)
        self.goal_done  = np.zeros(self.N, dtype=bool)
        self.collided   = np.zeros(self.N, dtype=bool)
        self.prev_dist  = np.zeros(self.N)
        self.init_dist  = np.ones(self.N)
        self.idle_count = np.zeros(self.N, dtype=int)
        self.t_goal     = np.full(self.N, np.nan)
        self._step = 0
        self.inter_collision = False
        self.last_reward = np.zeros(self.N, dtype=np.float32)

        placed = []
        for i in range(self.N):
            rx, ry = self._sample_free(placed)
            placed.append((rx, ry))
            self.x[i], self.y[i] = rx, ry
            self.yaw[i] = self._rng.uniform(-np.pi, np.pi)
            gx, gy = self._sample_free(placed + [(rx, ry)])
            self.gx[i], self.gy[i] = gx, gy
            d = np.hypot(gx - rx, gy - ry)
            self.prev_dist[i] = d
            self.init_dist[i] = max(d, 1e-3)
        return self._all_obs()

    # ── Observação ────────────────────────────────────────────
    def _obs_i(self, i: int) -> np.ndarray:
        # Vizinhos vivos viram obstáculos circulares dinâmicos no scan do robô i
        neigh = [(self.x[j], self.y[j], ROBOT_RADIUS)
                 for j in range(self.N) if j != i and not self.done[j]]
        ranges = _scan(self.x[i], self.y[i], self.yaw[i],
                       self.obstacles + neigh, self.arena)
        return make_observation(ranges, self.x[i], self.y[i], self.yaw[i],
                                self.gx[i], self.gy[i],
                                self.v_norm[i], self.w_norm[i])

    def _all_obs(self) -> np.ndarray:
        return np.stack([self._obs_i(i) for i in range(self.N)])

    # ── Passo ─────────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        """Retorna (obs, reward, done_flags, all_done). `reward` inclui uma
        penalidade COMPARTILHADA de colisão inter-robô (aplicada aos dois
        agentes envolvidos, não só a quem "causou"), necessária para treino
        MARL com incentivo real de coordenação — ausente na versão original
        deste ambiente, que só servia para avaliar políticas já treinadas
        independentemente. Ver [[project-treino-sparse-08jul]] (09/07/2026)."""
        self._step += 1
        actions = np.asarray(actions).reshape(self.N, 2)
        # Robôs já terminados (goal ou colisão) não zeram a reward média do
        # grupo enquanto os demais continuam ativos — mantêm a última reward
        # terminal (0 se preferir neutralidade) até all_done. Sem isso, um
        # robô que chega ao goal cedo é "punido" na média por dezenas de
        # passos de reward=0 dos companheiros ainda ativos, afogando o sinal
        # de sucesso (achado 26/07/2026, goal_rate=0% reprodutível em 3 seeds).
        reward = np.array([self.last_reward[i] if self.done[i] else 0.0
                            for i in range(self.N)], dtype=np.float32)

        for i in range(self.N):
            if self.done[i]:
                continue
            v_norm = float(np.clip(actions[i, 0], -1, 1))
            w_norm = float(np.clip(actions[i, 1], -1, 1))
            v = v_norm * LINEAR_VEL_MAX
            w = w_norm * ANGULAR_VEL_MAX

            self.yaw[i] += w * DT
            self.yaw[i]  = (self.yaw[i] + np.pi) % (2 * np.pi) - np.pi
            nx = self.x[i] + v * np.cos(self.yaw[i]) * DT
            ny = self.y[i] + v * np.sin(self.yaw[i]) * DT

            half = self.arena / 2.0 - ROBOT_RADIUS
            wall = not (-half <= nx <= half and -half <= ny <= half)
            obs_hit = any(np.hypot(nx - cx, ny - cy) < cr + ROBOT_RADIUS
                          for cx, cy, cr in self.obstacles)
            env_coll = wall or obs_hit
            if not env_coll:
                self.x[i], self.y[i] = nx, ny

            self.v_norm[i] = v_norm
            self.w_norm[i] = w_norm

            # Deadlock: parado sem progresso
            if abs(v) < DEADLOCK_VEL_THRESH:
                self.idle_count[i] += 1
            else:
                self.idle_count[i] = 0

            old_dist = self.prev_dist[i]
            dist = float(np.hypot(self.gx[i] - self.x[i], self.gy[i] - self.y[i]))
            goal = dist < GOAL_RADIUS
            self.prev_dist[i] = dist

            rprox = float(np.clip(1.0 - dist / max(old_dist, 1e-3), 0.0, 1.0))
            if env_coll:
                self.collided[i] = True
                self.done[i] = True
                reward[i] = R_COLLISION + rprox
            elif goal:
                self.goal_done[i] = True
                self.done[i] = True
                self.t_goal[i] = self._step * DT
                reward[i] = R_GOAL
            else:
                reward[i] = R_SURVIVAL + R_APPROACH * max(0.0, old_dist - dist)

        # Colisão inter-robô (qualquer par vivo) — penalidade COMPARTILHADA:
        # os dois agentes envolvidos recebem R_COLLISION, não só um "culpado"
        # arbitrário. Isso é o que diferencia este treino de RL independente:
        # o incentivo de evitar o outro robô existe para ambos os lados.
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.done[i] and self.done[j]:
                    continue
                if np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j]) < INTER_COLLISION_DIST:
                    self.inter_collision = True
                    for k in (i, j):
                        if not self.goal_done[k]:
                            self.collided[k] = True
                            self.done[k] = True
                            reward[k] = R_COLLISION

        timeout = self._step >= self.max_steps
        # `terminate_on_any=True` encerra o episódio assim que QUALQUER robô
        # conclui (goal ou colisão), não só quando todos concluem. Achado
        # 26/07/2026 (DEVELOPMENT_LOG.md Fase 11): manter o episódio vivo até
        # o último agente terminar prolonga a "cauda" de passos de reward
        # pequena somados/mediados depois que um agente já teve sucesso,
        # diluindo o sinal de R_GOAL para a política MARL centralizada.
        # Default False preserva o comportamento original (episódio completo)
        # para avaliadores que medem o grupo inteiro, ex. inter_collision em
        # eval_multi_2d.py.
        stop_condition = self.done.any() if self.terminate_on_any else self.done.all()
        all_done = bool(stop_condition or timeout)
        # Exposto como atributo (não no retorno) para não quebrar chamadores
        # existentes que esperam `obs, done = env.step(actions)`.
        self.last_reward = reward
        return self._all_obs(), all_done

    # ── Métricas do trial ─────────────────────────────────────
    def metrics(self) -> dict:
        deadlock = bool((self.idle_count >= DEADLOCK_STEPS).any())
        return {
            "n_agents":        self.N,
            "inter_collision": bool(self.inter_collision),
            "env_collision":   bool(self.collided.sum() > 0 and not self.inter_collision),
            "deadlock":        deadlock,
            "goal_rate":       float(self.goal_done.mean()),
            "time_to_goal":    float(np.nanmean(self.t_goal)) if self.goal_done.any()
                               else float(self.max_steps * DT),
        }
