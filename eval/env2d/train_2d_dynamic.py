"""
train_2d_dynamic.py — Treina SAC com obstáculos DINÂMICOS (outros robôs em movimento).

Usa MultiAgentEnv2D com N robôs: o agente 0 é o "aprendiz" (SAC), os demais
executam política de linha reta (A* analítico). O aprendiz vê os outros no LIDAR
como obstáculos dinâmicos e aprende a desviá-los — sem comunicação, sem MARL.

Resultado: modelo sac_2d_dynobs.zip com melhor evasão de colisão inter-robô.

Uso:
    python3 -m eval.env2d.train_2d_dynamic
    python3 -m eval.env2d.train_2d_dynamic --n-others 2 --world dense --steps 80000
"""

import os, sys, argparse, time
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from eval.env2d.env_2d_multi import MultiAgentEnv2D
from eval.env2d.env_2d import Env2D, WORLDS, ROBOT_RADIUS, GOAL_RADIUS
from eval.env2d.save_utils import safe_backup

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(ROOT, "paper", "figs")
MODS = os.path.join(ROOT, "models")
os.makedirs(MODS, exist_ok=True)


def _astar_action(env: MultiAgentEnv2D, i: int) -> np.ndarray:
    """Política de linha reta para agente i."""
    dx = env.gx[i] - env.x[i]
    dy = env.gy[i] - env.y[i]
    dist = np.hypot(dx, dy)
    ang  = np.arctan2(dy, dx)
    dtheta = (ang - env.yaw[i] + np.pi) % (2 * np.pi) - np.pi
    v = min(1.0, dist / 0.5) * (abs(dtheta) < 0.6)
    w = np.clip(dtheta / np.pi, -1.0, 1.0)
    return np.array([v, w], dtype=np.float32)


class DynamicObsEnv(gym.Env):
    """
    Wrapper single-agent sobre MultiAgentEnv2D.

    O agente 0 (aprendiz) recebe obs/reward/done individuais.
    Os agentes 1..N-1 executam política analítica (A* linha reta).
    Isso expõe o aprendiz a obstáculos dinâmicos durante o treino.
    """

    metadata = {}

    def __init__(self, n_others: int = 2, world: str = "sparse", seed: int = 42):
        super().__init__()
        self.n_others = n_others
        self.world    = world
        self._seed    = seed
        self._ep_seed = seed

        # Cria env interno para inferir espaços
        self._make_inner()

        # Espaços do agente 0
        inner_single = Env2D(world=world)
        self.observation_space = inner_single.observation_space
        self.action_space      = inner_single.action_space

    def _make_inner(self):
        self._env = MultiAgentEnv2D(
            n_agents=self.n_others + 1,
            world=self.world,
            seed=self._ep_seed,
        )

    def reset(self, *, seed=None, options=None):
        self._ep_seed += 1
        self._make_inner()
        self._env.reset()
        self._done = False
        obs0 = self._env._obs_i(0).astype(np.float32)
        return obs0, {}

    def step(self, action):
        actions = np.zeros((self.n_others + 1, 2), dtype=np.float32)
        actions[0] = np.clip(action, -1.0, 1.0)
        for i in range(1, self.n_others + 1):
            actions[i] = _astar_action(self._env, i)

        _, done = self._env.step(actions)

        # Reward e info do agente 0
        env   = self._env
        x, y  = env.x[0], env.y[0]
        gx, gy = env.gx[0], env.gy[0]
        dist  = np.hypot(x - gx, y - gy)

        goal_reached = env.goal_done[0]
        collision    = env.collided[0]

        if goal_reached:
            reward = 100.0
        elif collision:
            reward = -100.0
        else:
            prev_dist = getattr(self, "_prev_dist", dist)
            approach  = max(0.0, prev_dist - dist)
            reward    = 0.1 + 10.0 * approach   # R_SURV + R_APPROACH

        self._prev_dist = dist

        terminated = bool(goal_reached or collision)
        truncated  = bool(done and not terminated)

        # Obs do agente 0
        obs0 = self._env._obs_i(0).astype(np.float32)

        info = {
            "goal_reached": bool(goal_reached),
            "collision":    bool(collision),
        }
        return obs0, float(reward), terminated, truncated, info


# ── Callback de avaliação ─────────────────────────────────────
class SuccessCallback(BaseCallback):
    def __init__(self, save_path, eval_freq=2000, n_eval=20, world="sparse",
                 n_others=2, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval    = n_eval
        self.world     = world
        self.n_others  = n_others
        self.best_sr   = -1.0
        self.history   = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Avalia num env single (sem outros robôs) para comparar com baseline
        eval_env = Env2D(world=self.world)
        successes, lengths, rewards = [], [], []
        for ep in range(self.n_eval):
            obs, _ = eval_env.reset()
            done = False; ep_r = 0.0; ep_l = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = eval_env.step(action)
                ep_r += r; ep_l += 1
                done = term or trunc
            successes.append(float(info.get("goal_reached", False)))
            lengths.append(ep_l)
            rewards.append(ep_r)

        sr = np.mean(successes)
        self.history.append((self.num_timesteps, sr,
                             np.mean(lengths), np.mean(rewards)))

        if self.verbose:
            print(f"  [eval @ {self.num_timesteps:>7}]  "
                  f"success={sr:.0%}  ep_len={np.mean(lengths):.0f}  "
                  f"ep_rew={np.mean(rewards):.1f}")

        if sr > self.best_sr:
            self.best_sr = sr
            self.model.save(self.save_path)
            if self.verbose:
                print(f"  ✓ Novo best model salvo (success={sr:.0%})")

        if sr >= 0.90:
            print(f"\n  ★ Convergência atingida: success={sr:.0%} @ {self.num_timesteps} steps")
            return False
        return True


def train(args):
    print(f"Treinando SAC com obstáculos dinâmicos — "
          f"world={args.world}, n_others={args.n_others}, steps={args.steps}")
    t0 = time.time()

    def make_env():
        e = DynamicObsEnv(n_others=args.n_others, world=args.world, seed=args.seed)
        return Monitor(e)

    vec_env = DummyVecEnv([make_env])

    # Warm-start: carrega sac_2d_best se existir
    best_existing = os.path.join(MODS, "sac_2d_best.zip")
    save_path     = os.path.join(MODS, "sac_2d_dynobs")
    # Backup ANTES do treino começar: o callback vai sobrescrever save_path.zip
    # a cada novo "best" -- se já existia um best de um treino anterior, preservar.
    safe_backup(save_path + ".zip")
    safe_backup(os.path.join(MODS, "sac_2d_dynobs_final.zip"))

    if os.path.exists(best_existing) and not args.from_scratch:
        print("  Warm-start: carregando sac_2d_best...")
        model = SAC.load(best_existing, env=vec_env, device="cpu")
        model.set_env(vec_env)
    else:
        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=1e-4,       # menor LR para fine-tune
            buffer_size=200_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef=0.1,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=args.seed,
            device="cpu",
        )

    cb = SuccessCallback(
        save_path=save_path,
        eval_freq=1000, n_eval=20,
        world=args.world, n_others=args.n_others,
    )

    model.learn(total_timesteps=args.steps, callback=cb,
                progress_bar=False, reset_num_timesteps=True)

    model.save(os.path.join(MODS, "sac_2d_dynobs_final"))
    elapsed = time.time() - t0
    print(f"\nTreino concluído em {elapsed/60:.1f} min")
    print(f"Modelo salvo: models/sac_2d_dynobs.zip")
    print(f"Best success rate (single-robot eval): {cb.best_sr:.1%}")

    _plot_curve(cb.history, args.world, args.n_others)


def _plot_curve(history, world, n_others):
    if not history:
        return
    steps = [h[0] for h in history]
    sr    = [h[1] * 100 for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, sr, "o-", color="#2196F3", lw=2, ms=5)
    ax.axhline(90, ls="--", color="#4CAF50", lw=1.5, label="Meta: 90%")
    ax.set_xlabel("Steps"); ax.set_ylabel("Taxa de sucesso (%)")
    ax.set_title(f"SAC com obstáculos dinâmicos — {world}, {n_others} outros robôs")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGS, "2d", "fig_2d_learning_curve_dynobs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",        type=int, default=80_000)
    ap.add_argument("--world",        type=str, default="sparse")
    ap.add_argument("--n-others",     type=int, default=2,
                    help="Número de outros robôs (obstáculos dinâmicos) no treino")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--from-scratch", action="store_true",
                    help="Ignora sac_2d_best e treina do zero")
    args = ap.parse_args()
    train(args)
