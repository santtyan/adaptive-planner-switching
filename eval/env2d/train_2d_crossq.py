"""
train_2d_crossq.py — Testa CrossQ (sb3_contrib) no env 2D leve.

CrossQ (Bhatt et al., ICLR 2024): BatchNorm no crítico, sem rede-alvo, policy_delay.
Alternativa SOTA ao DroQ para eficiência de amostra — ver [[project-sac-training-fix]].
Objetivo: comparar velocidade de convergência com o SAC baseline (90% @ 14k/6k steps)
ANTES de gastar horas testando no Gazebo (regra 2D-antes-Gazebo, 09/07/2026).

Uso:
    R_SURVIVAL_OVERRIDE=0.0 python3 -u -m eval.env2d.train_2d_crossq --steps 30000
"""

import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from eval.env2d.env_2d import Env2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODS = os.path.join(ROOT, "models")
os.makedirs(MODS, exist_ok=True)


class SuccessCallback(BaseCallback):
    def __init__(self, save_path: str, eval_freq: int = 1000,
                 n_eval: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.best_sr = -1.0
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        inner = self.training_env.envs[0]
        world = getattr(inner, "world", None) or getattr(getattr(inner, "env", None), "world", "sparse")
        env = Env2D(world=world)
        successes, lengths, rewards = [], [], []
        for _ in range(self.n_eval):
            obs, _ = env.reset()
            done = False; ep_r = 0.0; ep_l = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_r += r; ep_l += 1
                done = term or trunc
            successes.append(float(info.get("goal_reached", False)))
            lengths.append(ep_l)
            rewards.append(ep_r)
        sr = np.mean(successes)
        self.history.append((self.num_timesteps, sr, np.mean(lengths), np.mean(rewards)))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--world", type=str, default="sparse")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Treinando CrossQ no env 2D — world={args.world}, steps={args.steps}")
    t0 = time.time()

    def make_env():
        e = Env2D(world=args.world, seed=args.seed)
        e.world = args.world
        return Monitor(e)

    vec_env = DummyVecEnv([make_env])

    model = CrossQ(
        "MlpPolicy", vec_env,
        learning_rate=1e-3,      # default CrossQ (maior que SAC — sem rede-alvo, mais estável)
        buffer_size=100_000,
        learning_starts=500,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_delay=3,          # default CrossQ
        verbose=0,
        seed=args.seed,
        device="cpu",
    )

    best_path = os.path.join(MODS, "crossq_2d_best")
    cb = SuccessCallback(save_path=best_path, eval_freq=1000, n_eval=20, verbose=1)

    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)

    elapsed = time.time() - t0
    print(f"\nTreino CrossQ finalizado em {elapsed/60:.1f} min "
          f"({model.num_timesteps} steps, best_sr={cb.best_sr:.0%})")


if __name__ == "__main__":
    main()
