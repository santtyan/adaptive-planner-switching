"""
train_2d.py — Treina SAC no env 2D leve (~20-40 min em CPU).

Uso:
    python3 eval/env2d/train_2d.py
    python3 eval/env2d/train_2d.py --steps 200000 --world sparse
    python3 eval/env2d/train_2d.py --eval           # só avalia modelo salvo

Saída:
    models/sac_2d_best.zip   — melhor modelo por success_rate
    models/sac_2d_final.zip  — modelo final
    paper/figs/fig_2d_learning_curve.png/.pdf
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from eval.env2d.env_2d import Env2D
from eval.env2d.save_utils import safe_backup

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(os.path.dirname(HERE))
FIGS  = os.path.join(ROOT, "paper", "figs")
MODS  = os.path.join(ROOT, "models")
os.makedirs(MODS, exist_ok=True)


# ── Callback: monitora success_rate e salva best model ────────
class SuccessCallback(BaseCallback):
    def __init__(self, save_path: str, eval_freq: int = 2000,
                 n_eval: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.save_path  = save_path
        self.eval_freq  = eval_freq
        self.n_eval     = n_eval
        self.best_sr    = -1.0
        self.history    = []   # (step, success_rate, ep_len, ep_rew)

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

        sr  = np.mean(successes)
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

        # Para automaticamente se convergiu
        if sr >= 0.90:
            print(f"\n  ★ Convergência atingida: success={sr:.0%} @ {self.num_timesteps} steps")
            return False   # interrompe treino

        return True


def train(args):
    print(f"Treinando no env 2D — world={args.world}, steps={args.steps}")
    t0 = time.time()

    def make_env():
        e = Env2D(world=args.world, seed=args.seed)
        e.world = args.world
        return Monitor(e)   # registra ep_rew/ep_len automaticamente

    vec_env = DummyVecEnv([make_env])

    model = SAC(
        "MlpPolicy", vec_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=500,   # rápido para começar
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.1,
        train_freq=1,
        gradient_steps=1,      # 1 update/step — sem deadlock de threads CPU
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=args.seed,
        device="cpu",
    )

    best_path = os.path.join(MODS, "sac_2d_best")
    safe_backup(best_path + ".zip")
    safe_backup(os.path.join(MODS, "sac_2d_final.zip"))
    cb = SuccessCallback(save_path=best_path, eval_freq=1000,
                         n_eval=20, verbose=1)

    model.learn(total_timesteps=args.steps, callback=cb,
                progress_bar=False, reset_num_timesteps=True)

    model.save(os.path.join(MODS, "sac_2d_final"))
    elapsed = time.time() - t0
    print(f"\nTreino concluído em {elapsed/60:.1f} min")
    print(f"Best success rate: {cb.best_sr:.1%}")

    _plot_curve(cb.history, args.world)
    return cb.best_sr


def _plot_curve(history, world: str):
    if not history:
        return
    steps = [h[0] for h in history]
    sr    = [h[1] for h in history]
    elen  = [h[2] for h in history]
    erew  = [h[3] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(steps, [s*100 for s in sr], "o-", color="#2196F3", lw=2)
    axes[0].axhline(90, ls="--", color="#4CAF50", lw=1.5, label="Meta: 90%")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Taxa de sucesso (%)")
    axes[0].set_title("(a) Taxa de sucesso"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, elen, "s-", color="#FF9800", lw=2)
    axes[1].set_xlabel("Steps"); axes[1].set_ylabel("ep_len_mean")
    axes[1].set_title("(b) Comprimento de episódio"); axes[1].grid(alpha=0.3)

    axes[2].plot(steps, erew, "^-", color="#9C27B0", lw=2)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_xlabel("Steps"); axes[2].set_ylabel("ep_rew_mean")
    axes[2].set_title("(c) Reward médio"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Curva de aprendizado SAC — env 2D ({world})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(FIGS, f"fig_2d_learning_curve.{ext}")
        plt.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ fig_2d_learning_curve.png")


def evaluate(args):
    path = os.path.join(MODS, "sac_2d_best.zip")
    if not os.path.exists(path):
        print(f"Modelo não encontrado: {path}")
        return

    model = SAC.load(path, device="cpu")
    env   = Env2D(world=args.world)
    results = []

    for ep in range(50):
        obs, _ = env.reset()
        done = False; ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
        results.append(info.get("goal_reached", False))

    sr = np.mean(results)
    print(f"Avaliação (50 eps, world={args.world}): success={sr:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps",  type=int,   default=150_000)
    p.add_argument("--world",  type=str,   default="sparse")
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--eval",   action="store_true")
    args = p.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
