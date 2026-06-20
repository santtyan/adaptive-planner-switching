"""
Train SAC on TurtleBot3GazeboEnv.

SAC with automatic entropy tuning (ent_coef='auto') is preferred over DDPG
for continuous control — more sample-efficient and robust (Haarnoja 2018).

Usage (headless, recommended):
    TURTLEBOT3_MODEL=waffle gzserver worlds/dense_custom.world &
    ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=True &
    python3 train_sac.py --steps 500000 --seed 42

Parallel training (2 instances, separate ROS_DOMAIN_ID):
    ROS_DOMAIN_ID=0 python3 train_sac.py --seed 42 &
    ROS_DOMAIN_ID=1 python3 train_sac.py --seed 43 &

Plan B: if ep_rew_mean does not cross 50 by step 300k, abort and use PPO only.

Outputs:
    models/sac_<seed>_<steps>.zip        best model
    models/sac_<seed>_<steps>_final.zip  final model
    logs/sac_<seed>/                     TensorBoard logs
"""

import argparse
import os

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="logs")
    # Early abort (Plan B) — DESLIGADO por padrão (20/06/2026).
    # O threshold antigo (ep_rew_mean<50 @ 300k) era inatingível em mundo denso
    # (R_GOAL=50 exigiria quase todo episódio fechando goal) → matava o run.
    # Ativar explicitamente com --planb-enable se desejar o early-abort.
    p.add_argument("--planb-enable", action="store_true",
                   help="Habilita o early-abort Plan B (desligado por padrão)")
    p.add_argument("--planb-step", type=int, default=300_000,
                   help="Check ep_rew_mean at this step; abort if < planb-threshold")
    p.add_argument("--planb-threshold", type=float, default=50.0)
    return p.parse_args()


class PlanBCallback(BaseCallback):
    """Stop training early if SAC has not converged by planb_step.

    BaseCallback subclass (não um callable solto) para ser compatível com
    CallbackList quando passado em lista junto a EvalCallback/CheckpointCallback.
    """

    def __init__(self, check_step: int, threshold: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._check_step = check_step
        self._threshold = threshold
        self._triggered = False

    def _on_step(self) -> bool:
        if self._triggered:
            return True
        if self.num_timesteps >= self._check_step:
            mean_reward = self.logger.name_to_value.get(
                "rollout/ep_rew_mean", float("inf")
            )
            if mean_reward < self._threshold:
                print(
                    f"\n[PlanB] ep_rew_mean={mean_reward:.1f} < {self._threshold} "
                    f"at step {self.num_timesteps}. Aborting SAC — use PPO only.\n"
                )
                self._triggered = True
                return False  # stops training
        return True


class BestRolloutModelCallback(BaseCallback):
    """Salva o melhor modelo pela média de recompensa dos episódios recentes.

    Substitui o EvalCallback: o env de avaliação compartilhava o mesmo
    _GazeboEnvNode do treino (um único publisher /cmd_vel e subscriber /scan),
    então a eval corrompia o estado e selecionava o best_model com base em
    reward de avaliação inválido. Aqui usamos só a métrica de rollout do treino,
    sem segundo env.

    FIX (20/06/2026): a versão anterior lia ``logger.name_to_value`` apenas em
    ``num_timesteps % check_freq == 0``, instante em que a chave costuma estar
    ausente (o SB3 só popula no dump por episódio) → ficava ``-inf`` e o
    best_model nunca era salvo. Agora a média vem de ``model.ep_info_buffer``,
    que o SB3 mantém populado com os últimos episódios encerrados.
    """

    def __init__(self, save_path: str, check_freq: int = 2_000,
                 min_episodes: int = 20, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._save_path = os.path.join(save_path, "best_model.zip")
        self._check_freq = check_freq
        self._min_episodes = min_episodes
        self._best = float("-inf")

    def _on_step(self) -> bool:
        if self.num_timesteps % self._check_freq != 0:
            return True
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < self._min_episodes:
            return True
        mean_reward = sum(ep["r"] for ep in buf) / len(buf)
        if mean_reward > self._best:
            self._best = mean_reward
            self.model.save(self._save_path)
            if self.verbose:
                print(
                    f"[BestModel] novo melhor ep_rew_mean={mean_reward:.1f} "
                    f"(janela {len(buf)} eps) @ step {self.num_timesteps} "
                    f"→ {self._save_path}"
                )
        return True


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    rclpy.init()
    node = _GazeboEnvNode()

    train_env = Monitor(TurtleBot3GazeboEnv(node=node, seed=args.seed))

    model_name = f"sac_{args.seed}_{args.steps}"

    best_cb = BestRolloutModelCallback(
        save_path=args.models_dir,
        check_freq=args.eval_freq,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.models_dir,
        name_prefix=f"sac_{args.seed}_ckpt",
        verbose=0,
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = args.logs_dir
    except ImportError:
        print("[WARN] tensorboard not installed — logging disabled")
        tb_log = None

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,   # padrão-ouro off-policy: + exploração inicial
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,         # 4 updates por passo — passos de sim são caros
        ent_coef="auto",          # automatic entropy tuning
        target_entropy="auto",
        verbose=1,
        tensorboard_log=tb_log,
        seed=args.seed,
    )

    callbacks = [best_cb, ckpt_cb]
    if args.planb_enable:
        callbacks.append(PlanBCallback(
            check_step=args.planb_step,
            threshold=args.planb_threshold,
        ))
        print(f"[PlanB] habilitado: abort se ep_rew_mean<{args.planb_threshold} "
              f"@ step {args.planb_step}")
    else:
        print("[PlanB] desabilitado (default) — treino corre até o fim.")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        tb_log_name=f"sac_{args.seed}",
        progress_bar=False,
    )

    final_path = os.path.join(args.models_dir, f"{model_name}_final.zip")
    model.save(final_path)
    print(f"Saved final model: {final_path}")

    train_env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
