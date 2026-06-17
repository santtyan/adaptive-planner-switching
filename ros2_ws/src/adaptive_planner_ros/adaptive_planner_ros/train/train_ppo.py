"""
Train PPO on TurtleBot3GazeboEnv.

Usage (headless, recommended):
    TURTLEBOT3_MODEL=waffle gzserver worlds/dense_custom.world &
    ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=True &
    python3 train_ppo.py --steps 500000 --seed 42

Outputs:
    models/ppo_<seed>_<steps>.zip      best model (saved by EvalCallback)
    models/ppo_<seed>_<steps>_final.zip  final model
    logs/ppo_<seed>/                   TensorBoard logs
"""

import argparse
import os

import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Run evaluation every N steps")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    rclpy.init()
    node = _GazeboEnvNode()

    # Training environment
    train_env = Monitor(TurtleBot3GazeboEnv(node=node, seed=args.seed))

    # Separate eval env (same node — Gazebo is single-instance)
    eval_env = Monitor(TurtleBot3GazeboEnv(node=node, seed=args.seed + 1))

    model_name = f"ppo_{args.seed}_{args.steps}"

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=args.models_dir,
            log_path=os.path.join(args.logs_dir, f"ppo_{args.seed}"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=args.models_dir,
            name_prefix=f"ppo_{args.seed}_ckpt",
            verbose=0,
        ),
    ]

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.logs_dir,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        tb_log_name=f"ppo_{args.seed}",
        progress_bar=True,
    )

    final_path = os.path.join(args.models_dir, f"{model_name}_final.zip")
    model.save(final_path)
    print(f"Saved final model: {final_path}")

    train_env.close()
    eval_env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
