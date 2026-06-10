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
import sys

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../.."))

from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="logs")
    # Early abort threshold: if mean reward < this at step planb_step, stop
    p.add_argument("--planb-step", type=int, default=300_000,
                   help="Check ep_rew_mean at this step; abort if < planb-threshold")
    p.add_argument("--planb-threshold", type=float, default=50.0)
    return p.parse_args()


class PlanBCallback:
    """Stop training early if SAC has not converged by planb_step."""

    def __init__(self, check_step: int, threshold: float) -> None:
        self._check_step = check_step
        self._threshold = threshold
        self._triggered = False

    def __call__(self, locals_: dict, globals_: dict) -> bool:
        if self._triggered:
            return True
        num_steps = locals_.get("self").num_timesteps
        if num_steps >= self._check_step:
            mean_reward = locals_.get("self").logger.name_to_value.get(
                "rollout/ep_rew_mean", float("inf")
            )
            if mean_reward < self._threshold:
                print(
                    f"\n[PlanB] ep_rew_mean={mean_reward:.1f} < {self._threshold} "
                    f"at step {num_steps}. Aborting SAC — use PPO only.\n"
                )
                self._triggered = True
                return False  # stops training
        return True


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    rclpy.init()
    node = _GazeboEnvNode()

    train_env = Monitor(TurtleBot3GazeboEnv(node=node, seed=args.seed))
    eval_env = Monitor(TurtleBot3GazeboEnv(node=node, seed=args.seed + 1))

    model_name = f"sac_{args.seed}_{args.steps}"

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.models_dir,
        log_path=os.path.join(args.logs_dir, f"sac_{args.seed}"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.models_dir,
        name_prefix=f"sac_{args.seed}_ckpt",
        verbose=0,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",          # automatic entropy tuning
        target_entropy="auto",
        verbose=1,
        tensorboard_log=args.logs_dir,
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=f"sac_{args.seed}",
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
