"""
train_gazebo_bc.py — Behavior Cloning no Gazebo real, mesma metodologia
validada no 2D (train_2d_bc.py, 98% sucesso, 09/07/2026).

Expert: campo potencial reativo (mesma lógica do 2D), usando posição REAL
(odometria/get_robot_pose) e obstáculos REAIS do sparse.world — NÃO usa
Nav2/SmacPlanner2D (não confirmado rodando neste setup; ver decisão em
[[project-treino-sparse-08jul]] sobre priorizar velocidade/confiabilidade
dado o horário). É um "professor" geométrico, não RL — sem problema de
exploração/reward, mesma vantagem observada no 2D.

Executar DENTRO do container (mesmo padrão dos smoke-tests desta sessão):
    docker compose exec train-all bash -c "
      source /opt/ros/humble/setup.bash &&
      source /workspace/ros2_ws/install/setup.bash &&
      python3 -u /workspace/ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/train/train_gazebo_bc.py \
        --episodes 60 --epochs 30"
"""
import os
import sys
import time
import argparse

import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode

# Obstáculos reais do sparse.world (paper/../worlds/sparse.world) — raio
# aproximado do modelo cilíndrico (mesmo padrão usado no gêmeo 2D).
OBSTACLES = [
    (-1.2, 1.2, 0.20),
    (1.0, 0.0, 0.20),
    (-0.3, -1.3, 0.20),
]

MODELS_DIR = "/workspace/models"


def expert_action(x: float, y: float, yaw: float, gx: float, gy: float) -> np.ndarray:
    """Campo potencial: mesma lógica do train_2d_bc.py, agora com pose real."""
    dx, dy = gx - x, gy - y
    dist_goal = np.hypot(dx, dy)
    fx, fy = dx / max(dist_goal, 1e-6), dy / max(dist_goal, 1e-6)

    for cx, cy, cr in OBSTACLES:
        ox, oy = x - cx, y - cy
        dist_obs = np.hypot(ox, oy) - cr
        influence = 0.6
        if 0 < dist_obs < influence:
            strength = (1.0 / max(dist_obs, 0.05) - 1.0 / influence)
            fx += strength * ox / max(np.hypot(ox, oy), 1e-6)
            fy += strength * oy / max(np.hypot(ox, oy), 1e-6)

    desired_yaw = np.arctan2(fy, fx)
    yaw_err = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
    w_norm = float(np.clip(yaw_err / (np.pi / 2), -1.0, 1.0))
    v_norm = float(np.clip(1.0 - abs(yaw_err) / (np.pi / 2), 0.15, 1.0))
    return np.array([v_norm, w_norm], dtype=np.float32)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def collect_demonstrations(env: TurtleBot3GazeboEnv, n_episodes: int):
    obs_list, act_list = [], []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            x, y, yaw = env.get_robot_pose()
            gx, gy = env._goal
            action = expert_action(x, y, yaw, gx, gy)
            obs_list.append(obs.copy())
            act_list.append(action.copy())
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        successes += int(info.get("goal_reached", False))
        if (ep + 1) % 10 == 0:
            print(f"  [expert] {ep+1}/{n_episodes} episódios, "
                  f"success acumulado={successes/(ep+1):.0%}", flush=True)
    print(f"Expert (campo potencial, Gazebo real): "
          f"success={successes/n_episodes:.0%} em {n_episodes} episódios "
          f"({len(obs_list)} transições)", flush=True)
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def evaluate_bc(env: TurtleBot3GazeboEnv, policy: BCPolicy, n_eval: int):
    policy.eval()
    successes = 0
    with torch.no_grad():
        for _ in range(n_eval):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 300:
                a = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                obs, r, term, trunc, info = env.step(a.squeeze(0).numpy())
                done = term or trunc
                steps += 1
            successes += int(info.get("goal_reached", False))
    policy.train()
    return successes / n_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rclpy.init()
    node = _GazeboEnvNode()
    env = TurtleBot3GazeboEnv(node=node, seed=args.seed)

    t0 = time.time()
    print(f"Coletando demonstrações do expert no Gazebo REAL — "
          f"{args.episodes} episódios...", flush=True)
    obs, act = collect_demonstrations(env, args.episodes)

    obs_t = torch.tensor(obs)
    act_t = torch.tensor(act)
    n = len(obs_t)
    dataset = torch.utils.data.TensorDataset(obs_t, act_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    policy = BCPolicy(obs.shape[1], act.shape[1])
    opt = optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Treinando BC — {n} transições reais, {args.epochs} épocas "
          f"(offline, sem Gazebo)...", flush=True)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [epoch {epoch+1:>3}/{args.epochs}] mse={total_loss/n:.4f}",
                  flush=True)

    print("Avaliando política BC no Gazebo real (20 episódios)...", flush=True)
    final_sr = evaluate_bc(env, policy, n_eval=20)

    elapsed = time.time() - t0
    save_path = os.path.join(MODELS_DIR, "bc_gazebo_policy.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"\nBC-Gazebo finalizado em {elapsed/60:.1f} min — "
          f"success final={final_sr:.0%}", flush=True)
    print(f"Modelo salvo em {save_path}", flush=True)

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
