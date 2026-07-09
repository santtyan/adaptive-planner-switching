"""
train_2d_bc.py — Behavior Cloning (imitação supervisionada) no env 2D leve.

Expert: controlador reativo de campo potencial (acesso privilegiado a
posição/goal/obstáculos — NÃO é RL, é um "professor" geométrico determinístico
usado só para gerar demonstrações). O aluno (MLP) aprende (obs LIDAR 29-dim →
ação) por regressão supervisionada, sem problema de exploração/reward — é o
caminho mais rápido pra ter um "componente aprendido" funcionando no Gazebo
se o RL não convergir a tempo (backup independente, ver [[project-treino-sparse-08jul]]).

Uso:
    R_SURVIVAL_OVERRIDE=0.0 PYTHONUNBUFFERED=1 python3 -u -m eval.env2d.train_2d_bc \
        --episodes 300 --epochs 50
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.env2d.env_2d import Env2D, LINEAR_VEL_MAX, ANGULAR_VEL_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODS = os.path.join(ROOT, "models")
os.makedirs(MODS, exist_ok=True)


def expert_action(env: Env2D) -> np.ndarray:
    """Campo potencial: atrai pro goal, repele de obstáculos próximos.
    Acesso privilegiado ao estado interno do env (posição real, não LIDAR) —
    válido para gerar demonstrações, não para inferência (o aluno usa só obs).
    """
    gx, gy = env._gx - env._x, env._gy - env._y
    dist_goal = np.hypot(gx, gy)
    fx, fy = gx / max(dist_goal, 1e-6), gy / max(dist_goal, 1e-6)  # atrativa unitária

    for cx, cy, cr in env.obstacles:
        ox, oy = env._x - cx, env._y - cy
        dist_obs = np.hypot(ox, oy) - cr
        influence = 0.6  # m — raio de influência repulsiva
        if 0 < dist_obs < influence:
            strength = (1.0 / max(dist_obs, 0.05) - 1.0 / influence)
            fx += strength * ox / max(np.hypot(ox, oy), 1e-6)
            fy += strength * oy / max(np.hypot(ox, oy), 1e-6)

    desired_yaw = np.arctan2(fy, fx)
    yaw_err = (desired_yaw - env._yaw + np.pi) % (2 * np.pi) - np.pi

    w_norm = float(np.clip(yaw_err / (np.pi / 2), -1.0, 1.0))
    # desacelera em curvas fechadas — evita colisão por excesso de velocidade
    v_norm = float(np.clip(1.0 - abs(yaw_err) / (np.pi / 2), 0.15, 1.0))
    return np.array([v_norm, w_norm], dtype=np.float32)


def collect_demonstrations(world: str, n_episodes: int, seed: int = 42):
    env = Env2D(world=world, seed=seed)
    obs_list, act_list = [], []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = expert_action(env)
            obs_list.append(obs.copy())
            act_list.append(action.copy())
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        successes += int(info.get("goal_reached", False))
    print(f"Expert (campo potencial): success={successes/n_episodes:.0%} "
          f"em {n_episodes} episódios ({len(obs_list)} transições coletadas)")
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),   # ação em [-1,1]
        )

    def forward(self, x):
        return self.net(x)


def evaluate_bc(policy: BCPolicy, world: str, n_eval: int = 50, seed: int = 123):
    env = Env2D(world=world, seed=seed)
    policy.eval()
    successes = 0
    with torch.no_grad():
        for _ in range(n_eval):
            obs, _ = env.reset()
            done = False
            while not done:
                a = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                obs, r, term, trunc, info = env.step(a.squeeze(0).numpy())
                done = term or trunc
            successes += int(info.get("goal_reached", False))
    policy.train()
    return successes / n_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=str, default="sparse")
    p.add_argument("--episodes", type=int, default=300, help="episódios de demonstração do expert")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print(f"Coletando demonstrações do expert — world={args.world}, "
          f"{args.episodes} episódios...")
    obs, act = collect_demonstrations(args.world, args.episodes, args.seed)

    obs_t = torch.tensor(obs)
    act_t = torch.tensor(act)
    n = len(obs_t)
    dataset = torch.utils.data.TensorDataset(obs_t, act_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    policy = BCPolicy(obs.shape[1], act.shape[1])
    opt = optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Treinando BC — {n} transições, {args.epochs} épocas...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            sr = evaluate_bc(policy, args.world, n_eval=30, seed=999)
            print(f"  [epoch {epoch+1:>3}/{args.epochs}]  "
                  f"mse={total_loss/n:.4f}  success={sr:.0%}")

    final_sr = evaluate_bc(policy, args.world, n_eval=50, seed=999)
    elapsed = time.time() - t0
    save_path = os.path.join(MODS, "bc_2d_policy.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"\nBC finalizado em {elapsed/60:.1f} min — success final={final_sr:.0%}")
    print(f"Modelo salvo em {save_path}")


if __name__ == "__main__":
    main()
