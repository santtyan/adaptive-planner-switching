"""
train_2d_marl.py — MARL simplificado (reward compartilhada, política
centralizada) no gêmeo 2D multiagente, para comparar contra o RL
independente já medido (ver [[project-treino-sparse-08jul]]).

Formulação: os N robôs são tratados como um único "meta-agente" cuja
observação é a concatenação das observações individuais e cuja ação é a
concatenação das ações individuais — uma política centralizada única,
treinada com PPO padrão do Stable-Baselines3, sem nenhuma modificação de
biblioteca. A reward por passo é a MÉDIA das rewards individuais
(incluindo a penalidade compartilhada de colisão inter-robô adicionada em
env_2d_multi.py), criando um objetivo de equipe genuíno — diferente do RL
independente, onde cada robô otimiza só a própria reward sem nenhum
incentivo de considerar o outro.

Isto é uma simplificação de MARL (treino e execução centralizados, não
CTDE com política descentralizada tipo QMIX/MAPPO), mas é um treino
conjunto real com reward compartilhada — a peça que faltava para
distinguir de "RL independente" de verdade, não apenas mais uma coincidência
de nome.

Uso:
    python3 -m eval.env2d.train_2d_marl --n-agents 4 --world sparse --steps 30000
"""
import sys, os, argparse, time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.env2d.env_2d_multi import MultiAgentEnv2D
from eval.env2d.env_2d import OBS_DIM

MODS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


class CentralizedMultiAgentWrapper(gym.Env):
    """Achata N agentes de MultiAgentEnv2D num único env Gym padrão:
    obs = concat(obs_1..obs_N), action = concat(action_1..action_N),
    reward = média das rewards individuais (objetivo de equipe)."""

    def __init__(self, n_agents: int, world: str, seed: int = 42):
        super().__init__()
        self.n_agents = n_agents
        self.world = world
        self._seed = seed
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                             shape=(n_agents * OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                        shape=(n_agents * 2,), dtype=np.float32)
        self._env = None

    def reset(self, *, seed=None, options=None):
        s = seed if seed is not None else self._seed
        self._env = MultiAgentEnv2D(self.n_agents, world=self.world, seed=s)
        obs = self._env.reset()
        return obs.flatten().astype(np.float32), {}

    def step(self, action):
        actions = np.asarray(action, dtype=np.float32).reshape(self.n_agents, 2)
        obs, all_done = self._env.step(actions)
        reward = float(np.mean(self._env.last_reward))
        terminated = all_done
        truncated = False
        m = self._env.metrics()
        info = {"goal_rate": m["goal_rate"], "inter_collision": m["inter_collision"]}
        return obs.flatten().astype(np.float32), reward, terminated, truncated, info


def evaluate(model, n_agents, world, n_eval=30, seed0=999):
    goal_rates, inter_colls = [], []
    for t in range(n_eval):
        env = CentralizedMultiAgentWrapper(n_agents, world, seed=seed0 + t)
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        goal_rates.append(info["goal_rate"])
        inter_colls.append(info["inter_collision"])
    return float(np.mean(goal_rates)), float(np.mean(inter_colls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--world", default="sparse")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    def make_env():
        return CentralizedMultiAgentWrapper(args.n_agents, args.world, seed=args.seed)

    vec_env = make_vec_env(make_env, n_envs=8, seed=args.seed)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=0.99)

    t0 = time.time()
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=args.seed,
                n_steps=1024, batch_size=256, gamma=0.99, ent_coef=0.01,
                learning_rate=3e-4)
    print(f"Treinando MARL centralizado (reward compartilhada) — "
          f"N={args.n_agents} agentes, world={args.world}, {args.steps} passos...", flush=True)
    model.learn(total_timesteps=args.steps)
    elapsed = time.time() - t0

    # Avaliação através do próprio VecEnv (mesma normalização de obs do
    # treino, congelada) — avaliar com wrapper cru geraria obs fora de
    # escala e mediria a política errada.
    vec_env.training = False
    vec_env.norm_reward = False
    goal_rates, inter_colls = [], []
    for t in range(30):
        obs = vec_env.reset()
        done = [False]
        info = [{}]
        while not done[0]:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec_env.step(a)
        goal_rates.append(info[0]["goal_rate"])
        inter_colls.append(info[0]["inter_collision"])
    gr, ic = float(np.mean(goal_rates)), float(np.mean(inter_colls))
    print(f"\nMARL centralizado ({elapsed/60:.1f} min): "
          f"goal_rate={gr:.0%}  inter_collision={ic:.0%}", flush=True)

    save_path = os.path.join(MODS, f"marl_centralized_{args.world}_N{args.n_agents}.zip")
    model.save(save_path)
    vec_env.save(os.path.join(MODS, f"marl_centralized_{args.world}_N{args.n_agents}_vecnorm.pkl"))
    print(f"Modelo salvo em {save_path}")


if __name__ == "__main__":
    main()
