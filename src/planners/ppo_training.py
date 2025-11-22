# src/planners/ppo_training.py - Training PPO real
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import time
import math

from environment import SimpleEnvironment

class NavigationGymEnv(gym.Env):
    '''Ambiente Gym para training PPO'''
    
    def __init__(self, obstacle_density=0.3, grid_size=100):
        super().__init__()
        
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.env = SimpleEnvironment(obstacle_density=obstacle_density)
        
        # Action space: [dx, dy] movimento
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [current_x, current_y, goal_x, goal_y, density]
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(5,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Posições válidas aleatórias
        while True:
            self.current_pos = np.random.uniform(5, self.grid_size-5, 2)
            if self.env.is_valid(self.current_pos[0], self.current_pos[1]):
                break
                
        while True:
            self.goal_pos = np.random.uniform(5, self.grid_size-5, 2)
            if (self.env.is_valid(self.goal_pos[0], self.goal_pos[1]) and
                np.linalg.norm(self.goal_pos - self.current_pos) > 20):
                break
        
        self.step_count = 0
        self.max_steps = 200
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Aplicar ação
        new_pos = self.current_pos + action
        new_pos = np.clip(new_pos, 0, self.grid_size)
        
        # Check collision
        if self.env.is_valid(new_pos[0], new_pos[1]):
            self.current_pos = new_pos
            collision = False
        else:
            collision = True
        
        # Calcular reward
        reward = self._calculate_reward(collision)
        
        # Check termination
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        terminated = distance_to_goal < 3.0  # Chegou ao goal
        truncated = self.step_count >= self.max_steps  # Timeout
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        '''Estado: [x, y, goal_x, goal_y, density]'''
        return np.array([
            self.current_pos[0],
            self.current_pos[1], 
            self.goal_pos[0],
            self.goal_pos[1],
            self.obstacle_density
        ], dtype=np.float32)
    
    def _calculate_reward(self, collision):
        '''Reward function'''
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        
        # Reward components
        reward = 0.0
        
        # Distance reward (closer = better)
        reward -= distance_to_goal * 0.01
        
        # Goal reward
        if distance_to_goal < 3.0:
            reward += 100.0
        
        # Collision penalty
        if collision:
            reward -= 10.0
            
        # Living penalty
        reward -= 0.1
        
        return reward

def train_ppo_model(total_timesteps=50000, density=0.35):
    '''Treinar modelo PPO'''
    print(f'Iniciando training PPO para densidade {density}')
    print(f'Timesteps: {total_timesteps}')
    print('='*50)
    
    # Criar ambiente
    def make_env():
        return NavigationGymEnv(obstacle_density=density)
    
    # Vectorized environment
    env = make_vec_env(make_env, n_envs=4)
    
    # Modelo PPO
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Training
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    training_time = time.time() - start_time
    
    # Salvar modelo
    model_path = f'models/ppo_density_{density}_{total_timesteps}.zip'
    model.save(model_path)
    
    print(f'\\n✅ Training completo!')
    print(f'✅ Tempo: {training_time:.1f}s')
    print(f'✅ Modelo salvo: {model_path}')
    
    return model, model_path

def test_trained_model(model_path, n_episodes=10):
    '''Testar modelo treinado'''
    print(f'Testando modelo: {model_path}')
    
    env = NavigationGymEnv(obstacle_density=0.35)
    model = PPO.load(model_path)
    
    successes = 0
    total_time = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Check success (chegou perto do goal)
        distance = np.linalg.norm(env.current_pos - env.goal_pos)
        if distance < 3.0:
            successes += 1
        
        total_time += steps
    
    success_rate = successes / n_episodes
    avg_steps = total_time / n_episodes
    
    print(f'Success rate: {success_rate:.1%}')
    print(f'Avg steps: {avg_steps:.1f}')
    
    return success_rate

if __name__ == '__main__':
    # Training rápido para teste
    model, model_path = train_ppo_model(total_timesteps=25000, density=0.35)
    
    # Teste
    test_trained_model(model_path)
