# src/planners/ppo_training_v2.py - Versão otimizada
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import time
import math

from environment import SimpleEnvironment

class NavigationGymEnvV2(gym.Env):
    '''Ambiente otimizado para convergência PPO'''
    
    def __init__(self, obstacle_density=0.3, grid_size=100):
        super().__init__()
        
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.env = SimpleEnvironment(obstacle_density=obstacle_density)
        
        # Action space menor para controle mais fino
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space normalizado
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Start/goal mais próximos para facilitar learning
        while True:
            self.current_pos = np.random.uniform(10, self.grid_size-10, 2)
            if self.env.is_valid(self.current_pos[0], self.current_pos[1]):
                break
                
        while True:
            # Goal entre 15-35 unidades de distância
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(15, 35)
            self.goal_pos = self.current_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            
            if (0 <= self.goal_pos[0] <= self.grid_size and 
                0 <= self.goal_pos[1] <= self.grid_size and
                self.env.is_valid(self.goal_pos[0], self.goal_pos[1])):
                break
        
        self.step_count = 0
        self.max_steps = 100  # Reduzido para episodes mais rápidos
        self.initial_distance = np.linalg.norm(self.goal_pos - self.current_pos)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Aplicar ação com clipping
        new_pos = self.current_pos + action
        new_pos = np.clip(new_pos, 0, self.grid_size)
        
        # Check collision
        collision = not self.env.is_valid(new_pos[0], new_pos[1])
        
        if not collision:
            old_distance = np.linalg.norm(self.current_pos - self.goal_pos)
            self.current_pos = new_pos
            new_distance = np.linalg.norm(self.current_pos - self.goal_pos)
            distance_improvement = old_distance - new_distance
        else:
            distance_improvement = 0
            new_distance = np.linalg.norm(self.current_pos - self.goal_pos)
        
        # Reward otimizado
        reward = self._calculate_reward_v2(collision, distance_improvement, new_distance)
        
        # Termination
        terminated = new_distance < 2.0  # Goal atingido
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        '''Estado normalizado'''
        current_norm = self.current_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size
        
        distance = np.linalg.norm(self.current_pos - self.goal_pos)
        distance_norm = distance / self.grid_size
        
        direction = (self.goal_pos - self.current_pos)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        return np.array([
            current_norm[0],    # x normalizado
            current_norm[1],    # y normalizado  
            goal_norm[0],       # goal_x normalizado
            goal_norm[1],       # goal_y normalizado
            distance_norm,      # distância normalizada
            self.obstacle_density  # densidade
        ], dtype=np.float32)
    
    def _calculate_reward_v2(self, collision, distance_improvement, current_distance):
        '''Reward function otimizada'''
        reward = 0.0
        
        # 1. Goal reward (dominante)
        if current_distance < 2.0:
            reward += 100.0
        
        # 2. Progress reward (encoraja aproximação)
        reward += distance_improvement * 5.0
        
        # 3. Distance penalty (suave)
        reward -= current_distance * 0.02
        
        # 4. Collision penalty (forte)
        if collision:
            reward -= 5.0
        
        # 5. Time penalty (leve)
        reward -= 0.05
        
        return reward

def train_ppo_optimized(total_timesteps=100000, density=0.35):
    '''Training otimizado PPO'''
    print(f'🚀 PPO Training Otimizado')
    print(f'Densidade: {density}')
    print(f'Timesteps: {total_timesteps}')
    print('='*50)
    
    # Ambiente otimizado
    def make_env():
        return NavigationGymEnvV2(obstacle_density=density)
    
    env = make_vec_env(make_env, n_envs=8)  # Mais environments paralelos
    
    # Hiperparâmetros otimizados
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-4,  # Learning rate menor
        n_steps=1024,        # Steps reduzido  
        batch_size=128,      # Batch maior
        n_epochs=20,         # Mais epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Entropia para exploração
        verbose=1
    )
    
    # Callback para parar quando atingir sucesso
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=50.0, verbose=1
    )
    
    eval_callback = EvalCallback(
        make_env(),
        best_model_save_path='./models/',
        log_path='./models/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )
    
    # Training
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    training_time = time.time() - start_time
    
    # Salvar modelo final
    model_path = f'models/ppo_optimized_density_{density}_{total_timesteps}.zip'
    model.save(model_path)
    
    print(f'\\n✅ Training completo!')
    print(f'✅ Tempo: {training_time:.1f}s')
    print(f'✅ Modelo salvo: {model_path}')
    
    return model, model_path

def test_optimized_model(model_path, n_episodes=20):
    '''Teste extensivo do modelo'''
    print(f'\\n🧪 Testando modelo otimizado: {model_path}')
    
    env = NavigationGymEnvV2(obstacle_density=0.35)
    model = PPO.load(model_path)
    
    successes = 0
    total_steps = 0
    total_rewards = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_reward += reward
        
        # Check success
        distance = np.linalg.norm(env.current_pos - env.goal_pos)
        if distance < 2.0:
            successes += 1
        
        total_steps += steps
        total_rewards += episode_reward
    
    success_rate = successes / n_episodes
    avg_steps = total_steps / n_episodes  
    avg_reward = total_rewards / n_episodes
    
    print(f'✅ Success rate: {success_rate:.1%}')
    print(f'✅ Avg steps: {avg_steps:.1f}')
    print(f'✅ Avg reward: {avg_reward:.1f}')
    
    return success_rate

if __name__ == '__main__':
    # Training otimizado (100k timesteps)
    model, model_path = train_ppo_optimized(total_timesteps=100000, density=0.35)
    
    # Teste extensivo
    success_rate = test_optimized_model(model_path)
    
    print(f'\\n🎯 META: Success rate > 70%')
    if success_rate > 0.7:
        print('✅ PPO CONVERGIU COM SUCESSO!')
    else:
        print('⚠️ Precisa mais training ou ajuste')
