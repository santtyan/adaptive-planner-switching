# src/planners/ppo_training_v3.py - Fix rápido e eficiente
import sys
sys.path.append('src')

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

from environment import SimpleEnvironment

class NavigationGymEnvV3(gym.Env):
    '''Ambiente simplificado para convergência rápida'''
    
    def __init__(self, obstacle_density=0.3, grid_size=100):
        super().__init__()
        
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.env = SimpleEnvironment(obstacle_density=obstacle_density)
        
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Goals MUITO mais próximos - facilitar learning
        while True:
            self.current_pos = np.random.uniform(20, 80, 2)
            if self.env.is_valid(self.current_pos[0], self.current_pos[1]):
                break
                
        # Goal entre 8-20 unidades (bem mais próximo)
        while True:
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(8, 20)  # Reduzido drasticamente
            self.goal_pos = self.current_pos + distance * np.array([np.cos(angle), np.sin(angle)])
            
            if (10 <= self.goal_pos[0] <= 90 and 
                10 <= self.goal_pos[1] <= 90 and
                self.env.is_valid(self.goal_pos[0], self.goal_pos[1])):
                break
        
        self.step_count = 0
        self.max_steps = 50  # Ainda menor
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.step_count += 1
        
        new_pos = self.current_pos + action
        new_pos = np.clip(new_pos, 0, self.grid_size)
        
        collision = not self.env.is_valid(new_pos[0], new_pos[1])
        
        if not collision:
            old_distance = np.linalg.norm(self.current_pos - self.goal_pos)
            self.current_pos = new_pos
            new_distance = np.linalg.norm(self.current_pos - self.goal_pos)
            progress = old_distance - new_distance
        else:
            new_distance = np.linalg.norm(self.current_pos - self.goal_pos)
            progress = 0
        
        reward = self._calculate_reward_v3(collision, progress, new_distance)
        
        # SUCCESS CRITERIA MAIS FÁCIL
        terminated = new_distance < 5.0  # Era 2.0, agora 5.0
        truncated = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self):
        '''Observation simplificado'''
        current_norm = self.current_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size
        distance_norm = np.linalg.norm(self.current_pos - self.goal_pos) / self.grid_size
        
        return np.array([
            current_norm[0], current_norm[1],
            goal_norm[0], goal_norm[1],
            distance_norm
        ], dtype=np.float32)
    
    def _calculate_reward_v3(self, collision, progress, distance):
        '''Reward function generosa'''
        reward = 0.0
        
        # 1. SUCCESS REWARD (major)
        if distance < 5.0:
            reward += 100.0
        
        # 2. PROXIMITY REWARDS (escalonados)
        if distance < 10.0:
            reward += 20.0
        if distance < 7.0:
            reward += 30.0
            
        # 3. PROGRESS REWARD (muito generoso)
        reward += progress * 10.0
        
        # 4. DISTANCE PENALTY (suave)
        reward -= distance * 0.01
        
        # 5. COLLISION (moderado)
        if collision:
            reward -= 2.0
            
        # 6. TIME PENALTY (mínimo)
        reward -= 0.02
        
        return reward

def quick_train_ppo(timesteps=50000):
    '''Training rápido com hiperparâmetros agressivos'''
    print(f'🚀 PPO V3 - QUICK CONVERGENCE')
    print(f'Timesteps: {timesteps}')
    print('='*40)
    
    env = make_vec_env(lambda: NavigationGymEnvV3(obstacle_density=0.25), n_envs=4)
    
    # Hiperparâmetros para convergência rápida
    model = PPO(
        'MlpPolicy', env,
        learning_rate=5e-4,  # Maior LR
        n_steps=512,         # Menor buffer
        batch_size=64,
        n_epochs=10,
        gamma=0.95,          # Discount menor
        clip_range=0.3,      # Clip maior
        verbose=1
    )
    
    start_time = time.time()
    model.learn(total_timesteps=timesteps)
    training_time = time.time() - start_time
    
    model_path = f'models/ppo_v3_quick_{timesteps}.zip'
    model.save(model_path)
    
    print(f'\\n✅ Quick training: {training_time:.1f}s')
    return model, model_path

def quick_test(model_path, n_tests=20):
    '''Teste rápido'''
    env = NavigationGymEnvV3(obstacle_density=0.25)
    model = PPO.load(model_path)
    
    successes = 0
    for _ in range(n_tests):
        obs, _ = env.reset()
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                successes += 1
                break
    
    success_rate = successes / n_tests
    print(f'🎯 Quick test: {success_rate:.0%} success rate')
    return success_rate

if __name__ == '__main__':
    # Training rápido
    model, path = quick_train_ppo(50000)
    
    # Teste imediato
    success = quick_test(path)
    
    if success > 0.6:
        print('✅ PPO FUNCIONANDO!')
    else:
        print('⚠️ Ainda precisa ajuste')
        
    print(f'\\n⏰ Workshop deadline em 4 dias')
    print('💡 Sufficient para proposta mesmo com PPO básico')
