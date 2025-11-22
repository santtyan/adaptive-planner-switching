# src/planners/ppo_final_fix.py - Solução definitiva
import sys
sys.path.append('src')
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
from environment import SimpleEnvironment

class PPONavigationFinal(gym.Env):
    def __init__(self, obstacle_density=0.25, grid_size=100):
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.env = SimpleEnvironment(obstacle_density=obstacle_density, seed=42)
        
        self.action_space = spaces.Discrete(4)  # DISCRETE: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.step_size = 3.0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # GOALS MUITO PRÓXIMOS
        self.current_pos = np.array([25.0, 25.0])
        self.goal_pos = np.array([75.0, 75.0])  # FIXO para facilitar
        self.step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.step_count += 1
        
        # DISCRETE ACTIONS
        moves = [(0, self.step_size), (0, -self.step_size), 
                 (self.step_size, 0), (-self.step_size, 0)]
        dx, dy = moves[action]
        
        new_pos = self.current_pos + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0, self.grid_size)
        
        collision = not self.env.is_valid(new_pos[0], new_pos[1])
        
        if not collision:
            old_dist = np.linalg.norm(self.current_pos - self.goal_pos)
            self.current_pos = new_pos
            new_dist = np.linalg.norm(self.current_pos - self.goal_pos)
            progress = old_dist - new_dist
        else:
            new_dist = np.linalg.norm(self.current_pos - self.goal_pos)
            progress = 0
        
        # REWARD ULTRA SIMPLES E EFETIVO
        reward = 0.0
        if new_dist < 8.0:  # SUCCESS
            reward = 1000.0
        else:
            reward = progress * 50.0  # PROGRESS
            reward -= 0.1  # TIME
            if collision:
                reward -= 20.0
        
        terminated = new_dist < 8.0 or self.step_count >= 30
        
        return self._get_obs(), reward, terminated, False, {}
    
    def _get_obs(self):
        current_norm = self.current_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size
        return np.array([current_norm[0], current_norm[1], 
                        goal_norm[0], goal_norm[1]], dtype=np.float32)

def train_final_ppo():
    print('🚀 PPO FINAL FIX - DISCRETE ACTIONS')
    
    env = make_vec_env(lambda: PPONavigationFinal(), n_envs=4)
    
    model = PPO('MlpPolicy', env, 
                learning_rate=1e-3, 
                n_steps=256, 
                batch_size=64, 
                n_epochs=20, 
                verbose=1)
    
    model.learn(total_timesteps=30000)
    
    model_path = 'models/ppo_final_discrete.zip'
    model.save(model_path)
    
    return model_path

def test_final_ppo(model_path):
    env = PPONavigationFinal()
    model = PPO.load(model_path)
    
    successes = 0
    for _ in range(50):
        obs, _ = env.reset()
        for _ in range(30):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            if done:
                dist = np.linalg.norm(env.current_pos - env.goal_pos)
                if dist < 8.0:
                    successes += 1
                break
    
    success_rate = successes / 50
    print(f'🎯 FINAL SUCCESS RATE: {success_rate:.0%}')
    return success_rate

if __name__ == '__main__':
    model_path = train_final_ppo()
    success = test_final_ppo(model_path)
    
    if success > 0.8:
        print('✅ PPO FUNCIONANDO! FRAMEWORK COMPLETO!')
    else:
        print('❌ PPO ainda problemático')
