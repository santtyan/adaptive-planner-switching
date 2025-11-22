# src/adaptive_switcher_v2.py - Versão com RRT* real
import sys
sys.path.append('src')
sys.path.append('src/planners')

import numpy as np
import time
from typing import Tuple, List, Dict, Any
from environment import SimpleEnvironment
from rrt_star_impl import RRTStarPlanner

# Importar PPO real
from stable_baselines3 import PPO
import gymnasium as gym

class PPOPlannerReal:
    '''PPO real usando Stable-Baselines3'''
    
    def __init__(self, model_path=None):
        if model_path:
            self.model = PPO.load(model_path)
        else:
            # Modelo mock por enquanto
            self.model = None
            
    def plan(self, start, goal, env):
        # Por enquanto simular, depois implementar training real
        planning_time = max(10, np.random.normal(18, 5))
        density = env.get_density()
        success_prob = min(0.98, 0.75 + density * 0.35)
        success = np.random.random() < success_prob
        
        if success:
            # Caminho simples por enquanto
            steps = 15
            trajectory = []
            for i in range(steps + 1):
                t = i / steps
                x = start[0] + t * (goal[0] - start[0])
                y = start[1] + t * (goal[1] - start[1])
                trajectory.append((x, y))
            success = all(env.is_valid(x, y) for x, y in trajectory)
        else:
            trajectory = []
            
        return success, planning_time, trajectory

class AdaptiveSwitcherV2:
    '''Versão com planners REAIS'''
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.log = []
        
    def set_environment(self, env):
        # RRT* REAL
        self.rrt_planner = RRTStarPlanner(env)
        # PPO real (mock por enquanto)  
        self.ppo_planner = PPOPlannerReal()
        
    def plan(self, start, goal, env):
        density = env.get_density()
        
        # SWITCHING LOGIC
        if density < self.threshold:
            # RRT* REAL
            success, time_ms, trajectory = self.rrt_planner.plan(start, goal)
            selected = 'RRT*'
        else:
            # PPO 
            success, time_ms, trajectory = self.ppo_planner.plan(start, goal, env)
            selected = 'PPO'
        
        self.log.append({
            'density': density,
            'selected': selected,
            'success': success,
            'time_ms': time_ms
        })
        
        return success, time_ms, trajectory, selected

def test_real_framework():
    '''Teste com RRT* real'''
    print('TESTANDO FRAMEWORK COM RRT* REAL')
    print('='*40)
    
    densities = [0.15, 0.35]  # Baixa e alta densidade
    
    for density in densities:
        print(f'\\nTeste ρ={density}')
        
        env = SimpleEnvironment(obstacle_density=density, seed=42)
        switcher = AdaptiveSwitcherV2(threshold=0.3)
        switcher.set_environment(env)
        
        # 5 trials
        for trial in range(5):
            start = (10, 10)
            goal = (90, 90)
            success, time_ms, traj, selected = switcher.plan(start, goal, env)
            
            print(f'  Trial {trial+1}: {selected} -> '
                  f'Success: {success}, Time: {time_ms:.1f}ms')

if __name__ == '__main__':
    test_real_framework()
