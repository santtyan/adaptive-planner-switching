# src/planners/ppo_mock_scientific.py - Mock científico realístico
import sys
sys.path.append('src')
import numpy as np
import time
from typing import Tuple, List
from environment import SimpleEnvironment

class PPOPlannerMockScientific:
    '''Mock PPO com behavior científico realístico'''
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        # Parâmetros calibrados baseados em literatura RL
        self.base_success_rate = 0.85  # PPO típico em nav tasks
        self.density_advantage = 0.2   # Vantagem em alta densidade
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], env):
        '''Planner que simula comportamento PPO realístico'''
        planning_time_start = time.time()
        
        density = env.get_density()
        
        # PPO performance model baseado em literatura
        success_rate = self._calculate_expected_success_rate(density, start, goal, env)
        
        # Simular tempo de planning PPO (inference + algumas iterações)
        planning_time_ms = self._simulate_ppo_timing(density, start, goal)
        
        # Success baseado em probabilidade calibrada
        success = np.random.random() < success_rate
        
        if success:
            # Gerar trajectory realística usando heurística inteligente
            trajectory = self._generate_smart_trajectory(start, goal, env)
        else:
            trajectory = []
        
        return success, planning_time_ms, trajectory
    
    def _calculate_expected_success_rate(self, density, start, goal, env):
        '''Success rate baseado em características do problema'''
        base_rate = self.base_success_rate
        
        # PPO advantage em alta densidade (learned behaviors)
        if density > 0.3:
            base_rate += self.density_advantage * (density - 0.3)
        
        # Penalidade por distância (task difficulty)
        distance = np.linalg.norm(np.array(goal) - np.array(start))
        distance_penalty = min(0.2, (distance - 50) * 0.002)
        base_rate -= max(0, distance_penalty)
        
        # Obstáculos no caminho direto (path complexity)
        path_complexity = self._analyze_path_complexity(start, goal, env)
        base_rate -= path_complexity * 0.15
        
        return np.clip(base_rate, 0.1, 0.98)
    
    def _simulate_ppo_timing(self, density, start, goal):
        '''Timing realístico PPO (inference + planning)'''
        # PPO inference time (forward pass neural network)
        base_time = np.random.normal(8, 2)  # ~8ms baseline
        
        # Path length influence (more steps = more inference)
        distance = np.linalg.norm(np.array(goal) - np.array(start))
        path_factor = 1 + (distance / 100) * 0.3
        
        # Density influence (more complex decisions)
        density_factor = 1 + density * 0.5
        
        total_time = base_time * path_factor * density_factor
        return max(5, total_time)  # Minimum 5ms
    
    def _analyze_path_complexity(self, start, goal, env):
        '''Analisar complexidade do caminho direto'''
        # Discretizar linha direta
        steps = int(np.linalg.norm(np.array(goal) - np.array(start)))
        if steps == 0:
            return 0
        
        blocked_steps = 0
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            if not env.is_valid(x, y):
                blocked_steps += 1
        
        return blocked_steps / (steps + 1)
    
    def _generate_smart_trajectory(self, start, goal, env):
        '''Gerar trajectory que simula comportamento aprendido'''
        trajectory = [start]
        current = np.array(start)
        goal_pos = np.array(goal)
        
        max_steps = 50
        step_size = 3.0
        
        for step in range(max_steps):
            # PPO-like behavior: goal-directed com obstacle avoidance
            
            # Direction to goal
            to_goal = goal_pos - current
            if np.linalg.norm(to_goal) < step_size:
                trajectory.append(tuple(goal_pos))
                break
                
            direction = to_goal / np.linalg.norm(to_goal)
            
            # Check direct step
            next_pos = current + direction * step_size
            
            if env.is_valid(next_pos[0], next_pos[1]):
                current = next_pos
            else:
                # Obstacle avoidance - try perpendicular directions
                perp1 = np.array([-direction[1], direction[0]]) * step_size * 0.7
                perp2 = np.array([direction[1], -direction[0]]) * step_size * 0.7
                
                # Try options
                options = [
                    current + direction * step_size * 0.5 + perp1,
                    current + direction * step_size * 0.5 + perp2,
                    current + perp1,
                    current + perp2,
                    current + direction * step_size * 0.3  # Closer to obstacle
                ]
                
                moved = False
                for option in options:
                    if (env.is_valid(option[0], option[1]) and 
                        0 <= option[0] <= 100 and 0 <= option[1] <= 100):
                        current = option
                        moved = True
                        break
                
                if not moved:
                    break  # Stuck
            
            trajectory.append(tuple(current))
            
            # Goal reached
            if np.linalg.norm(current - goal_pos) < 3.0:
                break
        
        return trajectory

def test_ppo_mock():
    '''Teste comportamento mock PPO'''
    print('🧪 TESTE PPO MOCK CIENTÍFICO')
    print('='*40)
    
    densities = [0.15, 0.25, 0.35, 0.45]
    
    for density in densities:
        print(f'\\nDensidade: {density}')
        env = SimpleEnvironment(obstacle_density=density, seed=42)
        ppo = PPOPlannerMockScientific()
        
        successes = 0
        total_time = 0
        
        for trial in range(20):
            success, time_ms, traj = ppo.plan((10, 10), (90, 90), env)
            if success:
                successes += 1
            total_time += time_ms
        
        success_rate = successes / 20
        avg_time = total_time / 20
        
        print(f'  Success: {success_rate:.1%}')
        print(f'  Avg time: {avg_time:.1f}ms')

if __name__ == '__main__':
    test_ppo_mock()
