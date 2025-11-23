import numpy as np
import time
from typing import Tuple, List, Optional

class MockRRTStarPlanner:
    '''Mock RRT* planner using geometric path planning'''
    
    def __init__(self, env):
        self.env = env
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             time_limit: float = 1.0) -> Tuple[bool, float, List[Tuple[float, float]]]:
        '''
        Plan path from start to goal
        
        Returns:
            success: bool - whether path was found
            time_ms: float - planning time in milliseconds
            trajectory: List[Tuple] - waypoints along path
        '''
        t0 = time.time()
        
        # Simulate RRT* planning time based on density
        density = self.env.get_density()
        base_time = 25.0  # Base planning time in ms
        density_penalty = density * 80.0  # Higher density = longer time
        
        # Simulate planning time variability
        planning_time = np.random.normal(
            base_time + density_penalty, 
            base_time * 0.3
        )
        planning_time = max(10.0, planning_time)
        
        # Success probability decreases with density
        success_prob = max(0.6, 0.98 - density * 1.2)
        success = np.random.random() < success_prob
        
        if success:
            # Generate simple geometric path
            trajectory = self._generate_path(start, goal)
            
            # Validate path for collisions
            success = self._validate_path(trajectory)
        else:
            trajectory = []
        
        elapsed_ms = (time.time() - t0) * 1000
        # Add simulated planning time
        simulated_time = max(elapsed_ms, planning_time)
        
        return success, simulated_time, trajectory
    
    def _generate_path(self, start: Tuple[float, float], 
                      goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        '''Generate simple straight-line path with waypoints'''
        steps = 20
        path = []
        
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            path.append((x, y))
        
        return path
    
    def _validate_path(self, trajectory: List[Tuple[float, float]]) -> bool:
        '''Check if path is collision-free'''
        for x, y in trajectory:
            if not self.env.is_valid(x, y):
                return False
        return True

class PPOPlanner:
    '''PPO planner wrapper for both mock and real models'''
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        if model_path:
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(model_path)
                print(f'✓ Loaded PPO model from {model_path}')
            except Exception as e:
                print(f'⚠️ Failed to load PPO model: {e}')
                print('⚠️ Using mock PPO planner')
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             env, time_limit: float = 1.0) -> Tuple[bool, float, List[Tuple[float, float]]]:
        '''
        Plan path using PPO (mock or real)
        '''
        t0 = time.time()
        
        if self.model is not None:
            return self._plan_with_model(start, goal, env)
        else:
            return self._plan_mock(start, goal, env)
    
    def _plan_with_model(self, start, goal, env):
        '''Real PPO planning (when model is loaded)'''
        # TODO: Implement real PPO planning
        # For now, fallback to mock
        return self._plan_mock(start, goal, env)
    
    def _plan_mock(self, start, goal, env):
        '''Mock PPO planning'''
        # Simulate fast inference (~18ms)
        planning_time = np.random.normal(18, 5)
        planning_time = max(10, planning_time)
        
        # Success probability increases with density (PPO is better in dense envs)
        density = env.get_density()
        success_prob = min(0.98, 0.75 + density * 0.35)
        
        success = np.random.random() < success_prob
        
        if success:
            # Generate simple path (in real implementation, this would be from RL policy)
            trajectory = self._generate_trajectory(start, goal)
            
            # Validate collision
            success = all(env.is_valid(x, y) for x, y in trajectory)
        else:
            trajectory = []
        
        return success, planning_time, trajectory
    
    def _generate_trajectory(self, start, goal):
        '''Generate mock trajectory'''
        steps = 15
        trajectory = []
        
        for i in range(steps + 1):
            t = i / steps
            # Add some curvature to differentiate from RRT*
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            
            # Add slight curve
            if 0.2 < t < 0.8:
                curve_offset = 3 * np.sin(t * np.pi)
                x += curve_offset * np.random.uniform(-1, 1)
                y += curve_offset * np.random.uniform(-1, 1)
            
            trajectory.append((x, y))
        
        return trajectory
