import sys
sys.path.append('src')

from environment import SimpleEnvironment
from typing import Tuple, List, Dict, Any
import numpy as np
import time

# Import classes directly from planners.py
sys.path.append('src/planners')
from planners import MockRRTStarPlanner, PPOPlanner

class AdaptiveSwitcher:
    '''
    Adaptive switching framework between RRT* and PPO
    Core contribution: treats planner selection as optimization variable
    '''
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.log = []
        
        # Initialize planners (will be set via set_environment)
        self.rrt_planner = None
        self.ppo_planner = None
        
    def set_environment(self, env: SimpleEnvironment):
        '''Initialize planners for given environment'''
        self.rrt_planner = MockRRTStarPlanner(env)
        self.ppo_planner = PPOPlanner()  # Mock for now
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             env: SimpleEnvironment) -> Tuple[bool, float, List[Tuple[float, float]], str]:
        '''CORE SWITCHING LOGIC'''
        
        # Context analysis
        density = env.get_density()
        
        # DECISION RULE (core contribution)
        if density < self.threshold:
            # Low density: use RRT* (good for exploration)  
            success, time_ms, trajectory = self.rrt_planner.plan(start, goal)
            selected = 'RRT*'
        else:
            # High density: use PPO (good for reactivity)
            success, time_ms, trajectory = self.ppo_planner.plan(start, goal, env)
            selected = 'PPO'
        
        # Log decision for analysis
        self.log.append({
            'density': density,
            'threshold': self.threshold,
            'selected': selected,
            'success': success,
            'time_ms': time_ms,
            'start': start,
            'goal': goal
        })
        
        return success, time_ms, trajectory, selected
    
    def get_statistics(self) -> Dict[str, Any]:
        '''Get switching statistics'''
        if not self.log:
            return {}
        
        total_trials = len(self.log)
        rrt_count = sum(1 for entry in self.log if entry['selected'] == 'RRT*')
        ppo_count = total_trials - rrt_count
        
        rrt_success = [e for e in self.log if e['selected'] == 'RRT*' and e['success']]
        ppo_success = [e for e in self.log if e['selected'] == 'PPO' and e['success']]
        
        return {
            'total_trials': total_trials,
            'rrt_selections': rrt_count,
            'ppo_selections': ppo_count,
            'rrt_success_rate': len(rrt_success) / rrt_count if rrt_count > 0 else 0.0,
            'ppo_success_rate': len(ppo_success) / ppo_count if ppo_count > 0 else 0.0,
            'overall_success_rate': sum(e['success'] for e in self.log) / total_trials,
            'avg_time_ms': sum(e['time_ms'] for e in self.log) / total_trials
        }
    
    def clear_log(self):
        '''Clear decision log'''
        self.log = []
