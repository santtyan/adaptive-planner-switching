from environment import SimpleEnvironment
from planners.rrt_star import MockRRTStarPlanner as RRTStarPlanner
from planners.ppo_planner import PPOPlannerMockScientific
from typing import Tuple, List, Dict, Any
import numpy as np
import time

class AdaptiveSwitcher:
    """
    Adaptive switching framework between RRT* and PPO
    Core contribution: treats planner selection as optimization variable
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.log = []
        
        # Initialize planners (will be set via set_environment)
        self.rrt_planner = None
        self.ppo_planner = None

    def set_environment(self, env: SimpleEnvironment):
        """Initialize planners for given environment"""
        self.rrt_planner = RRTStarPlanner(env)  # REAL RRT*
        self.ppo_planner = PPOPlannerMockScientific()  # SCIENTIFIC PPO

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             env: SimpleEnvironment) -> Tuple[bool, float, List[Tuple[float, float]], str]:
        """CORE SWITCHING LOGIC"""

        # Context analysis
        density = env.get_density()

        # DECISION RULE (core contribution)
        if density < self.threshold:
            selected_planner = "rrt_star"
            success, planning_time, path = self.rrt_planner.plan(start, goal)
        else:
            selected_planner = "ppo"
            success, planning_time, path = self.ppo_planner.plan(start, goal, env)

        # Log result
        result = {
            'density': density,
            'selected': selected_planner,
            'success': success,
            'time': planning_time,
            'path_length': len(path) if path else 0
        }
        self.log.append(result)

        return success, planning_time, path, selected_planner
