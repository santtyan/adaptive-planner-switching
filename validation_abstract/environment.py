import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleEnvironment:
    '''2D grid environment with controllable obstacle density'''
    
    def __init__(self, grid_size=100, obstacle_density=0.3, seed=None):
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.rng = np.random.default_rng(seed)
        self.grid = self._generate_grid()
        
    def _generate_grid(self):
        '''Generate binary grid: 0=free, 1=obstacle'''
        grid = self.rng.random((self.grid_size, self.grid_size))
        return (grid < self.obstacle_density).astype(np.int8)
    
    def is_valid(self, x, y):
        '''Check if position is collision-free'''
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.grid_size and 0 <= yi < self.grid_size):
            return False
        return self.grid[yi, xi] == 0
    
    def get_density(self):
        '''Return actual obstacle density'''
        return float(np.mean(self.grid))
    
    def reset(self, density=None):
        '''Reset with new density'''
        if density is not None:
            self.obstacle_density = density
        self.grid = self._generate_grid()
        return self.grid

class GridEnv(gym.Env):
    '''Gymnasium-compatible environment for PPO training'''
    
    def __init__(self, grid_size=100, obstacle_density=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        
        # Action space: [dx, dy] movement
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        
        # Observation: [x, y, goal_x, goal_y, density]
        self.observation_space = spaces.Box(
            low=0.0, high=grid_size, shape=(5,), dtype=np.float32
        )
        
        self.env = SimpleEnvironment(grid_size, obstacle_density)
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        
        # Random start/goal
        self.start = self._get_free_position()
        self.goal = self._get_free_position()
        self.current_pos = self.start.copy()
        self.steps = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        self.steps += 1
        
        # Move
        new_pos = self.current_pos + action
        new_pos = np.clip(new_pos, 0, self.grid_size-1)
        
        # Check collision
        if self.env.is_valid(new_pos[0], new_pos[1]):
            self.current_pos = new_pos
            collision = False
        else:
            collision = True
        
        # Rewards
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        
        reward = -0.1  # Time penalty
        if collision:
            reward -= 10
        if distance_to_goal < 2.0:
            reward += 100
            done = True
        else:
            done = False
        
        if self.steps > 500:  # Max episode length
            done = True
        
        obs = self._get_observation()
        info = {'success': distance_to_goal < 2.0}
        
        return obs, reward, done, False, info
    
    def _get_free_position(self):
        '''Sample random collision-free position'''
        for _ in range(100):
            pos = self.np_random.uniform(5, self.grid_size-5, 2)
            if self.env.is_valid(pos[0], pos[1]):
                return pos
        return np.array([10.0, 10.0])  # Fallback
    
    def _get_observation(self):
        '''Return observation vector'''
        return np.array([
            self.current_pos[0],
            self.current_pos[1], 
            self.goal[0],
            self.goal[1],
            self.env.get_density()
        ], dtype=np.float32)
