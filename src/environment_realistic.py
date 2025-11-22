# src/environment_realistic.py - Cenários automotivos reais
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

class AutomotiveEnvironment:
    '''Ambientes realísticos de navegação autônoma'''
    
    def __init__(self, scenario='urban_intersection', seed=42):
        np.random.seed(seed)
        self.scenario = scenario
        self.grid_size = 200  # Maior resolução
        self.obstacles = []
        self.dynamic_obstacles = []
        self.scenario_config = self._load_scenario(scenario)
        self._generate_realistic_environment()
        
    def _load_scenario(self, scenario):
        '''Configurações de cenários automotivos'''
        scenarios = {
            'urban_intersection': {
                'static_vehicles': 15,
                'pedestrians': 8,
                'traffic_density': 0.25,
                'intersection_points': [(100, 100)],
                'road_width': 12
            },
            'highway_merge': {
                'static_vehicles': 20,
                'pedestrians': 2,
                'traffic_density': 0.35,
                'merge_point': (150, 100),
                'road_width': 8
            },
            'parking_lot': {
                'static_vehicles': 30,
                'pedestrians': 5,
                'traffic_density': 0.45,
                'parking_spots': 40,
                'road_width': 6
            }
        }
        return scenarios.get(scenario, scenarios['urban_intersection'])
    
    def _generate_realistic_environment(self):
        '''Gerar ambiente baseado em cenário automotivo'''
        if self.scenario == 'urban_intersection':
            self._generate_urban_intersection()
        elif self.scenario == 'highway_merge':
            self._generate_highway_merge()
        elif self.scenario == 'parking_lot':
            self._generate_parking_lot()
            
    def _generate_urban_intersection(self):
        '''Cenário: Interseção urbana complexa'''
        # Buildings (static obstacles)
        buildings = [
            (20, 20, 60, 60),    # Building 1
            (140, 20, 60, 60),   # Building 2
            (20, 140, 60, 60),   # Building 3
            (140, 140, 60, 60)   # Building 4
        ]
        
        for x, y, w, h in buildings:
            self.obstacles.extend(self._create_rectangle(x, y, w, h))
        
        # Parked vehicles
        parked_positions = [
            (85, 30), (95, 30), (105, 30),  # Street parking
            (85, 170), (95, 170), (105, 170),
            (30, 85), (30, 95), (30, 105),
            (170, 85), (170, 95), (170, 105)
        ]
        
        for x, y in parked_positions[:self.scenario_config['static_vehicles']]:
            self.obstacles.extend(self._create_vehicle(x, y))
        
        # Pedestrian crossings (higher density areas)
        self.pedestrian_zones = [
            (80, 95, 40, 10),  # Horizontal crossing
            (95, 80, 10, 40)   # Vertical crossing
        ]
    
    def _generate_highway_merge(self):
        '''Cenário: Merge em highway'''
        # Highway barriers
        barrier_points = []
        # Main highway barriers
        for i in range(0, 200, 2):
            barrier_points.extend([(i, 85), (i, 115)])  # Top and bottom
        
        # Merge lane barriers
        for i in range(120, 180, 2):
            barrier_points.extend([(i, 60), (i, 140)])
        
        self.obstacles = barrier_points
        
        # Moving vehicles (simulated as static for planning)
        vehicle_positions = [
            (50, 100), (80, 100), (110, 100), (140, 100),  # Main lane
            (130, 70), (150, 70), (170, 70),  # Merge lane
        ]
        
        for x, y in vehicle_positions[:self.scenario_config['static_vehicles']]:
            self.obstacles.extend(self._create_vehicle(x, y))
    
    def _generate_parking_lot(self):
        '''Cenário: Estacionamento complexo'''
        # Parking structure
        for row in range(3):
            for col in range(8):
                x = 30 + col * 20
                y = 40 + row * 40
                if np.random.random() < 0.7:  # 70% occupied
                    self.obstacles.extend(self._create_vehicle(x, y))
        
        # Pillars/structural elements
        pillars = [(60, 80), (120, 80), (60, 140), (120, 140)]
        for x, y in pillars:
            self.obstacles.extend(self._create_rectangle(x-2, y-2, 4, 4))
    
    def _create_vehicle(self, x, y):
        '''Criar representação de veículo (4x2 meters)'''
        vehicle_points = []
        for dx in range(-2, 3):
            for dy in range(-1, 2):
                vehicle_points.append((x + dx, y + dy))
        return vehicle_points
    
    def _create_rectangle(self, x, y, width, height):
        '''Criar retângulo de obstáculos'''
        points = []
        for i in range(x, x + width):
            for j in range(y, y + height):
                points.append((i, j))
        return points
    
    def is_valid(self, x, y):
        '''Check if position is collision-free'''
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        
        # Check static obstacles
        for obs_x, obs_y in self.obstacles:
            if abs(x - obs_x) < 1.0 and abs(y - obs_y) < 1.0:
                return False
        
        return True
    
    def get_density(self):
        '''Calculate realistic density based on scenario'''
        return self.scenario_config['traffic_density']
    
    def visualize(self, path=None, save_path=None):
        '''Visualizar ambiente realístico'''
        plt.figure(figsize=(10, 10))
        
        # Draw obstacles
        if self.obstacles:
            obs_x, obs_y = zip(*self.obstacles)
            plt.scatter(obs_x, obs_y, c='red', s=1, alpha=0.6, label='Obstacles')
        
        # Draw path if provided
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
            plt.scatter([path[0][0], path[-1][0]], [path[0][1], path[-1][1]], 
                       c=['green', 'blue'], s=100, label='Start/Goal')
        
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        plt.title(f'Automotive Environment: {self.scenario}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def test_realistic_environments():
    '''Teste ambientes realísticos'''
    scenarios = ['urban_intersection', 'highway_merge', 'parking_lot']
    
    for scenario in scenarios:
        print(f'\\n🚗 Testing: {scenario}')
        env = AutomotiveEnvironment(scenario=scenario)
        
        # Test navigation task
        if scenario == 'urban_intersection':
            start, goal = (10, 100), (190, 100)  # Cross intersection
        elif scenario == 'highway_merge':
            start, goal = (140, 70), (180, 100)  # Merge maneuver
        else:  # parking_lot
            start, goal = (10, 100), (190, 100)  # Navigate through lot
        
        print(f'  Density: {env.get_density()}')
        print(f'  Obstacles: {len(env.obstacles)}')
        print(f'  Start: {start}, Goal: {goal}')
        
        # Quick validation
        print(f'  Start valid: {env.is_valid(start[0], start[1])}')
        print(f'  Goal valid: {env.is_valid(goal[0], goal[1])}')

if __name__ == '__main__':
    test_realistic_environments()
