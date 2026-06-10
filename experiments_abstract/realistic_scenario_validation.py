"""
Complex Realistic Environments - Final Enhancement
Automotive-inspired scenarios with structured obstacles
"""

import sys
sys.path.append("../validation_abstract")
import numpy as np
import pandas as pd
from adaptive_switcher_fixed import AdaptiveSwitcher

class AutomotiveEnvironment:
    """Realistic automotive environment generator"""
    
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    
    def create_intersection_scenario(self):
        """Urban intersection with structured obstacles"""
        self.grid.fill(0)
        
        # Roads (clear areas)
        road_width = 8
        center = self.grid_size // 2
        
        # Horizontal road
        self.grid[center-road_width//2:center+road_width//2, :] = 0
        
        # Vertical road  
        self.grid[:, center-road_width//2:center+road_width//2] = 0
        
        # Buildings (obstacles)
        for i in range(4):
            for j in range(4):
                if i < 2 and j < 2:  # Top-left quadrant
                    start_x, start_y = i*20+5, j*20+5
                    end_x, end_y = start_x+15, start_y+15
                    if end_x < center-road_width//2 and end_y < center-road_width//2:
                        self.grid[start_y:end_y, start_x:end_x] = 1
        
        return self.grid.copy()
    
    def create_highway_scenario(self):
        """Highway with lane obstacles"""
        self.grid.fill(0)
        
        # Highway lanes
        lane_width = 6
        lanes = 3
        
        for lane in range(lanes):
            y_start = 20 + lane * (lane_width + 2)
            y_end = y_start + lane_width
            
            # Add vehicles (obstacles) sporadically
            for x in range(10, self.grid_size-10, 20):
                if np.random.random() < 0.3:  # 30% chance of vehicle
                    vehicle_len = 4
                    self.grid[y_start:y_end, x:x+vehicle_len] = 1
        
        return self.grid.copy()
    
    def create_parking_lot_scenario(self):
        """Parking lot with parked cars"""
        self.grid.fill(0)
        
        # Parking spaces grid
        space_width, space_height = 3, 6
        aisle_width = 4
        
        for row in range(0, self.grid_size-space_height, space_height+aisle_width):
            for col in range(0, self.grid_size-space_width, space_width+1):
                if np.random.random() < 0.7:  # 70% occupied
                    self.grid[row:row+space_height, col:col+space_width] = 1
        
        return self.grid.copy()
    
    def get_density(self):
        """Calculate obstacle density"""
        return float(np.mean(self.grid))
    
    def is_valid(self, x, y):
        """Check if position is valid"""
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.grid_size and 0 <= yi < self.grid_size):
            return False
        return self.grid[yi, xi] == 0

def realistic_scenario_validation():
    """Test framework on realistic automotive scenarios"""
    print("=== REALISTIC SCENARIO VALIDATION ===")
    
    scenarios = [
        ("Urban Intersection", "intersection"),
        ("Highway", "highway"), 
        ("Parking Lot", "parking")
    ]
    
    realistic_results = []
    n_trials = 20
    
    for scenario_name, scenario_type in scenarios:
        print(f"🚗 Testing {scenario_name} scenario")
        
        for trial in range(n_trials):
            # Create realistic environment
            env = AutomotiveEnvironment()
            
            if scenario_type == "intersection":
                env.create_intersection_scenario()
            elif scenario_type == "highway":
                env.create_highway_scenario()
            else:  # parking
                env.create_parking_lot_scenario()
            
            # Test adaptive framework
            switcher = AdaptiveSwitcher()
            switcher.set_environment(env)
            
            # Scenario-appropriate start/goal
            if scenario_type == "intersection":
                start = (10, 50)  # Enter from west
                goal = (90, 50)   # Exit to east
            elif scenario_type == "highway":
                start = (5, 30)   # Highway entrance
                goal = (95, 30)   # Highway exit
            else:  # parking
                start = (5, 5)    # Parking lot entrance
                goal = (50, 50)   # Find parking space
            
            success, time_ms, path, selected = switcher.plan(start, goal, env)
            
            realistic_results.append({
                'scenario': scenario_name,
                'scenario_type': scenario_type,
                'trial': trial,
                'density': env.get_density(),
                'success': success,
                'selected_planner': selected,
                'planning_time_ms': time_ms,
                'path_length': len(path) if path else 0,
                'realistic_challenge': True
            })
    
    # Analysis
    realistic_df = pd.DataFrame(realistic_results)
    
    print(f"\n📊 REALISTIC SCENARIO RESULTS:")
    
    # Performance by scenario
    scenario_performance = realistic_df.groupby('scenario').agg({
        'success': ['mean', 'count'],
        'planning_time_ms': 'mean',
        'density': 'mean'
    }).round(3)
    
    print(f"\nPerformance by scenario:")
    print(scenario_performance)
    
    # Planner selection by scenario
    planner_selection = realistic_df.groupby(['scenario', 'selected_planner']).size().unstack(fill_value=0)
    print(f"\nPlanner selection by scenario:")
    print(planner_selection)
    
    # Overall performance
    overall_success = realistic_df['success'].mean()
    print(f"\nOverall realistic scenario success rate: {overall_success:.1%}")
    
    # Save results
    realistic_df.to_csv('results/realistic_scenario_results.csv', index=False)
    
    return realistic_df, {
        'overall_realistic_success': overall_success,
        'best_scenario': scenario_performance['success']['mean'].idxmax(),
        'most_challenging': scenario_performance['success']['mean'].idxmin(),
        'demonstrates_real_world_applicability': overall_success > 0.6
    }

if __name__ == "__main__":
    results, insights = realistic_scenario_validation()
    print(f"\n✅ REALISTIC SCENARIO VALIDATION COMPLETE!")
    
    for key, value in insights.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
