# src/adaptive_realistic_evaluation.py - Framework com ambientes realísticos
import sys
sys.path.append('src')
sys.path.append('src/planners')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from environment_realistic import AutomotiveEnvironment
from rrt_star_impl import RRTStarPlanner
from ppo_mock_scientific import PPOPlannerMockScientific

class AdaptiveAutomotiveFramework:
    '''Framework adaptivo para cenários automotivos reais'''
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.results = []
        
    def evaluate_scenario(self, scenario_name, trials=30):
        '''Avaliar framework em cenário automotivo específico'''
        print(f'\\n🚗 Evaluating: {scenario_name.upper()}')
        print('='*40)
        
        env = AutomotiveEnvironment(scenario=scenario_name, seed=42)
        density = env.get_density()
        
        # Setup planners
        rrt_planner = RRTStarPlanner(env)
        ppo_planner = PPOPlannerMockScientific()
        
        # Define realistic navigation tasks for each scenario
        tasks = self._get_scenario_tasks(scenario_name)
        
        scenario_results = []
        
        for task_idx, (start, goal, task_name) in enumerate(tasks):
            print(f'  Task: {task_name}')
            
            task_results = {'RRT*': [], 'PPO': [], 'Adaptive': []}
            
            for trial in range(trials):
                # Test RRT* directly
                rrt_success, rrt_time, rrt_path = rrt_planner.plan(start, goal)
                task_results['RRT*'].append({
                    'success': rrt_success,
                    'time_ms': rrt_time,
                    'path_length': len(rrt_path) if rrt_path else 0
                })
                
                # Test PPO directly  
                ppo_success, ppo_time, ppo_path = ppo_planner.plan(start, goal, env)
                task_results['PPO'].append({
                    'success': ppo_success,
                    'time_ms': ppo_time,
                    'path_length': len(ppo_path) if ppo_path else 0
                })
                
                # Test Adaptive Framework
                if density < self.threshold:
                    selected_planner = 'RRT*'
                    success, plan_time, trajectory = rrt_planner.plan(start, goal)
                else:
                    selected_planner = 'PPO'
                    success, plan_time, trajectory = ppo_planner.plan(start, goal, env)
                
                task_results['Adaptive'].append({
                    'success': success,
                    'time_ms': plan_time,
                    'path_length': len(trajectory) if trajectory else 0,
                    'selected': selected_planner
                })
                
                # Store detailed results
                self.results.append({
                    'scenario': scenario_name,
                    'task': task_name,
                    'density': density,
                    'trial': trial,
                    'start': start,
                    'goal': goal,
                    'rrt_success': rrt_success,
                    'rrt_time': rrt_time,
                    'ppo_success': ppo_success,
                    'ppo_time': ppo_time,
                    'adaptive_success': success,
                    'adaptive_time': plan_time,
                    'adaptive_selected': selected_planner,
                    'threshold': self.threshold
                })
            
            # Calculate and display metrics
            self._display_task_results(task_name, task_results, density)
        
        return scenario_results
    
    def _get_scenario_tasks(self, scenario):
        '''Define navigation tasks for each automotive scenario'''
        if scenario == 'urban_intersection':
            return [
                ((10, 100), (190, 100), 'Cross_Intersection'),
                ((100, 10), (100, 190), 'North_South_Transit'),
                ((30, 30), (170, 170), 'Diagonal_Navigation')
            ]
        elif scenario == 'highway_merge':
            return [
                ((140, 70), (180, 100), 'Merge_Maneuver'),
                ((20, 100), (180, 100), 'Highway_Transit'),
                ((160, 70), (40, 100), 'Lane_Change_Exit')
            ]
        elif scenario == 'parking_lot':
            return [
                ((10, 100), (190, 100), 'Cross_Parking_Lot'),
                ((50, 20), (150, 180), 'Find_Parking_Space'),
                ((30, 50), (170, 150), 'Navigate_Between_Cars')
            ]
    
    def _display_task_results(self, task_name, results, density):
        '''Display comparative results for a task'''
        print(f'    {task_name} (ρ={density:.2f}):')
        
        for method in ['RRT*', 'PPO', 'Adaptive']:
            data = results[method]
            success_rate = np.mean([r['success'] for r in data])
            avg_time = np.mean([r['time_ms'] for r in data])
            
            if method == 'Adaptive':
                selected_counts = {}
                for r in data:
                    if 'selected' in r:
                        selected_counts[r['selected']] = selected_counts.get(r['selected'], 0) + 1
                selection_info = f" ({selected_counts})"
            else:
                selection_info = ""
            
            print(f'      {method:8}: {success_rate:5.1%} success, {avg_time:6.1f}ms{selection_info}')

def run_comprehensive_automotive_evaluation():
    '''Avaliação completa em cenários automotivos'''
    print('🔬 COMPREHENSIVE AUTOMOTIVE EVALUATION')
    print('='*50)
    
    framework = AdaptiveAutomotiveFramework(threshold=0.3)
    
    scenarios = ['urban_intersection', 'highway_merge', 'parking_lot']
    
    for scenario in scenarios:
        framework.evaluate_scenario(scenario, trials=20)
    
    # Save comprehensive results
    results_df = pd.DataFrame(framework.results)
    results_df.to_csv('results/automotive_evaluation_results.csv', index=False)
    
    # Summary analysis
    print(f'\\n📊 SUMMARY ANALYSIS')
    print('='*30)
    
    for scenario in scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario]
        
        overall_adaptive_success = scenario_data['adaptive_success'].mean()
        overall_rrt_success = scenario_data['rrt_success'].mean()
        overall_ppo_success = scenario_data['ppo_success'].mean()
        
        print(f'\\n{scenario.upper()}:')
        print(f'  Adaptive: {overall_adaptive_success:.1%} success')
        print(f'  RRT* only: {overall_rrt_success:.1%} success')
        print(f'  PPO only: {overall_ppo_success:.1%} success')
        
        # Show adaptation behavior
        selections = scenario_data['adaptive_selected'].value_counts()
        print(f'  Selection: {dict(selections)}')

if __name__ == '__main__':
    run_comprehensive_automotive_evaluation()
