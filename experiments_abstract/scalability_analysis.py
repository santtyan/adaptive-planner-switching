"""
Scalability Analysis - Different Environment Sizes
Critical for demonstrating framework robustness
"""

import sys
sys.path.append("../validation_abstract")
import numpy as np
import pandas as pd
import time
from adaptive_switcher_fixed import AdaptiveSwitcher
from environment import SimpleEnvironment

def scalability_analysis():
    """Test framework performance across different environment sizes"""
    print("=== SCALABILITY ANALYSIS ===")
    
    # Different environment sizes
    grid_sizes = [50, 100, 150, 200, 300]  # 50x50 to 300x300
    density = 0.3  # Fixed density for comparison
    n_trials = 15  # Reasonable for larger environments
    
    scalability_results = []
    
    for grid_size in grid_sizes:
        print(f"📏 Testing grid size {grid_size}x{grid_size}")
        
        for trial in range(n_trials):
            # Create environment with specific size
            env = SimpleEnvironment(grid_size=grid_size, obstacle_density=density)
            switcher = AdaptiveSwitcher()
            switcher.set_environment(env)
            
            # Scale start/goal to environment size
            margin = grid_size * 0.1  # 10% margin
            start = (np.random.uniform(margin, grid_size-margin), 
                    np.random.uniform(margin, grid_size-margin))
            goal = (np.random.uniform(margin, grid_size-margin),
                   np.random.uniform(margin, grid_size-margin))
            
            # Measure planning time more precisely
            start_time = time.perf_counter()
            success, time_ms, path, selected = switcher.plan(start, goal, env)
            actual_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Distance scaling
            distance = np.linalg.norm(np.array(goal) - np.array(start))
            normalized_distance = distance / grid_size  # Normalize by environment size
            
            scalability_results.append({
                'grid_size': grid_size,
                'environment_complexity': grid_size * grid_size,
                'trial': trial,
                'success': success,
                'selected_planner': selected,
                'reported_time_ms': time_ms,
                'actual_time_ms': actual_time,
                'path_length': len(path) if path else 0,
                'distance': distance,
                'normalized_distance': normalized_distance,
                'actual_density': env.get_density(),
                'planning_efficiency': (1000.0 / actual_time) if actual_time > 0 else 0  # Plans per second
            })
    
    # Analysis
    scale_df = pd.DataFrame(scalability_results)
    
    print(f"\n📊 SCALABILITY RESULTS:")
    
    # Success rate by size
    success_by_size = scale_df.groupby('grid_size')['success'].agg(['mean', 'count'])
    print(f"\nSuccess rate by environment size:")
    print(success_by_size)
    
    # Planning time analysis
    time_analysis = scale_df.groupby('grid_size')['actual_time_ms'].agg(['mean', 'std', 'median'])
    print(f"\nPlanning time analysis (ms):")
    print(time_analysis.round(2))
    
    # Complexity scaling
    complexity_analysis = scale_df.groupby('grid_size').agg({
        'environment_complexity': 'first',
        'actual_time_ms': 'mean',
        'success': 'mean',
        'planning_efficiency': 'mean'
    }).round(3)
    
    print(f"\nComplexity vs Performance:")
    print(complexity_analysis)
    
    # Scalability insights
    max_size = scale_df['grid_size'].max()
    min_size = scale_df['grid_size'].min()
    
    max_time = scale_df[scale_df['grid_size'] == max_size]['actual_time_ms'].mean()
    min_time = scale_df[scale_df['grid_size'] == min_size]['actual_time_ms'].mean()
    
    scaling_factor = max_time / min_time if min_time > 0 else float('inf')
    complexity_growth = (max_size / min_size) ** 2  # Area growth
    
    print(f"\n🎯 SCALABILITY INSIGHTS:")
    print(f"Environment complexity growth: {complexity_growth:.1f}x")
    print(f"Planning time growth: {scaling_factor:.1f}x")
    print(f"Scalability efficiency: {complexity_growth/scaling_factor:.2f}")
    
    # Save results
    scale_df.to_csv('results/scalability_analysis_results.csv', index=False)
    
    return scale_df, {
        'scaling_factor': scaling_factor,
        'complexity_growth': complexity_growth,
        'efficiency_ratio': complexity_growth/scaling_factor if scaling_factor > 0 else 0,
        'max_environment_tested': max_size,
        'maintains_performance': scale_df[scale_df['grid_size'] == max_size]['success'].mean() > 0.7
    }

if __name__ == "__main__":
    results, insights = scalability_analysis()
    print(f"\n✅ SCALABILITY ANALYSIS COMPLETE!")
    
    for key, value in insights.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
