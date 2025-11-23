"""
Enhanced Multi-Objective Experimental Validation
Generates new data with path quality metrics
"""

import sys
sys.path.append("src")
import numpy as np
import pandas as pd
from multiobjective_analysis import PathQualityAnalyzer
from adaptive_switcher_fixed import AdaptiveSwitcher
from environment import SimpleEnvironment

def run_enhanced_validation():
    """Execute enhanced experiments with quality metrics"""
    print("=== ENHANCED MULTI-OBJECTIVE VALIDATION ===")
    
    analyzer = PathQualityAnalyzer()
    enhanced_results = []
    
    # Test configurations
    densities = [0.15, 0.25, 0.35, 0.45, 0.55]
    n_trials = 25  # Reasonable sample size
    
    for density in densities:
        print(f"📊 Enhanced analysis density {density:.2f}")
        
        env = SimpleEnvironment(obstacle_density=density)
        switcher = AdaptiveSwitcher()
        switcher.set_environment(env)
        
        for trial in range(n_trials):
            # Generate diverse start/goal pairs
            start = (np.random.uniform(5, 95), np.random.uniform(5, 95))
            goal = (np.random.uniform(5, 95), np.random.uniform(5, 95))
            
            # Ensure reasonable distance
            distance = np.linalg.norm(np.array(goal) - np.array(start))
            if distance < 20:  # Too close, skip
                continue
                
            success, time_ms, path, selected = switcher.plan(start, goal, env)
            
            if success and path and len(path) > 1:
                # Analyze path quality
                quality_metrics = analyzer.analyze_path_quality(path, env)
                
                enhanced_results.append({
                    'density': density,
                    'actual_density': env.get_density(),
                    'success': success,
                    'selected_planner': selected,
                    'planning_time_ms': time_ms,
                    'path_length': len(path),
                    'euclidean_distance': distance,
                    'smoothness': quality_metrics['smoothness'],
                    'energy_consumption': quality_metrics['energy_consumption'],
                    'safety_clearance': quality_metrics['safety_clearance'],
                    'length_optimality': quality_metrics['length_optimality'],
                    'composite_score': quality_metrics['composite_score'],
                    'start_x': start[0], 'start_y': start[1],
                    'goal_x': goal[0], 'goal_y': goal[1]
                })
            else:
                # Record failures too
                enhanced_results.append({
                    'density': density,
                    'actual_density': env.get_density(),
                    'success': success,
                    'selected_planner': selected,
                    'planning_time_ms': time_ms,
                    'path_length': 0,
                    'euclidean_distance': distance,
                    'smoothness': 0,
                    'energy_consumption': float('inf'),
                    'safety_clearance': 0,
                    'length_optimality': 0,
                    'composite_score': 0,
                    'start_x': start[0], 'start_y': start[1],
                    'goal_x': goal[0], 'goal_y': goal[1]
                })
    
    # Convert to DataFrame and analyze
    enhanced_df = pd.DataFrame(enhanced_results)
    
    print(f"\n📊 ENHANCED RESULTS SUMMARY:")
    print(f"Total trials: {len(enhanced_df)}")
    print(f"Successful trials: {enhanced_df['success'].sum()}")
    print(f"Overall success rate: {enhanced_df['success'].mean():.1%}")
    
    # Analysis by planner
    print(f"\n📈 QUALITY METRICS BY PLANNER:")
    quality_summary = enhanced_df[enhanced_df['success'] == True].groupby('selected_planner')[
        ['smoothness', 'energy_consumption', 'safety_clearance', 'length_optimality', 'composite_score']
    ].agg(['mean', 'std']).round(3)
    
    print(quality_summary)
    
    # Analysis by density
    print(f"\n📊 PERFORMANCE BY DENSITY:")
    density_summary = enhanced_df.groupby('density')[
        ['success', 'smoothness', 'composite_score']
    ].agg(['mean', 'count']).round(3)
    
    print(density_summary)
    
    # Save results
    enhanced_df.to_csv('results/enhanced_multiobjective_results.csv', index=False)
    print(f"\n💾 Enhanced results saved: results/enhanced_multiobjective_results.csv")
    
    # Key insights for papers
    rrt_data = enhanced_df[(enhanced_df['success']) & (enhanced_df['selected_planner'] == 'rrt_star')]
    ppo_data = enhanced_df[(enhanced_df['success']) & (enhanced_df['selected_planner'] == 'ppo')]
    
    insights = {
        'rrt_avg_smoothness': rrt_data['smoothness'].mean(),
        'ppo_avg_smoothness': ppo_data['smoothness'].mean(),
        'rrt_avg_energy': rrt_data['energy_consumption'].mean(),
        'ppo_avg_energy': ppo_data['energy_consumption'].mean(),
        'rrt_avg_safety': rrt_data['safety_clearance'].mean(),
        'ppo_avg_safety': ppo_data['safety_clearance'].mean(),
        'adaptive_composite_score': enhanced_df[enhanced_df['success']]['composite_score'].mean()
    }
    
    print(f"\n🎯 KEY INSIGHTS FOR PAPERS:")
    for key, value in insights.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return enhanced_df, insights

if __name__ == "__main__":
    results, insights = run_enhanced_validation()
    print(f"\n✅ ENHANCED VALIDATION COMPLETE!")
