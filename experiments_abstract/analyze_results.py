import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carregar dados dos experimentos
df = pd.read_csv("results/comprehensive_experiments_20251123_030145.csv")

print("=== ANÁLISE ESTATÍSTICA IEEE ACCESS ===")

# 1. Optimal Threshold Analysis
print("\n📊 THRESHOLD OPTIMIZATION:")
for threshold in [0.25, 0.30, 0.35]:
    subset = df[df['threshold'] == threshold]
    overall_success = subset['success'].mean()
    print(f"Threshold {threshold:.2f}: {overall_success:.1%} overall success")

# 2. Performance by Density & Planner
print("\n📈 PERFORMANCE BY PLANNER:")
performance = df.groupby(['threshold', 'selected_planner'])['success'].agg(['mean', 'std', 'count'])
print(performance)

# 3. Statistical Significance Tests
rrt_data = df[df['selected_planner'] == 'rrt_star']['success']
ppo_data = df[df['selected_planner'] == 'ppo']['success'] 

t_stat, p_value = stats.ttest_ind(rrt_data, ppo_data)
print(f"\n📊 SIGNIFICANCE TEST:")
print(f"RRT* vs PPO t-test: t={t_stat:.3f}, p={p_value:.6f}")
print(f"Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")

# 4. Create Publication Figure
plt.figure(figsize=(12, 8))

# Performance by density plot
for threshold in [0.25, 0.30, 0.35]:
    subset = df[df['threshold'] == threshold]
    density_performance = subset.groupby('actual_density')['success'].mean()
    
    plt.subplot(2, 2, int((threshold - 0.25) * 20) + 1)
    plt.plot(density_performance.index, density_performance.values, 'o-', 
             label=f'Adaptive (τ={threshold:.2f})', linewidth=2)
    
    # Add RRT* and PPO individual performance
    rrt_perf = subset[subset['selected_planner'] == 'rrt_star'].groupby('actual_density')['success'].mean()
    ppo_perf = subset[subset['selected_planner'] == 'ppo'].groupby('actual_density')['success'].mean()
    
    if len(rrt_perf) > 0:
        plt.plot(rrt_perf.index, rrt_perf.values, 's--', alpha=0.7, label='RRT* only')
    if len(ppo_perf) > 0:
        plt.plot(ppo_perf.index, ppo_perf.values, '^--', alpha=0.7, label='PPO only')
    
    plt.axvline(x=threshold, color='red', linestyle=':', alpha=0.7, label='Threshold')
    plt.xlabel('Obstacle Density')
    plt.ylabel('Success Rate')
    plt.title(f'Threshold = {threshold:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/ieee_access_figure.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Figure saved: results/ieee_access_figure.png")

# 5. Paper-Ready Statistics
print(f"\n📋 PAPER STATISTICS:")
print(f"Total experiments: {len(df):,}")
print(f"Success rate range: {df['success'].min():.1%} - {df['success'].max():.1%}")
print(f"Planning time range: {df['planning_time_ms'].min():.1f} - {df['planning_time_ms'].max():.1f}ms")
print(f"RRT* avg time: {df[df['selected_planner']=='rrt_star']['planning_time_ms'].mean():.1f}ms")
print(f"PPO avg time: {df[df['selected_planner']=='ppo']['planning_time_ms'].mean():.1f}ms")

print(f"\n✅ READY FOR IEEE ACCESS SUBMISSION!")
