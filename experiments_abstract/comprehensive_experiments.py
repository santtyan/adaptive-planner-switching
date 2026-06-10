import sys
sys.path.append("../validation_abstract")

from environment import SimpleEnvironment
from adaptive_switcher_fixed import AdaptiveSwitcher
import pandas as pd
import numpy as np
import time
from datetime import datetime

print("=== EXPERIMENTOS PARA PUBLICAÇÃO IEEE ACCESS ===")

# Configuração experimental rigorosa
densities = np.linspace(0.1, 0.8, 8)  # 8 densidades
n_trials = 50  # 50 trials por densidade  
thresholds = [0.25, 0.30, 0.35]  # Teste múltiplos thresholds

results = []

for threshold in thresholds:
    print(f"\n🔬 Threshold = {threshold:.2f}")
    switcher = AdaptiveSwitcher(threshold=threshold)
    
    for density in densities:
        print(f"  📊 Density = {density:.2f}")
        
        env = SimpleEnvironment(obstacle_density=density)
        switcher.set_environment(env)
        
        density_results = []
        
        for trial in range(n_trials):
            # Posições variadas para robustez
            start = (np.random.uniform(5, 95), np.random.uniform(5, 95))
            goal = (np.random.uniform(5, 95), np.random.uniform(5, 95))
            
            success, planning_time, path, selected = switcher.plan(start, goal, env)
            
            results.append({
                'threshold': threshold,
                'target_density': density,
                'actual_density': env.get_density(),
                'trial': trial,
                'success': success,
                'planning_time_ms': planning_time,
                'selected_planner': selected,
                'path_length': len(path) if path else 0,
                'start_x': start[0],
                'start_y': start[1],
                'goal_x': goal[0],
                'goal_y': goal[1]
            })
        
        # Progress feedback
        trial_successes = sum(1 for r in results[-n_trials:] if r['success'])
        print(f"     ✓ {trial_successes}/{n_trials} success ({trial_successes/n_trials:.1%})")

# Salvar dados para análise
df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/comprehensive_experiments_{timestamp}.csv"
df.to_csv(filename, index=False)

print(f"\n✅ EXPERIMENTOS COMPLETOS!")
print(f"📁 Dados salvos: {filename}")
print(f"📊 Total trials: {len(results)}")
print(f"🎯 Pronto para análise IEEE Access")

# Quick analysis
print(f"\n📈 PREVIEW DOS RESULTADOS:")
summary = df.groupby(['threshold', 'selected_planner'])['success'].mean()
print(summary)
