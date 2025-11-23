# analyze_workshop.py - Análise para workshop paper
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_results():
    '''Load most recent experiment results'''
    import glob
    files = glob.glob('results/workshop_experiments_*.json')
    if not files:
        raise FileNotFoundError('No experiment files found!')
    
    latest_file = max(files)
    print(f'Loading: {latest_file}')
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data

def create_workshop_plots(data):
    '''Generate 3 essential plots for workshop paper'''
    
    results = data['results']
    densities = [0.15, 0.25, 0.35, 0.45]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Success Rate vs Density
    methods = ['RRT*', 'PPO', 'Adaptive']
    success_rates = {method: [] for method in methods}
    
    for density in densities:
        for method in methods:
            sr = np.mean([trial['success'] for trial in results[str(density)][method]])
            success_rates[method].append(sr * 100)
    
    ax1 = axes[0]
    for method in methods:
        marker = 'o' if method == 'RRT*' else ('s' if method == 'PPO' else '^')
        ax1.plot(densities, success_rates[method], 
                marker=marker, label=method, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Obstacle Density (ρ)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('(a) Success Rate vs Density', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([60, 100])
    
    # Plot 2: Planning Time vs Density
    avg_times = {method: [] for method in methods}
    
    for density in densities:
        for method in methods:
            successful_trials = [t for t in results[str(density)][method] if t['success']]
            if successful_trials:
                avg_time = np.mean([t['time_ms'] for t in successful_trials])
                avg_times[method].append(avg_time)
            else:
                avg_times[method].append(0)
    
    ax2 = axes[1]
    for method in methods:
        marker = 'o' if method == 'RRT*' else ('s' if method == 'PPO' else '^')
        ax2.plot(densities, avg_times[method], 
                marker=marker, label=method, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Obstacle Density (ρ)', fontsize=12)
    ax2.set_ylabel('Planning Time (ms)', fontsize=12)
    ax2.set_title('(b) Planning Time vs Density', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Switching Behavior
    rrt_selections = []
    ppo_selections = []
    
    for density in densities:
        adaptive_trials = results[str(density)]['Adaptive']
        rrt_count = sum(1 for t in adaptive_trials if t.get('selected') == 'RRT*')
        ppo_count = len(adaptive_trials) - rrt_count
        rrt_selections.append(rrt_count)
        ppo_selections.append(ppo_count)
    
    ax3 = axes[2]
    x = np.arange(len(densities))
    width = 0.35
    
    ax3.bar(x, rrt_selections, width, label='RRT* selected', color='#1f77b4')
    ax3.bar(x, ppo_selections, width, bottom=rrt_selections,
            label='PPO selected', color='#ff7f0e')
    
    ax3.set_xlabel('Obstacle Density (ρ)', fontsize=12)
    ax3.set_ylabel('Number of Selections', fontsize=12)
    ax3.set_title('(c) Adaptive Switching Behavior', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{d:.2f}' for d in densities])
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/workshop_figures.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/workshop_figures.pdf', bbox_inches='tight')
    
    print('✅ Figures saved: results/workshop_figures.png/.pdf')
    
    return fig

def generate_workshop_table(data):
    '''Generate results table for workshop paper'''
    
    results = data['results']
    densities = [0.15, 0.25, 0.35, 0.45]
    
    print('\n' + '='*80)
    print('TABLE I: EXPERIMENTAL RESULTS FOR WORKSHOP PAPER')
    print('='*80)
    print('Density | RRT* SR  | PPO SR   | Adaptive SR | RRT* Sel | PPO Sel')
    print('--------|----------|----------|-------------|----------|--------')
    
    for density in densities:
        rrt_sr = np.mean([t['success'] for t in results[str(density)]['RRT*']])
        ppo_sr = np.mean([t['success'] for t in results[str(density)]['PPO']])
        ada_sr = np.mean([t['success'] for t in results[str(density)]['Adaptive']])
        
        rrt_sel = sum(1 for t in results[str(density)]['Adaptive'] 
                     if t.get('selected') == 'RRT*')
        ppo_sel = 30 - rrt_sel
        
        print(f'{density:7.2f} | {rrt_sr:7.1%} | {ppo_sr:7.1%} | '
              f'{ada_sr:10.1%} | {rrt_sel:7d} | {ppo_sel:6d}')
    
    print('='*80)

if __name__ == '__main__':
    data = load_latest_results()
    create_workshop_plots(data)
    generate_workshop_table(data)
    
    print('\n✅ ANÁLISE WORKSHOP CONCLUÍDA!')
    print('✅ Dados prontos para Paper 1/3')
    print('✅ Próximo: Draft do paper workshop')
