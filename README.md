# Adaptive Context-Based Planner Switching Framework

Framework adaptativo para seleção dinâmica entre RRT* e PPO baseado em densidade de obstáculos. Projeto de Iniciação Científica - UFG.

## Resultados Principais

- **85.3% success rate** (#1 vs 6 métodos SOTA)
- **100% switching accuracy**
- **Regret bounds ≤2.2%** vs oracle
- **1500+ experimentos** validados

## Problema

Métodos clássicos (RRT*, A*) ou modernos (PPO, SAC) são usados isoladamente. Trabalhos recentes (He et al. 2025, Sensors 2025) usam switching heurístico ou pesos fixos. Nossa solução: **switching adaptativo baseado em densidade de obstáculos** com threshold otimizado.

## Método

**Política de seleção:**
- RRT* quando densidade < 0.30 (ambientes abertos)
- PPO quando densidade ≥ 0.30 (ambientes densos)

**Componentes:**
- SimpleEnvironment (grid 100×100)
- RRTStarPlanner (implementação própria)
- PPOPlanner (Stable-Baselines3)
- AdaptiveSwitcher (threshold 0.30)

## Resultados vs SOTA

| Método | Success Rate |
|--------|--------------|
| **Adaptive (ours)** | **85.3%** |
| Neural Switching | 78.7% |
| Fixed PPO | 76.0% |
| Hybrid DRL | 66.0% |
| He Multi-opt | 54.0% |
| Fixed RRT* | 48.0% |

**Performance por densidade:**
- Baixa (ρ<0.30): RRT* 88-92%, PPO 73-76% → seleciona RRT*
- Alta (ρ≥0.30): RRT* 45-62%, PPO 71-78% → seleciona PPO

**Garantias teóricas:**
- Average regret: 2.2% vs oracle
- Optimality gap: 1.7%
- Performance: ≥93.3% do oracle

## Instalação

\\\ash
git clone https://github.com/santtyan/adaptive-planner-switching
cd adaptive-planner-switching
python -m venv venv_ic
.\venv_ic\Scripts\activate  # Windows
pip install -r requirements.txt
\\\

## Uso Rápido

\\\python
from src.environment import SimpleEnvironment
from src.adaptive_switcher import AdaptiveSwitcher

env = SimpleEnvironment(obstacle_density=0.35)
switcher = AdaptiveSwitcher(threshold=0.30)
switcher.set_environment(env)

success, time_ms, trajectory, selected = switcher.plan((10,10), (90,90), env)
print(f"{selected}: {success} em {time_ms:.2f}ms")
\\\

## Experimentos

\\\ash
python experiments/comprehensive_experiments.py  # 1500 trials
python experiments/sota_comparison.py           # 6 métodos
python experiments/theoretical_analysis.py      # regret bounds
python experiments/realistic_scenario_validation.py  # cenários automotivos
\\\

## Estrutura

\\\
src/                # Código core
├── environment.py
├── adaptive_switcher.py
└── planners/      # RRT* + PPO
experiments/       # 10 scripts validação
results/          # 8 CSVs + figuras
docs/             # Relatórios
\\\

## Publicações Planejadas

1. **IEEE Access (A4)** - Framework + experimentos (Jan 2026)
2. **Applied Sciences (B1)** - Multi-objetivo (Mar 2026)
3. **Sensors (A4)** - Teoria + SOTA (Abr 2026)

## Limitações

- Contexto unidimensional (densidade)
- Threshold fixo offline
- Ambiente 2D

## Próximos Passos

- Contexto multi-dimensional
- Threshold adaptativo online
- ROS 2/Gazebo

## Contribuição

Primeira abordagem sistemática para switching adaptivo com garantias formais entre planners clássicos e modernos.

## Citação

\\\ibtex
@misc{silva2025adaptive,
  title={Adaptive Context-Based Planner Switching for Autonomous Navigation},
  author={Silva, Yan and Aldo},
  year={2025},
  institution={Universidade Federal de Goiás}
}
\\\

---

**Estudante:** Yan Silva | **Orientador:** Prof. Aldo | **UFG** - Nov 2025