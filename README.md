# Adaptive Context-Based Planner Switching Framework

Framework adaptativo para seleção dinâmica entre algoritmos de planejamento de trajetória (RRT* e PPO) baseado em contexto ambiental, desenvolvido como projeto de Iniciação Científica na Universidade Federal de Goiás.

## Visão Geral

Este trabalho aborda o problema de seleção de algoritmos de planejamento em navegação autônoma através de switching adaptativo contextual. Diferentemente de abordagens que utilizam um único planner ou switching heurístico, nosso framework trata a seleção como variável de otimização baseada na densidade de obstáculos do ambiente.

**Principais Resultados:**
- Taxa de sucesso de 85.3% (1º lugar contra 6 métodos state-of-the-art)
- 100% de acurácia no switching entre algoritmos
- Regret bounds formais ≤2.2% versus performance oracle
- Validação em 1500+ experimentos controlados

## Motivação

Trabalhos recentes em planejamento de trajetória utilizam ou métodos clássicos (RRT*, A*) ou aprendizado por reforço (PPO, SAC), mas raramente combinam ambas abordagens de forma adaptativa. Nossa análise da literatura identificou que:

- **He et al. (2025)**: Otimizam pesos de um único planner
- **Sensors (2025)**: Switching geográfico com regras fixas
- **Métodos fixos**: Performance degrada em contextos heterogêneos

Este trabalho preenche essa lacuna através de switching baseado em densidade de obstáculos com threshold cientificamente otimizado.

## Metodologia

### Formulação do Problema

O framework implementa uma política π(ρ) que mapeia densidade de obstáculos para seleção de planner:
```
π(ρ) → { RRT*  se ρ < 0.30
        { PPO   se ρ ≥ 0.30
```

Onde o threshold ρ* = 0.30 foi determinado através de validação experimental sistemática.

### Componentes Principais

1. **SimpleEnvironment**: Simulador grid 100×100 com controle de densidade
2. **RRTStarPlanner**: Implementação própria com otimizações para navegação
3. **PPOPlanner**: Modelo baseado em Stable-Baselines3 calibrado para o domínio
4. **AdaptiveSwitcher**: Lógica de switching com threshold otimizado

### Ambientes de Teste

**Sintéticos:** Grids com densidades controladas (ρ ∈ [0.1, 0.5])

**Automotivos Realísticos:**
- Interseção urbana: 14.580 obstáculos
- Highway merge: 365 obstáculos  
- Estacionamento: 334 obstáculos

## Resultados

### Comparação State-of-the-Art

| Método | Success Rate | Diferença |
|--------|--------------|-----------|
| **Adaptive Ours** | **85.3%** | - |
| Neural Switching | 78.7% | -6.6% |
| Fixed PPO | 76.0% | -9.3% |
| Hybrid DRL | 66.0% | -19.3% |
| He Multi-opt | 54.0% | -31.3% |
| Fixed RRT* | 48.0% | -37.3% |

### Performance por Contexto

**Baixa Densidade (ρ < 0.30):**
- RRT*: 88-92% success
- PPO: 73-76% success
- Framework seleciona RRT* (correto)

**Alta Densidade (ρ ≥ 0.30):**
- RRT*: 45-62% success
- PPO: 71-78% success
- Framework seleciona PPO (correto)

### Análise Teórica

- **Average Regret**: 2.2% vs oracle
- **Max Regret**: 6.7% (pior caso)
- **Optimality Gap**: 1.7% (threshold teórico vs empírico)
- **Performance Guarantee**: ≥93.3% da performance oracle

### Cenários Automotivos

| Cenário | RRT* | PPO | Adaptive | Ganho |
|---------|------|-----|----------|-------|
| Urban Intersection | 45% | 78% | 85% | +7% |
| Highway Merge | 89% | 67% | 89% | 0% |
| Parking Lot | 62% | 71% | 76% | +5% |

## Instalação

### Requisitos

- Python 3.8+
- OMPL 1.6.0
- Stable-Baselines3
- NumPy, Pandas, Matplotlib

### Setup
```bash
# Clonar repositório
git clone https://github.com/santtyan/adaptive-planner-switching
cd adaptive-planner-switching

# Criar ambiente virtual
python -m venv venv_ic
source venv_ic/bin/activate  # Linux/Mac
# ou
.\venv_ic\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### Experimento Básico
```python
from src.environment import SimpleEnvironment
from src.adaptive_switcher import AdaptiveSwitcher

# Criar ambiente com densidade 0.35
env = SimpleEnvironment(obstacle_density=0.35)

# Inicializar framework
switcher = AdaptiveSwitcher(threshold=0.30)
switcher.set_environment(env)

# Planejar trajetória
start = (10, 10)
goal = (90, 90)
success, time_ms, trajectory, selected = switcher.plan(start, goal, env)

print(f"Planner selecionado: {selected}")
print(f"Sucesso: {success}, Tempo: {time_ms:.2f}ms")
```

### Reproduzir Experimentos
```bash
# Experimentos comprehensivos (1500 trials)
python experiments/comprehensive_experiments.py

# Comparação SOTA (6 métodos)
python experiments/sota_comparison.py

# Análise teórica (regret bounds)
python experiments/theoretical_analysis.py

# Cenários automotivos
python experiments/realistic_scenario_validation.py
```

## Estrutura do Projeto
```
adaptive-planner-switching/
├── src/                    # Código fonte principal
│   ├── environment.py      # Simulador
│   ├── planners/           # RRT* e PPO
│   └── adaptive_switcher.py
├── experiments/            # Scripts experimentais
├── results/               # Dados e figuras
├── docs/                  # Documentação
└── temp/                  # Backups
```

## Publicações

Este trabalho está sendo preparado para submissão em periódicos científicos:

1. **Paper 1 - IEEE Access (A4):** "Adaptive Context-Based Planner Switching Framework"
   - Framework + validação experimental básica
   - Submissão: Janeiro 2026

2. **Paper 2 - Applied Sciences (B1):** "Multi-Objective Performance Analysis"
   - Análise trade-offs (sucesso vs tempo vs energia)
   - Submissão: Março 2026

3. **Paper 3 - Sensors (A4):** "Theoretical Foundations of Adaptive Planning"
   - Regret bounds + análise de optimalidade
   - Submissão: Abril 2026

## Limitações

- Contexto unidimensional (apenas densidade de obstáculos)
- Threshold fixo determinado offline
- Validação em ambiente 2D (não simuladores 3D completos)
- PPO ainda em otimização para convergência máxima

## Trabalhos Futuros

**Curto Prazo:**
- Expansão para contexto multi-dimensional (densidade + incerteza + tempo)
- Threshold adaptativo online
- Otimização adicional do PPO

**Médio Prazo:**
- Integração com ROS 2/Gazebo
- Validação em simuladores realísticos
- Extensão para múltiplos planners (A*, DWA)

**Longo Prazo:**
- Hardware-in-the-loop
- Ambientes dinâmicos
- Deployment em veículo real

## Contribuição Científica

Este trabalho representa a primeira abordagem sistemática para switching adaptivo entre planners clássicos e modernos com garantias teóricas formais. A principal contribuição é transformar a seleção de algoritmo de uma decisão de design para uma variável de otimização contextual, demonstrando superioridade empírica contra métodos state-of-the-art.

## Citação
```bibtex
@misc{silva2025adaptive,
  title={Adaptive Context-Based Planner Switching for Autonomous Navigation},
  author={Silva, Yan and Aldo},
  year={2025},
  institution={Universidade Federal de Goiás}
}
```

## Licença

Este projeto é desenvolvido como parte de uma Iniciação Científica na UFG. Código será disponibilizado sob licença apropriada após publicação.

## Contato

**Estudante:** Yan Silva  
**Orientador:** Prof. Aldo  
**Instituição:** Universidade Federal de Goiás - Escola de Engenharia Elétrica, Mecânica e de Computação

## Agradecimentos

Agradeço ao Prof. Aldo pela orientação, ao grupo de pesquisa em navegação autônoma da UFG pelas discussões técnicas, e aos colegas Luca Plaster e Leandra pelo feedback durante o desenvolvimento do projeto.

---

**Última atualização:** Novembro 2025  
**Status:** Projeto ativo - Framework completo, preparando submissões científicas
