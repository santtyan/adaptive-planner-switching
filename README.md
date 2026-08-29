# Adaptive Planner Switching — IC PIBIC/FAPEG UFG

Framework adaptativo para seleção dinâmica de planejador de trajetória com base na densidade local de obstáculos (**ρ-criterion**). Projeto de Iniciação Científica — EMC/UFG, bolsa FAPEG PI08078-2024.

**Estudante:** Yan Santos Leite | **Orientador:** Prof. Dr. Aldo André Diaz Salazar (INF/UFG)
**Período:** Set/2025 – Ago/2026 | **Stack:** ROS2 Humble · Gazebo Classic · TurtleBot3 Waffle · Stable-Baselines3

> **Como usar este README:** ele é o mapa do repositório, não o documento científico completo.
> Se você (ou o orientador) está procurando "onde está o script que gera X", comece pela
> [Tabela de algoritmos e onde encontrá-los](#algoritmos-implementados-e-onde-encontrá-los) ou pelo
> [mapa de pastas](#mapa-do-repositório). O relatório final (`paper/relatorio_final_pip.md`) é a
> fonte da verdade sobre método e resultados; este arquivo só organiza a navegação.

---

## Tese central (versão atual, corrigida)

> Um critério de densidade local de obstáculos (ρ) decide, em tempo real, entre um planejador
> clássico (A\*) e uma política aprendida (Behavior Cloning), alcançando o desempenho do melhor
> planejador fixo **a uma fração do seu custo computacional**.

**Isso substitui uma formulação anterior** ("o critério supera qualquer método fixo em taxa de
acerto"), que não se sustentou quando os planejadores mock foram trocados por implementações
reais (ver [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md), Fase 6). Com dados reais, o A\* vence o
critério adaptativo em acerto puro (88,2% vs. 84,3%, n=1.500), mas o BC mantém custo de decisão
~656× menor que o A\* em alta densidade (0,045 ms vs. 29,21 ms, medido no mesmo ambiente). A tese
hoje é sobre **custo**, não sobre superar em acerto — todo texto atual reflete essa correção.
⚠️ **Nota (29/08/2026):** um número de "custo do próprio ρ-criterion" (8,60 ms/episódio) que
circulava em slides antigos **não tem instrumentação real no código** — nenhum script mede esse
tempo de fato (ver nota em `eval/env2d/plot_pareto_success_vs_cost.py`); não usar esse valor até
a Etapa de instrumentação ser concluída.

```
π(ρ₀) = { A*  (clássico, Nav2 SmacPlanner2D / grid A*)   se ρ₀ < 0,30
        { BC  (aprendido, behavior cloning supervisionado) se ρ₀ ≥ 0,30
```

onde `ρ₀` é a densidade local de obstáculos (fração de raios de LIDAR abaixo de 1,0 m), medida
**uma vez, no início do episódio** — não recalculada a cada passo.

### O critério, em código

**Script de referência:** [`eval/env2d/rerun_h1_real.py`](eval/env2d/rerun_h1_real.py) — mesma
implementação usada em todas as validações reais deste trabalho (`rerun_h1_mixed.py` e
`rerun_h1_hysteresis.py` reusam a mesma `local_rho`).

```python
RHO_STAR = 0.30

def local_rho(env: Env2D) -> float:
    ranges = _scan(env._x, env._y, env._yaw, env.obstacles, env.arena)
    return float(np.mean(ranges < 1.0))

# no reset do episódio:
rho0 = local_rho(env)
use_astar = rho0 < RHO_STAR
```

Linha a linha:

- **`RHO_STAR = 0,30`**: o limiar único do critério, calibrado por varredura empírica (ver
  `results_abstract/threshold_sweep_real.csv`), não reajustado durante a execução.
- **`local_rho(env)`**: lê os raios do LIDAR e calcula a fração deles que retornou menos de
  1,0 m. Quanto maior essa fração, mais obstáculos perto do robô *agora* — a mesma métrica do
  heatmap de comutação (*switching*).
- **Quando roda:** só uma vez, no início do episódio (`reset`), não a cada passo — o que
  explica o custo baixo do critério (8,60 ms/episódio).
- **A decisão em si:** `rho0 < RHO_STAR` decide entre dois planejadores já prontos (A\*, BC);
  não treina nada novo, só escolhe qual dos dois usar naquele episódio.

Em uma frase: mede quão apertado está o ambiente antes de andar, e usa essa medida — não a
distância ao alvo nem a velocidade — para decidir se vale pagar o custo do A\* ou se o BC já
resolve.

---

## Conceitos-chave (glossário rápido)

| Conceito | O que é | Por que importa aqui |
|---|---|---|
| **ρ (rho), densidade local** | Fração das leituras de LIDAR do robô que retornaram menos de 1,0 m. Mede "quão apertado" está o ambiente ao redor do robô agora. | É a única variável que decide qual planejador usar — não distância ao alvo, não velocidade. |
| **ρ\* (rho-star), limiar** | Valor de corte fixo, `0,30`, calibrado por varredura empírica (n=1.500). | Abaixo dele usa A\*, acima usa BC. Não é reajustado durante a execução (é *offline*). |
| **A\*** | Busca em grafo clássica, informada por heurística. Sempre acha o caminho ótimo no grid, mas expande mais nós (custo cresce) quanto mais obstáculos existem. | Planejador clássico do critério; representa o lado "preciso e caro". |
| **BC (Behavior Cloning)** | Aprendizado **supervisionado** por imitação de um especialista (aqui, um campo potencial). **Não é RL** — importante: a tese central não usa nenhum RL, apesar do objetivo 3 do plano de trabalho pedir RL profundo (esse objetivo é cumprido separadamente pelo SAC, isolado do critério). | Representa o lado "barato e quase tão preciso" do trade-off. |
| **SAC (Soft Actor-Critic)** | Algoritmo de RL profundo (Stable-Baselines3). Cumpre o objetivo 3 do plano de trabalho oficial, mas **não compõe o critério adaptativo** — SAC perde para A\*/BC em todo regime de densidade testado. | Existe no repositório como evidência de RL testado, não como parte da solução final. |
| ***Regret*** | Perda de desempenho do critério adaptativo em relação ao melhor planejador fixo possível naquele episódio. | Métrica usada para calibrar ρ\* (varredura em `threshold_sweep_real.csv`). |
| **MARL (Multi-Agent RL)** | Tentativa de estender o critério para múltiplos robôs coordenados. | Resultado preliminar e não reproduzido (ver tabela de algoritmos); causa raiz é arquitetural (falta credit assignment por agente — precisaria de MAPPO real). |
| **Fase 1 vs. Fase 2** | Fase 1 = validação com planejadores **mock** (estatisticamente calibrados, não reais). Fase 2 = validação com planejadores **reais** (A\*/BC/SAC de verdade), incluindo cenário urbano dinâmico. | Fase 1 é histórica/exploratória; **todo número citado em slide ou relatório atual vem da Fase 2 com planejadores reais**, nunca dos mocks. |

---

## Algoritmos implementados e onde encontrá-los

Esta é a tabela para responder "onde está o script de X" na hora, sem precisar procurar.

| Algoritmo / componente | Arquivo principal | Tipo | Usado na tese final? |
|---|---|---|---|
| **A\* (grid, 2D)** | [`eval/env2d/astar_planner.py`](eval/env2d/astar_planner.py) | Clássico, busca em grafo com `heapq` | ✅ Sim — lado clássico do critério |
| **A\* / Dijkstra / Floyd-Warshall / Johnson (benchmark)** | [`validation_abstract/algorithms/classical.py`](validation_abstract/algorithms/classical.py), [`validation_abstract/benchmark_classical.py`](validation_abstract/benchmark_classical.py) | Clássicos, implementações otimizadas (`heapq`, matriz densa, Bellman-Ford) | ✅ Sim — responde ao Parecer do Consultor SIGAA (implementações otimizadas, não didáticas) |
| **Behavior Cloning (BC)** | [`eval/env2d/train_2d_bc.py`](eval/env2d/train_2d_bc.py) (treino) | Supervisionado, imitação de especialista (campo potencial) | ✅ Sim — lado "moderno" do critério |
| **ρ-criterion (o switcher em si)** | [`eval/env2d/rerun_h1_real.py`](eval/env2d/rerun_h1_real.py) (`RHO_STAR`, `local_rho`) — mesma lógica reusada em `rerun_h1_mixed.py` e `rerun_h1_hysteresis.py` | Regra de decisão, offline por episódio | ✅ Sim — é a contribuição central |
| **ρ-criterion (nó ROS2 real)** | [`ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/adaptive_switcher_node.py`](ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/adaptive_switcher_node.py), [`density_estimator.py`](ros2_ws/src/adaptive_planner_ros/adaptive_planner_ros/density_estimator.py) | Nó ROS2, mesmo critério em produção | ✅ Sim — Fase 2 (Gazebo) |
| **SAC (Soft Actor-Critic)** | [`eval/env2d/train_2d.py`](eval/env2d/train_2d.py) | RL profundo (Stable-Baselines3) | ⚠️ Cumpre objetivo 3 do plano, mas **isolado** — não entra no critério final (perde para A\*/BC) |
| **CrossQ** | [`eval/env2d/train_2d_crossq.py`](eval/env2d/train_2d_crossq.py) | RL profundo, alternativa ao SAC | ❌ Testado e descartado — não convergiu melhor que SAC |
| **MARL (recompensa compartilhada)** | [`eval/env2d/train_2d_marl.py`](eval/env2d/train_2d_marl.py) | RL multiagente, política única sobre observação concatenada | ⚠️ Resultado preliminar não reproduzido — ver limitações |
| **RRT\* (mock)** | [`validation_abstract/planners/rrt_star.py`](validation_abstract/planners/rrt_star.py) | **Mock estatístico** (gera linha reta interpolada, não busca real) | ❌ Não — só Fase 1 histórica, não usar como evidência de método real |
| **PPO (mock)** | [`validation_abstract/planners/ppo_planner.py`](validation_abstract/planners/ppo_planner.py) | **Mock estatístico** (sorteia sucesso por probabilidade calibrada) | ❌ Não — idem acima |
| **CBS (Conflict-Based Search)** | dados reais em [`results_abstract/cbs_scalability_20trials.csv`](results_abstract/cbs_scalability_20trials.csv) (via `atb033/multi_agent_path_planning`, externo) | Coordenação multiagente, clássico | ⚠️ Só benchmark de escalabilidade (Relatório Parcial); substituído por A\*/SAC na narrativa atual |
| **DDPG** | — não implementado | RL profundo | ❌ Previsto no Relatório Parcial, nunca testado nem descartado com nota formal — pendência aberta |

---

## Mapa do repositório

```
.
├── README.md                    # este arquivo — mapa de navegação
├── DEVELOPMENT_LOG.md           # histórico cronológico do processo de pesquisa (bugs, decisões, viradas)
├── docker-compose.yml           # serviços: train-all, gazebo, benchmark, train-sac, train-ppo, smoketest
│
├── eval/env2d/                  # ★ gêmeo 2D leve — onde quase todo dado real da tese atual é gerado
│   ├── astar_planner.py         # A* real (grid)
│   ├── env_2d.py                # ambiente Gymnasium 2D (single-agent)
│   ├── env_2d_multi.py          # ambiente 2D multi-agente
│   ├── train_2d*.py             # treino: SAC, BC, CrossQ, MARL
│   ├── rerun_h1_*.py            # validações do critério ρ (H1) — fonte dos números citados em slide/relatório
│   ├── sweep_threshold_real.py  # varredura de ρ* (gera threshold_sweep_real.csv)
│   ├── rerun_urban.py           # cenário urbano dinâmico (2.000 trials, 4 condições)
│   └── plot_*.py / visualize_*.py  # geram as figuras de paper/figs/
│
├── validation_abstract/         # Fase 1 histórica — validação Monte Carlo com planejadores calibrados/mock
│   ├── algorithms/classical.py  # Dijkstra/A*/Floyd-Warshall/Johnson otimizados (resposta ao parecer SIGAA)
│   ├── benchmark_classical.py   # benchmark de tempo/memória dos clássicos
│   └── planners/                # ⚠️ rrt_star.py e ppo_planner.py são MOCKS, não reais
│
├── ros2_ws/src/
│   ├── adaptive_planner_ros/    # nó ROS2 do critério ρ (Fase 2, Gazebo)
│   └── turtlebot3_gym_env/      # ambiente Gymnasium sobre Gazebo
│
├── results_abstract/            # CSVs de dados — ver tabela abaixo para os mais citados
├── paper/
│   ├── figs/CATALOG.md          # catálogo de TODAS as figuras científicas geradas
│   ├── relatorio_final_pip.md   # relatório final institucional (SIGAA) — fonte da verdade
│   ├── relatorio_final_pip.tex  # mesmo conteúdo, formato PDF
│   └── lafusion_2026_draft.md   # rascunho do artigo para submissão LAFusion 2026
│
├── overleaf4/apresentacao/      # slide de reunião com o orientador (.tex, Beamer/metropolis)
├── docs/
│   ├── relatorio_parcial.md     # relatório parcial já aprovado (01/04/2026) — narrativa anterior (RRT*/PPO)
│   └── PLANO_CORRECAO.md        # auditoria que motivou a correção da tese (de "supera" para "empata a custo menor")
│
└── models/                      # modelos treinados (.zip) — SAC, BC
```

### CSVs de dados mais citados

| Arquivo | Conteúdo | Onde é usado |
|---|---|---|
| [`results_abstract/h1_real_2d_mixed_pool.csv`](results_abstract/h1_real_2d_mixed_pool.csv) | 1.500 trials pareados, mundo sorteado por trial (A\*/BC/critério) | Números 88,2%/84,3% do slide e relatório |
| [`results_abstract/h1_real_2d_validation.csv`](results_abstract/h1_real_2d_validation.csv) | 100 trials/mundo × método × regime de densidade (sparse/dense/very_dense) | Comparação A\*/BC/SAC por regime |
| [`results_abstract/threshold_sweep_real.csv`](results_abstract/threshold_sweep_real.csv) | Varredura de ρ\* de 0,10 a 0,60 | Justificativa de por que ρ\*=0,30 foi mantido |
| [`results_abstract/urban_grid_results.csv`](results_abstract/urban_grid_results.csv) | Cenário urbano dinâmico, 2.000 execuções, 4 condições | Fecha itens "ambientes urbanos"/"obstáculos dinâmicos" do plano |
| [`results_abstract/cbs_scalability_20trials.csv`](results_abstract/cbs_scalability_20trials.csv) | Escalabilidade CBS multiagente | Relatório Parcial (narrativa anterior) |

---

## Documentos institucionais e sua relação

| Documento | O que é | Onde está |
|---|---|---|
| **Plano de Trabalho oficial** | Contrato original aprovado pela FAPEG (PI08078-2024): objetivos, metodologia, cronograma | SIGAA/UFG (fora do repositório) — checklist de cobertura no relatório final, Seção 2 |
| **Parecer do Consultor SIGAA** (13/06/2025) | Exige comparação com implementações **otimizadas** dos clássicos, não didáticas | Respondido em `paper/relatorio_final_pip.md`, Seção 2.1 |
| **Relatório Parcial** | Já aprovado (01/04/2026). Narrativa anterior: RRT\*/PPO, validado com CBS real | `docs/relatorio_parcial.md` |
| **Relatório Final** | Documento formal para o SIGAA, prazo 31/08/2026. Narrativa atual: A\*/BC/SAC, tese reformulada para custo | `paper/relatorio_final_pip.md` / `.tex` |
| **Slide de reunião com o orientador** | Deck de trabalho (não defesa), usado para prestar contas e levantar decisões pendentes | `overleaf4/apresentacao/apresentacao_prof_aldo.tex` (⚠️ fora do git, ver nota abaixo) |
| **Artigo LAFusion 2026** | Submissão para revisão por pares (Springer CCIS), prazo 16/08/2026 | `paper/lafusion_2026_draft.md` |

> **Nota:** `overleaf4/` está no `.gitignore` — mudanças no slide **nunca aparecem em `git status`**.
> Se o orientador perguntar sobre uma versão do slide, confirme a data/conteúdo diretamente no
> arquivo, não pelo histórico do git.

---

## Reprodução

### Fase 2 real (gêmeo 2D — a maioria dos números da tese atual vem daqui)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validação do critério ρ com planejadores reais (gera os números 88,2%/84,3%)
python eval/env2d/rerun_h1_mixed.py

# Varredura de ρ* (gera threshold_sweep_real.csv)
python eval/env2d/sweep_threshold_real.py

# Cenário urbano dinâmico (2.000 trials)
python eval/env2d/rerun_urban.py
```

### Treino (SAC / BC / Gazebo)

```bash
# Treino SAC no Gazebo (requer Docker)
docker compose run --rm train-all

# Treino BC (gêmeo 2D, sem Docker)
python eval/env2d/train_2d_bc.py
```

### Benchmark clássico (resposta ao Parecer do Consultor)

```bash
python validation_abstract/benchmark_classical.py
```

### Figuras científicas

Ver comandos completos por figura em [`paper/figs/CATALOG.md`](paper/figs/CATALOG.md).

---

## Limitações declaradas

1. **Fase 1** (`validation_abstract/`) usa planejadores **mock** (RRT\*/PPO estatisticamente calibrados), não reais — histórica, não usar como evidência atual.
2. **Fase 2 Gazebo**: robô navega corretamente quando testado dentro do próprio container do simulador, mas a bateria estatística completa (30 execuções) está bloqueada por um bug de infraestrutura (timeout de descoberta de serviço DDS entre containers Docker separados `benchmark`↔`gazebo`) — não é bug de planejamento.
3. **MARL**: resultado preliminar (recompensa compartilhada zera colisão entre robôs no primeiro treino) não se reproduziu em retreinos idênticos. Causa raiz identificada como arquitetural (política única sem *credit assignment* por agente); correção real exigiria reescrever o loop de treino para MAPPO.
4. **DDPG**: previsto no Relatório Parcial, substituído por SAC, nunca testado isoladamente nem descartado com nota formal — pendência aberta para o relatório final.
5. **Single-agente**: o critério ρ decide bem qual planejador usar, mas não resolve coordenação entre múltiplos robôs (ver MARL acima).
6. **Contexto unidimensional**: a decisão usa só densidade local — não incorpora curvatura do ambiente, velocidade de obstáculos, etc.

---

## Publicações / Apresentações

- **CONPEEX 2026** — Seminário de Iniciação à Pesquisa PIP
- **LAFusion 2026** — submissão em revisão por pares, prazo 16/08/2026 (Springer CCIS)
- **Relatório Final SIGAA** — prazo 31/08/2026

---

## Citação

```bibtex
@misc{santos2026adaptive,
  title={Seleção Adaptativa de Planejador de Trajetória baseada em Densidade Local de Obstáculos},
  author={Santos Leite, Yan and Diaz Salazar, Aldo André},
  year={2026},
  institution={Universidade Federal de Goiás},
  note={IC PIBIC/FAPEG PI08078-2024}
}
```
