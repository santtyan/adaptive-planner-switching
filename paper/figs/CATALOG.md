# Catálogo de Figuras — `paper/figs/`

Reescrito 17/07/2026 (faxina de imagens). Pasta reduzida de 108 imagens únicas
para as **16 essenciais** — cada uma com uso real confirmado em
`relatorio_final_pip.md`/`.tex` (sincronizados) ou `README.md`. As demais
(~92, incluindo todo o histórico de depuração de treino/reward, screenshots
antigos de mundos não usados, painéis de defesa/tese não referenciados, e
duplicatas) foram removidas do repositório — recuperáveis via `git log`/`git
show` se precisar no futuro.

Exceção: `gazebo_screenshots/` foi mantida por decisão explícita do Yan
mesmo sem uso atual em nenhum documento — são fotos reais do TurtleBot3 no
Gazebo, não regeráveis por script (única evidência visual de que a simulação
rodou de verdade).

## Estrutura de pastas

| Subpasta | Conteúdo |
|---|---|
| `core/` | ρ-criterion, switching heatmap, teste estatístico, progressão de densidade |
| `benchmark/` | Tempo dos algoritmos clássicos |
| `cbs/` | Escalabilidade do CBS multiagente |
| `marl/` | MARL, degradação por densidade, roadmap |
| `2d/` | Gêmeo 2D: BC, comparação de trajetórias, heatmap, degradação, learning curve, H1 real |
| `pareto/` | Justificativa formal do limiar τ*=0,30 |
| `gazebo_screenshots/` | Capturas reais do Gazebo (mantidas por valor de evidência, sem uso em documento hoje) |

## As 16 figuras essenciais

| Figura | Onde é usada | Notas |
|---|---|---|
| `benchmark/benchmark_time.png` | Relatório final 3.1; README | Tempo dos 4 algoritmos clássicos |
| `pareto/fig_pareto_boxplot_main.png` | Relatório final 3.2 | Justificativa formal τ*=0,30 |
| `core/fig_statistical_test.png` | Relatório final 3.2 | Sucesso por densidade, IC 95% |
| `core/switching_heatmap.png` | Relatório final 3.3; README | Decisão espacial A*/SAC |
| `core/density_progression.png` | README | Progressão sparse→dense |
| `2d/fig_2d_learning_curve_ci.png` | Relatório final 3.3/3.4 | Convergência SAC 2D, 3 seeds |
| `2d/fig_2d_compare_dense.png` | Relatório final 3.3/3.4 | Trajetórias A*/SAC/Adaptativo |
| `2d/fig_2d_heatmap_dense.png` | Relatório final 3.3/3.4 | Mapa de decisão ρ-criterion |
| `2d/fig_2d_degradation_singlerobot.png` | Relatório final 3.3/3.4 | Degradação por densidade fora da distribuição |
| `2d/fig_2d_bc_trajectory_sparse.png` | Relatório final 3.3 | Trajetória do agente BC (98% sucesso) |
| `2d/fig_2d_bc_episode_sparse.gif` | Relatório final 3.3 (citado no texto) | Versão animada do BC |
| `all_benchmarks_comparison.png` | Relatório final 3.3 | Painel consolidado 4 paradigmas |
| `2d/fig_2d_h1_real_validation.png` | Relatório final 3.3/3.5 | Sucesso e custo reais, A* vs BC |
| `2d/fig_2d_h1_boxplots.png` | Relatório final 3.3/3.5 | Distribuição/outliers de custo |
| `marl/fig_marl_shared_reward_comparison.png` | Relatório final 3.3/3.5 | RL independente vs MARL centralizado |
| `cbs/cbs_scalability.png` | Relatório final 3.4/3.6 | Tempo CBS vs N agentes |
| `marl/fig_marl_degradation_by_density_2d.png` | Relatório final 3.6 | Motivação MARL (degradação RL independente) |
| `marl/roadmap_marl_tj.png` | Relatório final 4/Conclusão; README | Roadmap Fase 1/2/3 |

## O que foi removido (17/07/2026)

- **`training/` inteira** (~25 arquivos): diagnóstico de *suicidal agent*, colapso de entropia, race condition de reset, gargalo de LIDAR, iterações de reward, currículo — histórico de depuração de processo, não resultado científico. Preservado em `DEVELOPMENT_LOG.md` (texto) e recuperável via git se precisar regenerar alguma figura.
- **`thesis/`** (6 arquivos): painéis de defesa/sensibilidade nunca referenciados em nenhum documento formal.
- **`obs/`** (3 arquivos): diagrama de observação 29-dim e arquitetura do sistema — nunca entraram no relatório final.
- **Variantes não usadas de `2d/`**: mundos `sparse`/`very_dense` sem uso (só `dense` está no relatório), GIFs de episódio, versão antiga de `fig_2d_learning_curve` (órfã na raiz de `figs/`, superada por `fig_2d_learning_curve_ci` com IC 95%).
- **Variantes não usadas de `benchmark/`**: `benchmark_memory`, `outcome_matrix`, `success_by_density` e outras — eram placeholders/mock nunca promovidos a figura final.
- **Variantes não usadas de `cbs/`**: `cbs_canyon_annotation`, `cbs_density_sweep`, `cbs_deviation_analysis`, `cbs_spatial_annotation`, `cbs_tpg_comparison`, `cbs_ppo_ratio_*` — citados em texto no relatório mas sem figura própria inserida; podem ser reincorporados no futuro se decidir ilustrar esses parágrafos (ver git history).
- **Variantes não usadas de `core/`**: `fig_adaptive_switching`, `fig_bootstrap_ci_global`, `fig_composite_heatmap`, `fig_planner_usage_distribution`, `fig_regret_analysis`, `fig_summary_panel`, `trajectory_comparison`, `duration_boxplot`, `density_heatmap`.
- **`core/rho_sensitivity.png`**: já estava marcado como ⛔ não publicar (métrica errada, pico em ρ=0,20).
- **Resto de `pareto/`**: `fig_pareto_threshold`, `fig_pareto_detail`, `fig_pareto_boxplot_time`, `fig_pareto_equations_panel` — redundantes com `fig_pareto_boxplot_main`, que já está no relatório.
- **Resto de `marl/`**: variantes de motivação (`fig_marl_motivation_*`), snapshot multiagente, GIFs de episódio — superados pelas 2 figuras MARL já no relatório.
- **`misc/conpeex_slide.png`**: slide do CONPEEX (evento já passou, resumo já submetido — ver `project_documents_status.md`).

## Se precisar recuperar algo

Os arquivos removidos continuam no histórico do git (`git log --diff-filter=D -- paper/figs/`).
Para restaurar um arquivo específico: `git checkout <commit-antes-da-remoção> -- paper/figs/<caminho>`.
