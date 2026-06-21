# Catálogo de Figuras — `paper/figs/`

Índice organizado **por utilidade**. Cada figura tem `.png` (150 dpi) + `.pdf`.
Todas geradas pelos scripts em `eval/` (saída fixa em `paper/figs/`).

Legenda de uso:
- 📄 **Relatório final** (`relatorio_final_pip.tex`/`.md`)
- 📝 **Resumo CONPEEX** (`resumo_conpeex_2026.tex`/`.md`)
- 🖼️ **Slide CONPEEX** (`generate_conpeex_slide.py`)
- 💤 **Não referenciada** (disponível para artigo/IEEE Access; ainda não inserida)
- ⛔ **Não publicar** (motivo na coluna Notas)

---

## 1. Núcleo da tese — critério ρ e seleção adaptiva

| Figura | Uso | Notas |
|---|---|---|
| `planner_time_vs_density.png` | 📄 | Motivação ρ*=0,30 (Seção 2) — tempo vs densidade |
| `switching_heatmap.png` | 📄 📝 | Heatmap espacial do switcher (modelo analítico) |
| `density_progression.png` | 📄 📝 | Progressão sparse→dense, decisão A*/SAC |
| `success_by_density.png` | 📄 | Taxa de sucesso por faixa ρ (Fase 2, placeholder) |
| `outcome_matrix.png` | 📄 | Matriz goal/colisão/timeout (Fase 2, placeholder) |
| `trajectory_comparison.png` | 📄 (placeholder) | Trajetórias A*/SAC/Adaptativo — gerar pós-convergência |
| `rho_sensitivity.png` | ⛔ | **NÃO PUBLICAR**: métrica de sucesso bruto tem pico em 0,20, não 0,30 — enfraquece H2. Refazer com regret vs oracle |
| `fig_adaptive_switching.png` | 💤 | Diagrama conceitual do switching |

## 2. Benchmark de algoritmos clássicos

| Figura | Uso | Notas |
|---|---|---|
| `benchmark_time.png` | 📄 | Tempo de execução (log-log) |
| `benchmark_memory.png` | 📄 | Memória de pico |

## 3. Multi-agente / CBS / motivação MARL (Fase 3)

| Figura | Uso | Notas |
|---|---|---|
| `cbs_scalability.png` | 📄 | Tempo CBS vs N agentes (motiva decisão local) |
| `cbs_tpg_comparison.png` | 📄 | CBS discreto → schedule cinemático (TPG) |
| `cbs_deviation_analysis.png` | 📄 | Experimento de desvio (Nash candidato) |
| `cbs_density_sweep.png` | 📄 | Transição de fase ρ≈0,28–0,32 (2/3/5 agentes) |
| `cbs_spatial_annotation.png` | 📄 | 3 agentes, 19% SAC zona densa |
| `cbs_canyon_annotation.png` | 📄 | 5 agentes canyon, 44% SAC |
| `cbs_ppo_ratio_comparison.png` | 📄 | 100 cenários reais (parcial) |
| `roadmap_marl_tj.png` | 📄 | Roadmap Fase 1/2/3 (MARL + teoria dos jogos) |
| `fig_marl_motivation_summary.png` | 📄 | **FIGURA CHAVE**: A* vs SAC vs CBS — nenhum resolve tudo → MARL necessário (dados sintéticos → substituir por Gazebo) |
| `fig_marl_motivation_degradation.png` | 📄 | Curva de degradação goal_rate + inter_collision vs N agentes |
| `cbs_ppo_ratio_2agents.png` | 💤 | Variante de 2 agentes (superada por `_comparison`) |

## 4. Arquitetura SAC e espaço de observação

| Figura | Uso | Notas |
|---|---|---|
| `fig_system_architecture.png` | 💤 | Diagrama de blocos do sistema completo |
| `fig_obs_29dim.png` | 💤 | Espaço de observação 29-dim (canônica) |
| `fig_obs_space_29dim.png` | 💤 | Variante do diagrama 29-dim (duplicata — preferir `fig_obs_29dim`) |
| `fig_curriculum_schedule.png` | 💤 | Currículo de distância 1→3 m |
| `fig_lidar_downsampling.png` | 💤 | Subamostragem LIDAR 360→24 raios |

## 5. Diagnóstico do treino — *suicidal agent* e reward (20/06/2026)

| Figura | Uso | Notas |
|---|---|---|
| `fig_suicidal_agent_diagnosis.png` | 💤 | Âncora: integral de penalidade vs colidir cedo |
| `fig_reward_iterations.png` | 💤 | Jornada das 6 versões de reward até convergir |
| `fig_ep_len_comparison.png` | 💤 | ep_len por versão de reward |
| `fig_reward_math_proof.png` | 💤 | Prova matemática por que cada reward falhou |
| `fig_sparse_world_signal.png` | 💤 | Breakthrough sparse.world (ep_rew=+19,9) |
| `fig_world_curriculum.png` | 💤 | Currículo por densidade de mundo (sparse→dense) |
| `fig_ent_coef_collapse.png` | 💤 | Colapso de entropia (auto+gSDE) vs fixo 0,1 |
| `fig_ep_len_suicidal_signature.png` | 💤 | Assinatura: ep_len cai ao ativar SAC (dados reais) |
| `fig_obstacle_reward_field.png` | 💤 | Campo de penalidade de obstáculo antes/depois |
| `fig_obstacle_reward_directional.png` | 💤 | Penalidade direcional de obstáculo |
| `fig_reward_rebalance.png` | 💤 | Rebalanceamento de magnitudes antes/depois |
| `fig_goldstandard_reward_comparison.png` | 💤 | Nosso reward vs Cimurs/de Jesus/HMP-DRL |
| `fig_reward_components.png` | 💤 | Componentes do reward shaping |
| `fig_sac_reward_shaping.png` | 💤 | Reward shaping SAC (visão geral) |

## 6. Otimização de treino / throughput / ROI

| Figura | Uso | Notas |
|---|---|---|
| `fig_cpu_bottleneck.png` | 💤 | Gargalo CPU-bound |
| `fig_scan_throughput.png` | 💤 | Throughput de entrega de scan (LIDAR 5 Hz) |
| `fig_optimization_roi.png` | 💤 | Quick wins hierarquizadas por ROI |
| `fig_sample_efficiency_ladder.png` | 💤 | Escada de sample-efficiency (SAC→DroQ/CrossQ) |
| `fig_planb_trigger.png` | 💤 | Gatilho do PlanB |
| `fig_reset_race_condition.png` | 💤 | Bug de race condition no reset |
| `fig_spawn_safety.png` | 💤 | Validação de spawn sem colisão |

## 7. Curvas de aprendizado

| Figura | Uso | Notas |
|---|---|---|
| `fig_sac_learning_curve_live.png` | 💤 | Curva ao vivo (`plot_live_training_curve.py`) |
| `sac_learning_curve.png` | 💤 | Curva de aprendizado (snapshot) |

## 8. Mundos e densidade

| Figura | Uso | Notas |
|---|---|---|
| `fig_world_density_comparison.png` | 💤 | Comparação de densidade entre mundos |
| `density_heatmap.png` | 💤 | Heatmap de densidade |
| `duration_boxplot.png` | 💤 | Boxplot de duração de episódio |

## 9. Apresentação

| Figura | Uso | Notas |
|---|---|---|
| `conpeex_slide.png` | 🖼️ | Slide único CONPEEX (regenerar após fotos Gazebo) |

---

## Pendências de figuras

- ⛔ `rho_sensitivity.png` — refazer com regret vs oracle antes de qualquer uso.
- 📸 `gz_sparse.png`, `gz_dense.png`, `gz_very_dense.png` — **fotos do TurtleBot3 no Gazebo**,
  referenciadas como placeholder no relatório e no resumo. Capturar manualmente após
  convergência do SAC (Julho/2026). Ao adicionar, descomentar `\includegraphics` e
  regenerar o slide via `python3 paper/generate_conpeex_slide.py`.
- 🧹 Duplicata: `fig_obs_space_29dim.png` ≈ `fig_obs_29dim.png` — usar a segunda.
