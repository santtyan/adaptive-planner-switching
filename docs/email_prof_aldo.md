# Email — Prof. Aldo (copiar e enviar)

**Para:** aldo.diaz@ufg.br
**Assunto:** IC PIBIC — Atualização do plano e reunião de alinhamento

---

Prof. Aldo,

Espero que esteja bem. Gostaria de marcar uma reunião esta semana ou na próxima para apresentar o estado atual do trabalho e alinhar as próximas etapas.

Desde a última reunião avancei bastante na estrutura do projeto. O repositório foi reorganizado em um workspace ROS2 único (`adaptive-planner-switching`) integrando os três frentes do trabalho. Os componentes já implementados são:

- `density_estimator_node` — estima ρ local a partir do costmap do Nav2 (8 testes pytest passando)
- `obs_utils.py` — módulo canônico de observação 27-dim (lidar + goal polar) compartilhado entre treino e inferência
- `adaptive_switcher_node` — FSM com histerese (ρ=0.30±0.05, dwell 1.5s) que alterna entre Nav2 e RL via twist_mux
- `rl_controller_node` — carrega modelo SB3 e publica /cmd_vel_rl
- `gazebo_gym_env.py` — wrapper Gymnasium sobre Gazebo Classic para treino do RL
- Scripts de treino PPO e SAC (SB3 2.3.0)
- `nav2_params.yaml` com SmacPlanner2D (A*) para TB3 diff-drive
- `demo.launch.py` — sobe todo o pipeline com um comando

Tenho algumas decisões que precisam da sua orientação antes de prosseguir:

1. Aceita PPO + SAC como os dois algoritmos RL (DDPG cortado — SAC já supera DDPG na literatura)?
2. N=30 trials por condição é suficiente para o benchmark?
3. A validação fica em simulação Gazebo (sem TB3 físico)?
4. O prazo de entrega seria 30/06 — isso é viável na visão do senhor?
5. Qual a data e formato da defesa/banca PIBIC?
6. Temos deadline do CONPEEX 2026? Vale submeter um resumo?

Posso enviar o repositório completo antes da reunião para o senhor revisar.

Qualquer horário disponível nesta semana serve.

Atenciosamente,
Yan Santos Leite
Matrícula: 202302594 — EMC/UFG
