# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto

Projeto de Iniciação Científica (PIBIC/FAPEG PI08078-2024, EMC/UFG). O código existe para
sustentar documentos científicos: relatório institucional SIGAA, artigo LAFusion 2026 e slides
de reunião. Toda mudança em código de experimento é, na prática, uma mudança em evidência
publicada — mudar um script que gera CSV citado em documento exige regenerar o dado E revisar
os números nos textos.

Idioma do projeto: **português**. Documentos, comentários e mensagens de commit em PT-BR.

Leia `README.md` (mapa do repositório, glossário, tabela de "onde está o algoritmo X") e
`DEVELOPMENT_LOG.md` (histórico de bugs/decisões, com causas raiz) antes de trabalhos não triviais.

## Tese e o que é/não é evidência atual

A tese é sobre **custo**, não sobre superar em acerto: o ρ-criterion escolhe entre A\* (clássico)
e BC (Behavior Cloning) por densidade local de obstáculos, atingindo desempenho próximo ao melhor
planejador fixo a uma fração do custo de decisão. O A\* real **vence** o critério em acerto puro.
Uma formulação antiga ("supera qualquer método fixo") foi refutada e não deve reaparecer em texto.

Divisão que importa em toda pergunta sobre dados:

- **Fase 1** (`validation_abstract/`): planejadores **mock** — `rrt_star.py` gera linha reta
  interpolada, `ppo_planner.py` sorteia sucesso por probabilidade calibrada. Histórica. Nunca
  citar como evidência de método real.
- **Fase 2** (`eval/env2d/` e `ros2_ws/`): planejadores reais. **Todo número citado em slide ou
  relatório atual vem daqui.**

O SAC existe para cumprir o objetivo 3 do plano de trabalho, mas é **isolado** do critério — perde
para A\*/BC em todo regime testado. BC é supervisionado, não RL; não descrever como RL.

## Comandos

```bash
source .venv/bin/activate          # venv já existe na raiz
pip install -r requirements.txt    # versões pinadas com ==, não relaxar para >=

# Validações do critério (Fase 2, gêmeo 2D) — fonte dos números citados
python eval/env2d/rerun_h1_mixed.py        # 1.500 trials pareados
python eval/env2d/sweep_threshold_real.py  # varredura de rho*
python eval/env2d/rerun_urban.py           # cenário urbano dinâmico, 2.000 trials

# Treino
python eval/env2d/train_2d_bc.py           # BC, 2D, sem Docker
python eval/env2d/train_2d.py              # SAC, 2D
docker compose run --rm train-all          # SAC no Gazebo (tudo num container)

# Benchmark dos clássicos (resposta ao Parecer do Consultor SIGAA)
python validation_abstract/benchmark_classical.py

# Testes (só existem dois, ambos ROS2/viabilidade)
pytest ros2_ws/src/adaptive_planner_ros/test/test_density_estimator.py
```

Figuras: cada `eval/env2d/plot_*.py` gera uma figura específica. Comando por figura em
`paper/figs/CATALOG.md`, que também mapeia figura → documento que a usa.

## Skills do projeto

- `.claude/skills/relatorio-pip/` — regras oficiais do PIP/UFG (formato, limites de página/
  resumo/PDF) e as armadilhas específicas deste repo (`.md` é fonte da verdade, `.tex` é o que
  compila para o PDF do SIGAA). Inclui `scripts/check_formato.py`.
- `.claude/skills/revisao-critica-relatorio/` — revisão crítica linha-por-linha padrão banca de
  prêmio (decisões arbitrárias sem justificativa, afirmação sem fonte, idiossincrasias de texto
  gerado por IA). Inclui `scripts/check_ai_tics.py`, detector determinístico de travessão,
  antítese "X, não Y", bibliografia órfã e hardware não declarado.

Docker: serviço `train-all` roda Gazebo e treino no **mesmo** container de propósito — os serviços
separados `gazebo`+`benchmark` sofrem timeout de descoberta DDS entre containers (bug de infra
conhecido, não de planejamento; ver `DEVELOPMENT_LOG.md` Fase 6).

## Estrutura

- `eval/env2d/` — gêmeo 2D leve (~1000× mais rápido que Gazebo). Onde quase todo dado real nasce.
  `rerun_h1_*.py` são as validações; `RHO_STAR`/`local_rho` vivem em `rerun_h1_real.py` e são
  reusados pelos outros `rerun_h1_*`.
- `validation_abstract/` — Fase 1 histórica (mocks) + `algorithms/classical.py` com Dijkstra/A\*/
  Floyd-Warshall/Johnson otimizados, que **é** evidência válida (responde ao parecer SIGAA).
- `ros2_ws/src/adaptive_planner_ros/` — o mesmo critério como nó ROS2 real (Fase 2 Gazebo).
- `results_abstract/` — CSVs canônicos. Alguns têm milhares de trials custosos de regenerar.
- `paper/` — relatório final, draft LAFusion, `figs/CATALOG.md`.
- `overleaf4/` — LaTeX de slide/relatório/resumo.

## Convenções que evitam retrabalho

**Validar no 2D antes do Gazebo.** Sempre testar reward/política em `eval/env2d/` antes de gastar
horas no Gazebo. Essa regra foi violada uma vez e custou dias: um `R_SURVIVAL` divergente entre 2D
e Gazebo mascarou a causa real de um platô de treino.

**Scripts de treino sobrescrevem artefatos sem aviso.** `train_2d_marl.py` e similares salvam por
cima do modelo existente em `models/`, sem checkpoint. Fazer backup antes de rodar, mesmo "só para
medir tempo". O mesmo vale para os `rerun_*.py`: eles reescrevem CSVs canônicos em
`results_abstract/` — verificar se o CSV alvo já contém uma bateria cara antes de executar.

**Nunca remover a resiliência de treino** (auto-resume por checkpoint, replay buffer salvo,
`restart: unless-stopped`) introduzida no commit `b6f0b5f`.

**Paralelismo satura a CPU.** Rodar 3+ treinos SAC simultâneos em 12 cores multiplicou o tempo
total em vez de dividir. Limitar threads por processo ou rodar sequencial.

**Verificar citações.** Duas de seis referências bibliográficas já se revelaram fabricadas (uma
com DOI válido apontando para outro artigo). Confirmar toda citação contra a fonte antes de
inseri-la em documento.

**Sem travessão (—) em texto** de documentos, slides ou prosa do projeto.

**Comprimir para caber em limite de página pode cortar substância, não só prosa.** Já aconteceu
com o relatório final: uma compressão de 20→12 páginas removeu um estudo de ablação inteiro e
uma seção de histerese junto com prosa redundante. Antes de aceitar uma versão comprimida como
final, `wc -w` do fonte (`.md`) vs. comprimido (`.tex`); se sobrou menos da metade, investigar
explicitamente o que foi removido. Preferir mover para Informações Complementares (não conta
página no PIP/UFG) a deletar.

**Sem `Co-Authored-By: Claude`** ou qualquer atribuição ao Claude em mensagens de commit.

## Armadilhas do repositório

- `models/`, `*.log`, `*.pkl` e `overleaf4/` estão no `.gitignore`. Mudanças em modelos treinados
  e no LaTeX **não aparecem em `git status`** — confirmar estado lendo o arquivo, nunca pelo git.
- O README ainda cita caminhos `overleaf2/`; a pasta atual é `overleaf4/` (foi renomeada duas
  vezes). Corrigir ao tocar nessas seções.
- Existe muito trabalho de dados/figuras historicamente não commitado no working tree. Verificar
  `git status` antes de qualquer operação destrutiva.
- `paper/overleaf_export/` (fora do git) é a pasta pronta para importar no Overleaf: `.tex` +
  só as figuras usadas. **Precisa ser resincronizada manualmente** a cada mudança em
  `paper/relatorio_final_pip.tex` — não há aviso automático de desatualização.
