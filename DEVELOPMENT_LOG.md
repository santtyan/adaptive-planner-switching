# Log de Desenvolvimento — adaptive-planner-switching

Registro cronológico do processo de pesquisa desta IC (PIBIC/FAPEG PI08078-2024): decisões tomadas, bugs encontrados e corrigidos, e o porquê de cada mudança de rumo. Este documento complementa o relatório final (`paper/relatorio_final_pip.md`), que descreve o método final; aqui está o processo que levou até ele.

## Por que este documento existe

O relatório final tem limite de páginas (15, referencial, regra oficial do PIP/UFG) e descreve metodologia final, não a jornada de depuração. Mas o processo de chegar até os resultados finais é, em si, evidência de rigor científico e de engenharia: tentativas que não deram certo, hipóteses refutadas por dados reais, bugs de infraestrutura raiz encontrados sob pressão de prazo. Registrar isso separadamente evita inchar o relatório formal enquanto mantém o processo auditável.

---

## Fase 1 — Fundação (2025)

**Plano de trabalho aprovado.** Objetivo geral: comparar métodos clássicos e modernos de planejamento de trajetória, com foco em adaptabilidade.

**13/06/2025 — Parecer do Consultor SIGAA.** Recomendação: comparar clássicos usando implementações otimizadas, não didáticas. Resposta aplicada desde então: `heapq` para Dijkstra/A*, matriz densa para Floyd-Warshall, Bellman-Ford para Johnson.

**01/04/2026 — Relatório Parcial aprovado.** Narrativa: RRT* (clássico) vs. PPO (RL), validado com dados reais de CBS (`atb033/multi_agent_path_planning`) sobre benchmarks públicos. Essa narrativa evoluiu depois para A*/SmacPlanner2D vs. SAC, com nota de transição explícita no relatório final (o critério ρ e o limiar 0,30 permanecem inalterados nessa troca, pois medem uma propriedade do ambiente, não do planejador).

## Fase 2 — Descoberta dos mocks (14/05/2026)

Auditoria do código revelou que os planejadores em `validation_abstract/planners/` (`rrt_star.py`, `ppo_planner.py`) são **mocks estatísticos**, não implementações reais: `MockRRTStarPlanner` gera linha reta interpolada, `PPOPlannerMockScientific` sorteia sucesso com probabilidade calibrada para reproduzir taxas publicadas na literatura. Decisão tomada então: declarar isso explicitamente no relatório como "validação Monte Carlo do critério" em vez de "validação com planejadores reais" — decisão que se mostraria crucial dois meses depois (ver Fase 6).

## Fase 3 — Reward shaping no Gazebo (junho de 2026)

Treino SAC no Gazebo apresentava o padrão clássico de *suicidal agent*: `ep_len_mean` caindo para ~8 passos, agente colidindo cedo de propósito. Causa raiz: penalidade de obstáculo por passo acumulava mais magnitude negativa ao longo do episódio do que a penalidade terminal de colisão, tornando racional colidir cedo. Correção seguindo Cimurs et al. (2022): eliminar o piso de penalidade por passo, usar recompensa de sobrevivência constante + progresso clipado ≥ 0 + terminais grandes (±100).

Currículo de distância adicionado (começa com goals próximos, expande com taxa de sucesso) para dar sinal de recompensa terminal alcançável desde o início do treino.

## Fase 4 — Achado multi-agente e extensão teórica (19-21/06/2026)

Formalização do problema multi-agente como Dec-POMDP (cada robô observa ρᵢ local, decide independentemente). Experimento de desvio (dados reais de CBS): agente que desvia da política ρ-criterion para sempre-A* paga +2,1%, para sempre-SAC paga +14,1% — evidência de equilíbrio de Nash candidato, não uma prova formal.

Um achado paralelo desse período (21/06, `eval_multi_2d.py`, comparando "A*" independente com SAC independente em N=4 robôs) mostrou 100%/0% de sucesso/colisão para "A*" contra ~38%/100% para SAC. **Este achado tinha um problema não descoberto até 09/07/2026: o "A*" ali é, na verdade, uma política analítica de linha reta até o goal, não busca A* real** (ver Fase 6). Nunca chegou a ser citado no relatório final ou no LAFusion — mas ficou registrado internamente como um achado forte, o que quase levou a uma citação incorreta meses depois.

## Fase 5 — A regra "2D antes do Gazebo" e o platô do SAC (julho de 2026)

Treino SAC no Gazebo travado em `ep_rew_mean ≈ -100` por 250 mil passos ou mais, mesmo após corrigir o *suicidal agent*. Regra do projeto (violada e depois relembrada): sempre validar reward/política no ambiente 2D leve (Env2D, ~1000× mais rápido que o Gazebo) antes de gastar horas no Gazebo.

Ao aplicar a regra: `env_2d.py` tinha `R_SURVIVAL=0,1` hardcoded, divergente do `R_SURVIVAL=0,0` real do Gazebo — a validação 2D nunca tinha sido feita com a configuração exata. Corrigido, parametrizado via variável de ambiente. Com a config correta, o SAC convergia normalmente no 2D — isolando o platô do Gazebo como um problema de infraestrutura, não de reward.

Resiliência de treino adicionada (commit `b6f0b5f`): auto-resume por checkpoint, replay buffer salvo, `restart: unless-stopped` no Docker — nunca mais perder um treino a um crash de container.

## Fase 6 — A madrugada dos bugs de infraestrutura (08-09/07/2026)

A investigação mais longa e mais produtiva da IC. Em ordem de descoberta:

1. **Exploração congelada**: `ent_coef` fixo em vez de `"auto"` impedia o SAC de re-explorar após um platô. Corrigido com lógica de troca pós-`load()` (não é possível simplesmente passar `ent_coef="auto"` no `SAC.load()` de um checkpoint com `ent_coef` fixo — o `set_parameters(exact_match=True)` rejeita a mudança de estrutura; a correção troca o modo manualmente após o load normal).

2. **Física assíncrona**: padrão-ouro de RL+Gazebo exige pausar a física a cada passo de controle, não só no reset — sem isso, ação e observação podem dessincronizar. Implementado; regressão inicial de fps (12 → <1) por causa de chamadas bloqueantes de serviço a cada passo, corrigida trocando para chamadas não-bloqueantes.

3. **`PlanBCallback` nunca disparava**: bug de padrão repetido (mesmo tipo de erro já visto e corrigido em outro callback antes) — o callback lia uma métrica do logger que só é populada em intervalos específicos, sempre caindo no valor padrão `+inf`. O treino passou de 300k, 500k, até 650.886 passos com `ep_rew_mean=-99` sustentado sem nunca abortar. Corrigido para ler `model.ep_info_buffer` diretamente.

4. **Nomes de serviço ROS2/Gazebo errados**: `/gazebo/set_entity_state`, `/gazebo/pause_physics`, `/gazebo/unpause_physics` **nunca existiram** nesta configuração de Gazebo Classic 11 — os nomes reais publicados não têm o prefixo `/gazebo/`. Consequência: `teleport_robot()` falhava silenciosamente em toda chamada, a sessão inteira (e provavelmente todo o histórico do projeto) — o robô nunca foi reposicionado entre episódios. Corrigido: `pause_physics`/`unpause_physics` com os nomes certos; `teleport_robot()` reimplementado como `delete_entity`+`spawn_entity` (os únicos serviços que de fato existem).

5. **Odometria nunca resetada**: consequência direta do bug 4 — sem teleporte funcional, `/odom` (dead-reckoning) acumulava erro monstruoso ao longo de milhares de episódios. Corrigido junto com o bug 4 (respawnar a entidade também reseta o integrador interno do plugin `diff_drive`).

6. **`docker compose restart` não aplica mudanças de `command:`** — descoberto ao investigar por que um fix já commitado (`--lockstep`) parecia não ter efeito. Só `docker compose up -d` ou `--force-recreate` aplicam mudanças de comando; isso significa que vários fixes anteriores da noite podem nunca ter sido de fato testados como se pensava.

7. **Robô não ganha velocidade real dentro de `env.step()`**: mesmo com todos os bugs acima corrigidos, o robô permanecia praticamente parado durante episódios reais. Diagnosticado por comparação: um comando manual sustentado (`ros2 topic pub`, sem pausar física) movia o robô normalmente; o ciclo `step()` real da pipeline (pausa → publica → despausa → espera 1 scan → pausa de novo), não. Hipótese mais provável: `max_wheel_acceleration` do modelo TurtleBot3 não tem tempo de gerar velocidade real dentro da janela minúscula de física despausada por passo. **Não resolvido.**

8. **`gzclient` (GUI) não renderiza robôs spawnados dinamicamente**: bug cosmético, sem relação com os anteriores. Tentativas: GPU passthrough (`/dev/dri`), correção de `GAZEBO_RESOURCE_PATH`, caminhos absolutos de mesh, respawn com cliente já conectado. Nenhuma resolveu. Não afeta os resultados (robô funciona, só não aparece na janela). **Não resolvido, mas isolado como irrelevante.**

## Fase 7 — Decisão de descartar o Gazebo (09/07/2026, ~15h)

Diante do prazo da IC e do bug 7 não resolvido, decisão consciente: encerrar a Fase 2 (Gazebo) no nível de infraestrutura, sem validação quantitativa de navegação real. O orientador (Prof. Dr. Aldo André Díaz Salazar), consultado via mensagem, sugeriu buscar apoio técnico (grupo Pequi Mecânico) antes de descartar — decisão final sobre retomar ou não fica para reunião de acompanhamento.

Relatório final e artigo LAFusion reescritos para tratar essa lacuna como limitação central declarada, não como resultado "pendente" ou placeholder vazio.

## Fase 8 — Revalidação de H1 com planejadores reais (09/07/2026, ~17-19h)

Ao tentar reduzir a dependência de mock sem o Gazebo, nova descoberta: o "A*" usado nas comparações multi-agente do ambiente 2D (`eval_multi_2d.py`, ver Fase 4) era uma política de linha reta, não busca real.

Implementado A* real (`eval/env2d/astar_planner.py`): busca em grade 8-conectada, heap binário, heurística octile. Primeiro teste revelou colisões inesperadas (40% em `very_dense`) — causa: o controlador de perseguição de caminho cortava cantos perto de obstáculos; corrigido com margem de segurança extra na rasterização.

Revalidação de H1 com protocolo correto (pool misto de densidade, pareado por trial, 500 trials): **a motivação original de H1 (política aprendida supera A* em taxa de sucesso sob densidade) não se sustentou com dados reais** — A* real venceu em quase todo o espectro testado (90,6% vs. 85,8% do ρ-criterion, regret real de 8,4% contra 2,9% do mock). A motivação que sobreviveu, mais forte que antes, foi a de custo computacional: A* real custa até ~600× mais que uma política de imitação supervisionada em alta densidade, medido no mesmo ambiente e trials do resultado de sucesso.

H1 reformulada em ambos os documentos: de "supera qualquer planejador fixo em acerto" para "mantém acerto comparável ao melhor fixo pagando uma fração pequena do seu custo computacional". Decisão consciente de terminar essa investigação até o fim (medir custo, reformular os dois documentos) em vez de deixar pela metade, por prioridade declarada de maximizar rigor para submissões com prêmio (LAFusion 2026, CONPEEX).

Boxplots de distribuição e outliers gerados na sequência (revelando, por exemplo, que 14 dos 150 trials de custo A* em `very_dense` são outliers estatísticos, chegando a quase o dobro da mediana) — reforça o argumento de fusão sensível a custo com um planejador de pior-caso intermitente.

Uma citação incorreta introduzida durante essa reformulação (atribuindo o achado tainted da Fase 4 à Seção 3.4, que na verdade cita dados CBS reais e limpos) foi encontrada e corrigida no mesmo dia, ao auditar consistência entre documentos.

## Fase 9 — Escala, significância estatística e MARL real (09/07/2026, ~19-20h)

Auditoria adicional revelou três pontos a fechar antes de considerar a revalidação de H1 completa:

1. **Escala e significância.** A revalidação de H1 (Fase 8) usava 500 trials, sem justificativa explícita para essa escolha e sem teste de significância estatística. Corrigido: rerodado com 1.500 trials (mesma escala do Monte Carlo original, permitindo comparação direta de poder estatístico) — resultado consistente com a amostra menor (A* real 88,7% vs. ρ-criterion real 84,1%, regret 9,1%), confirmando que não era artefato de amostra pequena. Teste de McNemar exato pareado confirma significância (p<0,000002).

2. **Outliers ainda não mostrados.** Distribuição completa (não só médias) do custo de decisão A*/BC e de ρ_local, por mundo, gerada via boxplots — revelando que 14 dos 150 trials de custo A* em `very_dense` são outliers estatísticos (até 51,9 ms, quase o dobro da mediana), atribuídos corretamente a pares start-goal que exigem busca mais extensa, não a artefato de medição.

3. **A citação multi-agente corrigida (Fase 8) ainda não tinha sido regerada com A* real** — só removida do texto. Regerado com busca real por agente: o A* real continua vencendo em taxa de chegada ao goal, mas **também tem colisão inter-robô alta (75-80%)**, muito diferente da alegação original (com a política de linha reta) de 0% de colisão — o "0% de colisão" era um artefato específico do controlador de linha reta (parava quando desalinhado), não qualidade de planejamento. Conclusão qualitativa preservada (nenhum planejador independente sem coordenação evita colisão de forma confiável), mas agora com dado correto.

Por fim, uma pergunta direta do Yan — "não conseguimos fazer nenhum MARL em 2D só pra comparar?" — revelou que `MultiAgentEnv2D` nunca teve reward implementada (só servia para avaliar políticas já treinadas independentemente, não para treinar). Implementado:
- Reward real por agente, com **penalidade de colisão inter-robô compartilhada** (os dois agentes envolvidos recebem a penalidade, não só um "culpado" arbitrário) — o elemento estrutural que falta no RL independente.
- Política única centralizada (observação e ação concatenadas dos N agentes), treinada com PPO padrão do SB3.
- Primeira tentativa de treino colapsou para uma política parada (0% de sucesso) — causa: `VecNormalize(norm_reward=True)` esmagava o sinal de recompensa esparsa (±100). Desabilitar a normalização de reward resolveu.
- Resultado real (N=4, `sparse`): RL independente 57% goal / 60% colisão inter-robô. MARL centralizado, 150 mil passos: 25% goal / **0%** colisão. MARL centralizado, 600 mil passos: 50% goal / **0%** colisão — tendência clara de goal rate subindo com o treino enquanto a colisão permanece nula desde cedo.

Este é o primeiro resultado empírico desta IC (não apenas formalização teórica) mostrando que treino conjunto com recompensa compartilhada resolve o problema de coordenação identificado desde a Fase 4 — ainda uma simplificação de MARL (treino e execução centralizados, não a arquitetura descentralizada completa tipo QMIX/MAPPO), mas evidência real, não apenas prometida como trabalho futuro.

## Fase 10 — Motivação do critério e chattering (09/07/2026, ~20-21h)

Duas lacunas apontadas diretamente pelo Yan, fechadas na mesma sessão:

**Motivação da criação do ρ-criterion.** O relatório definia a fórmula do critério (densidade local, limiar 0,30) mas nunca explicava *por que* densidade foi escolhida como variável de contexto, nem *por que* um limiar rígido em vez de fusão suave — essa justificativa só existia no rascunho do LAFusion. Adicionado no início da Seção 2.2 dos dois documentos: densidade é a variável que *causa* o trade-off (custo do A* cresce com obstáculos, custo do SAC/BC é ~constante), não apenas uma métrica correlacionada; o limiar rígido é uma escolha deliberada por restrição de tempo real, não uma limitação não percebida — revisitada como trabalho futuro (fusão suave/probabilística).

**Chattering no critério — pesquisa de padrão-ouro.** A pedido do Yan ("como tornar o critério mais científico"), pesquisa de literatura atual (controle supervisório chaveado, predição conformal para calibração de limiar, bandits contextuais, mixture-of-experts) revelou um defeito clássico não testado: o ρ-criterion usa um limiar único sem histerese, um padrão conhecido por causar oscilação de decisão ("chattering") perto do limiar em sistemas de controle chaveado (literatura de Hespanha & Morse, décadas de maturidade). Testado nos dados reais: confirmado — 46 de 300 episódios (15%) com 2 ou mais trocas de planejador, média de 3,4 trocas por episódio, um caso chegando a 48 trocas num único episódio de 200 passos.

Descoberta colateral no processo: a validação de H1 usada até este ponto (Fase 8-9) decidia o planejador **uma única vez no reset do episódio**, não a cada passo como o critério está formalmente definido ("em tempo de execução") — o chattering só se manifesta na versão per-step, que é a correta.

Corrigido com histerese (dois limiares: ρ_low=0,28 para voltar ao A*, ρ_high=0,32 para ir ao SAC/BC, zona morta entre os dois — padrão-ouro de controle supervisório). Piloto de 100 trials confirma a correção: chattering cai de 15% para 5% dos episódios, máximo de trocas por episódio cai de 48 para 4. A rodada completa de 1.500 trials (mesma escala das validações anteriores) ficou pendente para a próxima sessão, assim como a calibração do limiar via predição conformal (linha de pesquisa de 2025 identificada como o padrão-ouro mais atual para thresholds com garantia formal, ainda não implementada).

## Fase 11 — Checkpoint SAC 650k quebrado e regressão do MARL centralizado (26/07/2026)

**Checkpoint SAC 650k permanentemente quebrado.** Ao tentar retomar o treino SAC via `docker compose run --rm train-all` (resume automático do checkpoint mais recente), o `PlanBCallback` abortou de forma **determinística**: `ep_rew_mean=-99.0` no primeiro step avaliado após o resume (`num_timesteps=650001`), toda vez, em pelo menos 18 reinícios consecutivos observados. Como `restart: unless-stopped` estava ativo nesse serviço, o container entrou em crash-loop, reiniciando a cada poucos segundos e consumindo CPU real por horas sem produzir nada — só percebido ao investigar por que o Gazebo parecia sempre ocupado. `sac_42_ckpt_650000_steps.zip` está confirmado inutilizável como ponto de resume; `sac_42_ckpt_600000_steps.zip` é o último checkpoint não corrompido disponível. `restart` desse serviço mudado para `"no"` no `docker-compose.yml` até o checkpoint base ser trocado ou a causa do colapso entre 600k→650k ser investigada.

**Bug de escape `\$` no `docker-compose.yml`** impedia o serviço `gazebo` (compartilhado com `benchmark`) de sequer subir: `\$` não é o escape correto para variável de shell dentro de um `command:` do Compose (gera `"variable not set"` e, pior, quebra o parsing do bash quando a substituição vira string vazia dentro de um loop `for`) — o escape certo é `$$`. Corrigido nos três serviços afetados (`train-all`, `gazebo`, `gzclient`).

**Regressão do MARL centralizado — causa raiz é arquitetural, não hiperparâmetro.** A Fase 9 (09/07/2026) registrou um resultado real de 50% goal-rate / 0% colisão em 600 mil passos, com tendência de melhora clara. Cinco tentativas de retreino nesta sessão (mesmos hiperparâmetros base, mais dois fixes: reward de robô-já-terminado não zerar mais a média do grupo, e `ent_coef` 0,01→0,05) deram **0% goal-rate em todas**, incluindo uma de 1 milhão de passos.

Duas hipóteses testadas e refutadas com dado real, em ordem:

1. **Diluição do sinal por `np.mean`** (`train_2d_marl.py:63`): trocado para `np.sum` + episódio termina no primeiro agente pronto (`terminate_on_any=True`, novo parâmetro em `env_2d_multi.py`). Validado com 200k passos: **ainda goal_rate=0%**. Ao inspecionar a política treinada diretamente (não só a métrica agregada), a distância média ao goal por agente **piora** ao longo do episódio (ex.: 0,88m→1,66m em 200 passos) — a política diverge do goal, não apenas estagna perto dele como hipotetizado em sessões anteriores.
2. **`R_SURVIVAL=0,1` competindo com `R_APPROACH`**, hipótese levantada por pesquisa de literatura (ver abaixo): testado com `R_SURVIVAL_OVERRIDE=0.0`, mesmos 200k passos. **Também deu goal_rate=0%.**

Diante das duas hipóteses baratas refutadas, pesquisa de literatura (MARL, credit assignment, 2018-2025) foi conduzida antes de qualquer nova tentativa às cegas. Conclusão forte e citada: a arquitetura atual — uma única política PPO padrão do Stable-Baselines3 recebendo observação/ação **concatenada** dos 4 agentes como um "meta-agente" de dimensão 4x — não é uma simplificação válida de MARL, é controle centralizado single-agent sem nenhum mecanismo de credit assignment por agente. PPO não consegue separar qual dos 4 sub-vetores de ação (32 dims) causou qual parte da reward agregada (escalar único), com ou sem soma/média. O padrão-ouro citado é MAPPO (Yu et al., 2021, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", arXiv:2103.01955): pesos compartilhados entre agentes, mas GAE/advantage calculado **por agente**, não reduzido a escalar antes do backward — o mesmo paper introduz "death masking" (zerar o estado de um agente já terminado, mantendo apenas seu ID, em vez de continuar propagando estado real ou zerar tudo) como solução testada e superior para terminação assíncrona dentro do mesmo episódio, o problema estrutural equivalente ao que a hipótese 1 tentou remendar com `terminate_on_any`. Também citado: Foerster et al. 2018 (COMA) sobre credit assignment em ação conjunta, Du & Ding 2023 (arXiv:2312.05162, review) e de Witt et al. 2020 (arXiv:2011.09533) sobre aprendizado independente vs. centralizado em SMAC.

**Decisão consciente**: a correção raiz (MAPPO real, advantage por agente) é uma reescrita do loop de treino, não um ajuste de hiperparâmetro — fora do PPO padrão do SB3 usado até aqui. Adiada para uma sessão dedicada; este achado fica registrado como a explicação definitiva do porquê 7 tentativas consecutivas (5 desta sessão + 2 hipóteses adicionais testadas) convergiram sempre para o mesmo resultado nulo, apesar de mudanças de hiperparâmetro, reward shaping e agregação.

**Benchmark Gazebo com dados 100% inválidos, e uma segunda causa raiz de infraestrutura.** Com o bug de escape corrigido, o serviço `gazebo` subiu, e 15 trials do `run_benchmark.py` rodaram até o fim — mas todos deram `timeout=True`, `path_length_m=0,0`: o robô nunca se move. Causa 1 (corrigida): `run_benchmark.py` ainda usava `/gazebo/set_entity_state` para teleportar o robô entre trials, o mesmo serviço inexistente já diagnosticado na Fase 6 item 4 — corrigido para `delete_entity`+`spawn_entity`, espelhando `gazebo_gym_env.py` (o fix nunca tinha sido propagado para o script de benchmark; um segundo bug do mesmo tipo, nome de entidade `"turtlebot3_waffle"` em vez de `"waffle"`, também corrigido em `get_robot_pose()`). Causa 2 (nova, não resolvida): mesmo com o fix de nome de serviço, chamadas de `/spawn_entity` **do container `benchmark` para o container `gazebo`** (dois containers Docker separados, `network_mode: host`, mesmo `ROS_DOMAIN_ID=0`) dão timeout de forma consistente — testado 3x, 100% de falha —, enquanto a mesma chamada, feita de dentro do próprio container `gazebo`, funciona imediatamente. É um problema de descoberta DDS entre containers Docker distintos, diferente do bug de nome de serviço. O padrão que já funciona no projeto (`train-all`) evita isso rodando Gazebo e cliente no mesmo container — o próprio `docker-compose.yml` já documentava isso ("RECOMENDADO — tudo num único container, sem problema de DDS inter-container"). Correção (unificar `gazebo`+`benchmark`, ou investigar a config de DDS entre containers) fica para a próxima sessão.

---

## Lições gerais deste processo

- **Bugs de infraestrutura, não de algoritmo, dominaram o tempo de depuração.** Nenhum dos problemas graves encontrados nesta IC (nomes de serviço, callbacks silenciosamente quebrados, Docker não aplicando config) era sobre RL, reward, ou o critério ρ em si.
- **Validar em ambiente rápido antes do lento evita horas de debug no ambiente errado** — a regra "2D antes do Gazebo", quando seguida, isolou corretamente cada causa raiz em minutos em vez de horas.
- **Um resultado que não se sustenta sob dados reais não deve ser escondido atrás do resultado calibrado mais favorável** — a reformulação de H1 é o exemplo mais direto disso nesta IC.
- **Corrigir um bug de padrão repetido exige checar todas as ocorrências do mesmo padrão**, não só a instância que motivou a investigação original (aconteceu duas vezes: o bug do `PlanBCallback`, e a política de linha reta rotulada como "A*" em dois lugares diferentes do código).
- **Perguntas simples ("dá pra comparar com MARL?") às vezes revelam que a peça básica nunca existiu** — `MultiAgentEnv2D` nunca teve função de reward, apesar de já ser usado para "avaliar" políticas há semanas; só apareceu ao tentar treinar de verdade pela primeira vez.
- **"Decide uma vez" e "decide a cada passo" são políticas diferentes, mesmo quando o critério é o mesmo por escrito.** A validação de H1 rodou por duas fases inteiras (8 e 9) com uma política decide-uma-vez-no-reset sem que isso fosse percebido como divergente da definição formal do critério — só apareceu ao investigar chattering, que não pode existir numa política que decide uma única vez.
- **Normalização de reward pode destruir sinal esparso.** `VecNormalize(norm_reward=True)` colapsou o treino de MARL para uma política parada — recompensas terminais grandes (±100) contra a maioria dos passos com recompensa pequena fazem a normalização por variância comprimir o sinal que mais importa.
