# Seções excluídas do relatório final (guardadas caso precisem retornar)

Excluído em 29/08/2026 do `relatorio_final_pip.tex` para reduzir de 19 para ~15-16 páginas,
depois de aumentar o tamanho de exibição das figuras. Conteúdo científico real, não prosa
redundante — preservado aqui integralmente, com as figuras e tabelas originais, para o caso de
ser necessário reincorporar ao corpo ou usar como material suplementar/anexo.

## 1. Figuras da curva de aprendizado e comparação de trajetórias (Seção "Integração ROS2/Gazebo")

Figura (par empilhado):
- (a) `figs/2d/fig_2d_learning_curve_ci.png` — "Curva de aprendizado do SAC (sparse), 3 seeds."
- (b) `figs/2d/fig_2d_compare_dense.png` — "Trajetórias no ambiente denso: A*, BC e Adaptativo."

Texto de referência que citava essas figuras:

> Para acelerar o ciclo de iteração de reward e validar hiperparâmetros antes do Gazebo, foi
> implementado um ambiente 2D leve (`eval/env2d/`) com raycasting vetorizado NumPy, 851× mais
> rápido que o Gazebo Classic, reproduzindo fielmente a cinemática unicycle, a observação de 29
> dimensões e a mesma estrutura de reward do ambiente ROS2. O SAC convergiu para 90% de taxa de
> sucesso em ≤14.000 passos (≤3,3 minutos) no mundo esparso, reproduzível em 3 seeds
> independentes; a segunda figura compara as trajetórias de A* (linha reta analítica), BC e
> política Adaptativa nos três ambientes.

## 2. Diagnóstico do bug R_APPROACH (Gazebo)

> O ambiente 2D também serviu para diagnosticar por que o treinamento SAC no Gazebo não
> convergia: comparando as constantes de reward entre os dois ambientes, todas coincidiam exceto
> `R_APPROACH` (2,0 no Gazebo contra 10,0 no Env2D). Com sobrevivência valendo +20 em 200 passos
> parado, o bônus de aproximação precisa superar 6,67 para que navegar valha mais que ficar
> parado; com 2,0 o agente aprendia a não se mover. Corrigido para 10,0, o Gazebo passou a
> apresentar episódios estáveis e crescentes.

## 3. Nota sobre CrossQ/DDPG (parágrafo dos quatro paradigmas)

> Ao longo da IC foram aplicados e comparados quatro paradigmas de aprendizado: planejamento
> clássico determinístico (benchmark real de tempo/memória), RL single-agent (SAC, principal;
> CrossQ testado e descartado por baixo desempenho), aprendizado supervisionado (Behavior
> Cloning, melhor resultado quantitativo da IC) e planejamento multiagente via CBS (evidência
> real de que a arquitetura se sustenta fora do Monte Carlo; não é MARL treinado, apenas a base
> para o MARL da Fase 3).

(DDPG: mencionado nas Limitações do relatório final como "substituído por SAC sem teste isolado"
— essa menção pontual permanece no corpo, só o parágrafo dos "quatro paradigmas" acima foi
retirado.)

## 4. Seção completa: Chattering e correção por histerese + Cenário urbano dinâmico

### Chattering e correção por histerese

> Um limiar único sem zona morta é suscetível a oscilação de decisão (chattering) perto de
> ρ*=0,30, defeito clássico de sistemas de controle chaveado (Hespanha, Liberzon & Morse,
> Automatica 2003), que trabalhos recentes de comutação planejador clássico/aprendido continuam
> a apontar como lacuna aberta do campo por não modelarem histerese ou tempo mínimo de
> permanência entre trocas (Ji, Bamdad & Cruz, ACRA 2025). Um piloto de 300 trials estimou que
> 15% dos episódios sofriam ≥2 trocas de planejador (até 48 num único episódio de 200 passos),
> corrigido com histerese (ρ_low=0,28, ρ_high=0,32, zona morta entre os dois: a decisão só comuta
> de A* para BC acima de ρ_high e de BC para A* abaixo de ρ_low).
>
> Repetindo os dois protocolos completos em n=1.500 (mesma escala das demais validações desta
> seção), o efeito é real mas mais modesto que o piloto sugeria: episódios com ≥2 trocas caem de
> 4,9% (74/1.500, sem histerese) para 3,3% (49/1.500, com histerese), e o máximo de trocas por
> episódio cai de 6 para 4. A diferença entre a estimativa do piloto (15%→5%) e a medição
> completa (4,9%→3,3%) é, em si, um lembrete de que amostras pequenas superestimam efeitos,
> motivo adicional para preferir n=1.500 como padrão em toda revalidação deste trabalho.

**Nota de decisão editorial:** esta subseção específica (histerese) tem valor científico alto
(cita lacuna aberta do campo, dado real de n=1.500) — foi listada como prioridade alta na
hierarquização, não como candidata a corte. Confirmar antes de reincorporar se ela deve voltar
independentemente do resto desta seção urbana.

### Cenário urbano e obstáculo dinâmico

> As seções anteriores validam o critério em arenas circulares abertas. O item "ambientes
> urbanos" do Plano de Trabalho (PI08078-2024) e a extensão a obstáculos dinâmicos, prevista mas
> não coberta pelo Gazebo Classic, motivaram um terceiro testbed no gêmeo 2D: `urban_grid`, um
> layout com quatro quarteirões sólidos e corredores de 1,4 m formando um cruzamento em "+", com
> semântica de rua em vez da arena circular aberta usada até aqui. É o experimento com maior
> volume de dados reais desta IC: 2.000 trials (500 por condição, quatro condições), A* real e BC
> real casado por densidade.
>
> Às três condições dinâmicas somam-se duas extensões que isolam o efeito do número de
> obstáculos do efeito da velocidade: 3 obstáculos simultâneos cobrindo os dois eixos do
> cruzamento, e o mesmo layout ao dobro da velocidade.

**Tabela — Cenário urbano: sucesso e eficiência de rota (percorrido/ótimo) por condição,
N=500/condição, A* e BC reais pareados**

| Condição | A* (sucesso) | BC (sucesso) | ρ-crit. (sucesso) | A* (eficiência) | BC (eficiência) | ρ-crit. (eficiência) |
|---|---|---|---|---|---|---|
| Estático | 98,8% | 100% | 99,6% | 0,811 | 0,886 | 0,859 |
| 1 obstáculo | 49,2% | 48,8% | 48,6% | 0,551 | 0,636 | 0,602 |
| 3 obstáculos | 39,0% | 35,8% | 36,4% | 0,466 | 0,550 | 0,509 |
| 3 obstáculos, 2× vel. | 31,8% | 28,8% | 31,4% | 0,388 | 0,433 | 0,425 |

> O padrão central deste testbed é o empate. No estático repete-se a leve vantagem do BC em
> ambiente estruturado já vista nas seções de revalidação de H1 e multi-agente; sob perturbação
> dinâmica, nenhum dos três métodos se distingue do outro em nenhuma das três condições, ao
> contrário da revalidação de H1 em arena aberta (onde o A* vencia com folga). A degradação é
> monotônica e paralela entre os três conforme a complexidade cresce: nenhum absorve a
> perturbação melhor que os outros, e a escolha entre eles permanece justificada apenas pelo
> critério de custo.

## 5. Seção completa: Extensão Multi-Agente

> O Relatório Parcial (aprovado em 01/04/2026) já mostrou, com dados reais gerados pelo pipeline
> CBS sobre mapas públicos, que o ρ-criterion funciona com vários robôs sem adaptação: cada
> agente calcula a própria densidade e decide sozinho. Os números acompanham a densidade do
> ambiente: em 100 cenários com 2 agentes (ρ≈0,19), 93% dos passos usaram o planejador clássico;
> com 5 agentes, o congestionamento eleva o uso do RL a 26% (44% num cenário canyon). A troca
> ocorre sempre perto de ρ≈0,30, qualquer que seja o número de robôs, pois ρ mede uma propriedade
> do ambiente, não do planejador: critério e limiar permanecem inalterados mesmo com a transição
> RRT*/PPO → A*/SAC. Um experimento de desvio sobre os mesmos 100 cenários confirma o custo de
> não seguir o critério: sempre A* custa +2,1%; sempre SAC, +14,1%.
>
> **Por que decisão local: escalabilidade do CBS.** Embora o CBS produza soluções ótimas livres
> de colisão, seu custo cresce de forma super-linear com o número de agentes (Sharon et al.,
> 2015): num experimento de escalabilidade (grid 10×10, timeout 30 s), o tempo salta de ≈5 ms em
> N=2 para ≈160 ms em N=8, já com um caso censurado. O CBS centralizado não escala para frotas em
> tempo real; o ρ-criterion decide localmente em O(1) por agente, sem comunicação nem estado
> global.
>
> **O limite do RL independente: motivação para MARL.** Os experimentos acima usam trajetórias
> CBS pré-computadas, ou seja, coordenação garantida externamente. Sem essa garantia (N robôs
> executando a política de forma independente, vizinhos percebidos como obstáculos no LIDAR;
> dados reais, 20 trials/configuração), a chegada ao objetivo cai consistentemente com N (em N=8,
> de 38% no esparso a 12% no muito denso) e a colisão entre robôs satura próximo de 100% para
> N≥5, com deadlock em 0%: os robôs colidem, não travam. O problema é falta de coordenação, não
> dificuldade do ambiente. Nenhuma estratégia sem treino conjunto o resolve, o que situa a
> coordenação sob decisão local reativa como domínio do MARL (Fase 3). Para robôs reais, o plano
> discreto do CBS ainda precisaria ser convertido em schedule cinemático viável, via
> pós-processamento por Grafo de Plano Temporal (Hönig et al., 2016).

Figuras (par empilhado):
- (a) `figs/cbs/cbs_scalability.png` — "Escalabilidade do CBS (log): timeout já em N=8."
- (b) `figs/marl/fig_marl_degradation_by_density_2d.png` — "Degradação do RL independente:
  colisão ≈100% para N≥5."

## 6. Figura MARL preliminar não reproduzida

`figs/marl/fig_marl_shared_reward_comparison.png` — "MARL com reward compartilhada, N=4:
resultado preliminar não reproduzido em retreinos posteriores."

Texto de referência:

> O mesmo tratamento (linha reta → A* real) aplicado ao achado multiagente de uma sessão anterior
> confirma que nenhum planejador independente sem coordenação evita colisão entre robôs de forma
> confiável (A* real: 75–80% de colisão inter-robô, N=4, contra a alegação original incorreta de
> 0%), reforçando a necessidade de MARL. Uma versão simplificada de MARL centralizado (política
> única, PPO, reward compartilhada) chegou a reduzir a colisão inter-robô de 60% para 0% mantendo
> taxa de goal comparável, mas o achado não se reproduziu: sete retreinos posteriores convergiram
> todos para 0% de chegada ao goal. A causa provável é estrutural: uma política única com PPO
> padrão sobre observação/ação concatenadas não faz atribuição de crédito por agente, ao
> contrário do padrão MAPPO (Yu et al., NeurIPS 2022), que mantém pesos compartilhados mas
> calcula vantagem por agente. O resultado é tratado aqui como preliminar e não confirmado; a
> reformulação de H1 não depende deste achado, apenas do resultado de custo.

## Referências cruzadas a ajustar se este conteúdo voltar

Ao reincorporar, checar: `\label{sec:urbano}`, `\label{sec:multiagente}`,
`fig:ic_learning_curve`, `fig:ic_compare`, `fig:ic_cbs_scalability`, `fig:ic_marl_degradation`,
`fig:ic_marl_shared`, `tab:urbano` — todos removidos do `.tex` principal junto com o texto acima.
A Conclusão do relatório final também cita `Seção~\ref{sec:urbano}` e
`Seção~\ref{sec:multiagente}` na cobertura de objetivos (item 2) e no parágrafo de H1/H2 — essas
referências foram ajustadas para não apontar mais para seções removidas.
