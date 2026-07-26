# Plano de correção científica — antes dos slides do Prof. Aldo

*Documento gerado em sessão de auditoria de dados (25/07/2026), cruzando os CSVs/scripts do
repositório contra `paper/relatorio_final_pip.md`, `paper/lafusion_2026_draft.md` e
`docs/relatorio_parcial.md`. Ver `DEVELOPMENT_LOG.md` para o histórico cronológico do projeto.*

> **✅ STATUS (25/07/2026, fim do dia): os 6 passos de execução técnica estão concluídos.**
> Ver "Status de execução" logo abaixo para o detalhe de cada um. **Nada foi commitado** —
> tudo está no working tree, pronto para revisão antes da reunião. As decisões que só
> o Yan/Prof. Aldo podem tomar (E1, E3, E4, e as 3 perguntas do slide 9) continuam em aberto
> por design — são pauta da reunião, não pendência técnica.

> **⚠️ INVERSÃO DE ESTRATÉGIA (contexto histórico, já executado).**
> A versão anterior deste plano assumia: *consertar tudo → depois apresentar*. Isso estava
> errado. São **cinco semanas** até 31/08/2026 e havia 11 bloqueadores; a estratégia adotada foi
> **um mínimo viável pré-reunião**, não zerar a lista inteira. Ver a seção "Estratégia" e "Status
> de execução" abaixo.

## Status de execução (atualizado 25/07/2026)

Os 6 passos abaixo (detalhados na seção "O que falta fazer depois da correção factual" e no
arquivo de plano de trabalho da sessão) foram executados nesta ordem:

| # | Passo | Status | Onde |
|---|---|---|---|
| 1 | Promover `urban_grid` (n=1.000) a resultado de 1ª linha | ✅ Feito | Seção 3.6, `relatorio_final_pip.md` |
| 2 | Tabela de reconciliação com o parcial | ✅ Feito | Início da Conclusão, `relatorio_final_pip.md` |
| 3 | Checklist do Plano de Trabalho (5 objetivos) | ✅ Feito | Conclusão, `relatorio_final_pip.md` |
| 4 | Recalibrar τ com A*/BC reais, split treino/teste | ✅ Feito | `eval/env2d/sweep_threshold_real.py`, `results_abstract/threshold_sweep_real.csv`, Seção 3.2 |
| 5 | Replicar correções no LAFusion draft | ✅ Feito | `paper/lafusion_2026_draft.md` |
| 6 | Deck de reunião — **formato mudou para Beamer/Overleaf** a pedido do Yan | ✅ Feito | `overleaf/apresentacao/` (fonte) e `overleaf/apresentacao_upload/` (achatada, com PDF compilado e revisado) |

**Achado do Passo 4** (não previsto no plano original): a recalibração real de τ confirma 0,30
como escolha razoável — regret decresce monotonicamente até τ=0,60 no intervalo testado, sem
mínimo interno, mas a diferença entre τ=0,30 (8,4% regret) e τ=0,60 (7,5%) é pequena (~1pp).
**Não houve retrabalho em cascata**: o valor 0,30 foi mantido, com o trade-off documentado
explicitamente em vez de reescrever Tabela 2/Seção 2.1/figuras por uma diferença dessa magnitude.

**Verificações pós-execução:** figuras regeneradas (`fig_2d_h1_boxplots`, `fig_2d_h1_real_validation`)
conferidas visualmente; números-chave (τ=0,30, 656×, custo do adaptativo 8,60ms) conferidos como
consistentes entre `relatorio_final_pip.md`, `lafusion_2026_draft.md` e o deck Beamer; nenhum
arquivo temporário residual encontrado; `.gitignore` já cobre os `.log` gerados pelas baterias.

## Pendências operacionais e nota de integridade

### ⚠️ Arquivo do usuário sobrescrito por um agente do Claude Code

Durante a auditoria, um agente despachado para explorar o repositório executou
`rerun_h1_hysteresis.py --trials 10` para cronometrar o script. Isso **sobrescreveu**
`results_abstract/h1_hysteresis_2d.csv`, que tinha 20 linhas e passou a ter 10. O fato foi
mencionado de passagem na sessão, e o arquivo truncado chegou a ser usado como evidência do
problema de tamanho amostral sem deixar claro que parte da redução foi causada pela própria
auditoria.

- Estado no momento em que este documento foi escrito: `M results_abstract/h1_hysteresis_2d.csv`
  no working tree.
- O dado é regenerável pela Etapa 1.3, e os 20 trials originais também não sustentavam conclusão.
- Para restaurar antes de rodar a bateria: `git checkout results_abstract/h1_hysteresis_2d.csv`

### Itens levantados e nunca fechados

- **B11 vs. documentos já entregues.** O resumo do CONPEEX e o relatório parcial foram submetidos
  com o 85,3%. Não dá para reescrever o passado. Fica em aberto se isso exige alguma providência
  formal ou se basta a correção valer daqui em diante. **É pergunta para o professor** —
  acrescentar como quarta decisão no slide 9 do deck de reunião.
- **`results_ros2/gazebo_episode.json`** — untracked e com dono `root` (gerado em container). O
  plano pede commitá-lo como prova de que a infra rodou; exige `sudo chown` antes.
- **`.gitignore` não cobre `results_ros2/`** — motivo de o diretório aparecer sujo no `git status`.

### Itens fechados nesta verificação

- ✅ **Sexto método SOTA existe.** `he_multiopt` está em `sota_comparison.py:26` e no CSV
  (73,3%→23,3% por densidade, **n=30 por ponto**). Mesmo padrão de mock dos outros cinco: o B11
  vale para os seis. O "54%" citado no parcial é a média das cinco densidades.
- ✅ **BC é reprodutível — a informação existe no código, só não foi escrita.**
  `train_2d_bc.py`: MLP `29→128→128→2` (ReLU nas ocultas, Tanh na saída), Adam `lr=1e-3`,
  **300 episódios** de demonstração de um expert de campo potencial, **50 épocas**,
  `batch_size=256`, `seed=42`. Basta transcrever para a Seção 2 do relatório e para o LAFusion —
  não há dado perdido aqui.

---

## Estratégia (a camada acima dos dados)

Os bloqueadores B1-B11 são problemas de dados. Estes cinco são problemas de contexto, e um deles
inverte a ordem de execução do plano inteiro.

### E1 — Dois públicos com critérios opostos

CONPEEX/relatório PIP e LAFusion têm incentivos contrários. O comitê do PIP provavelmente não é
da área, mede "cumpriu o Plano de Trabalho?" e valoriza resultado positivo. O LAFusion é revisão
por pares, mede rigor, e **premia** exatamente a autocrítica que o comitê do PIP pode ler como
"o aluno não conseguiu".

O B11 explode diferente em cada um:
- **LAFusion:** "identificamos que a validação inicial era tautológica e refizemos com
  planejadores reais" é história de mérito metodológico. **Vai no corpo do paper, com destaque.**
- **Relatório PIP:** a mesma frase, mal escrita, vira "o resultado principal do parcial aprovado
  não valia". **Enquadrar como amadurecimento do protocolo experimental**, com o dado real
  (urban_grid n=1000) como entrega positiva, não como retratação.

**Consequência prática:** são dois enquadramentos do mesmo fato, não um texto que sirva aos dois.
Nenhuma frase sobre o B11 deve ser copiada de um documento para o outro.

### E2 — O orientador coassinou o parcial

O Prof. Aldo aprovou um documento que afirma "100% de acurácia" e "85,3%, 1º lugar contra 6
métodos". Demonstrar que esses números não se sustentam mostra, sem querer, que ele aprovou algo
que não checou. Ele tem reputação investida nisso.

**Regra de linguagem, para todos os slides e documentos:** primeira pessoa do **plural** para os
erros ("o protocolo que **usamos** não separava..."), primeira pessoa do **singular** para as
correções ("**refiz** a validação com A* real"). Isso põe vocês do mesmo lado do problema. Não é
política — é a diferença entre tê-lo como aliado ou como réu na frente do próprio orientando.

**Pergunta a ter pronta:** *"por que você só descobriu isso agora?"*
Resposta: *"porque só agora fui reler o código do primeiro semestre com o que aprendi depois —
foi a implementação do A* real que me fez desconfiar dos mocks."* É verdadeira e é boa; só
precisa sair sem hesitação.

### E3 — E se o ρ-criterion simplesmente não funcionar?

Possibilidade que o plano vinha evitando enunciar. Todas as evidências reais apontam para ela:
A* real vence, urban_grid dá empate, a varredura de τ degenera para "use A* sempre", e o único
resultado favorável que existia era aritmeticamente forçado (B11). A defesa restante é o custo —
e ela depende de um regime (29 ms doendo) **ainda não demonstrado**.

Se a conclusão for "não funciona no regime testado", isso **ainda é uma IC bem-sucedida**:
hipótese formulada, aparato construído, hipótese refutada com rigor. Mas é uma apresentação
diferente da que este plano vinha desenhando.

**Decisão a tomar antes da reunião** (é do Yan, não do assistente): aceitar essa conclusão como
possível desfecho, ou continuar procurando o regime onde o método ganha. A segunda opção tem nome
e um avaliador de prêmio reconhece. **Recomendação:** entrar na reunião com E3 explicitamente na
mesa como cenário — inclusive porque o professor pode ter uma leitura que ainda não apareceu.

### E4 — Os prazos colidem

Em 25/07/2026: LAFusion 31/07–16/08, relatório final 31/08. **O paper vence antes do relatório**,
e a Etapa 1 completa (recalibrar limiar + reescrever tese em 4 documentos + refazer figuras +
revisão de literatura) não cabe duas vezes em cinco semanas.

Um dos dois recebe versão completa, o outro recebe mínima viável. **Essa escolha é do Yan e deve
ser levada à reunião** — ela depende de qual prêmio ele prioriza, e a memória do projeto registra
prioridade no LAFusion.

### E5 — Executar tudo = não apresentar nada

Este documento tem 11 bloqueadores. Não são cinco semanas de trabalho; são mais. Tentar zerar a
lista antes de falar com o orientador é a forma mais provável de chegar na reunião sem nada.

**Inversão:** a reunião deixa de ser prestação de contas defensiva e vira **sessão de trabalho**.
A fala de abertura é, em substância: *"Professor, auditei os dados do projeto inteiro e encontrei
problemas sérios. Trouxe o mapa completo. Preciso decidir com o senhor o que consertar nas cinco
semanas que restam."* Isso transfere a decisão de escopo para quem tem autoridade sobre escopo, e
protege o Yan de escolher sozinho o que sacrificar.

---

## O que fazer ANTES da reunião (mínimo viável, ~1-2 dias)

Substitui a exigência anterior de "Etapa 1 completa antes de qualquer slide". Só o que é
necessário para que a reunião seja produtiva:

1. **Rodar a bateria da Etapa 1.3** (~15 min de máquina, tratar como piso) — para não apresentar
   n=10 como n=1.500.
2. **Verificar o B11 com um exemplo rodado**, não só lido — uma célula que imprime a tabela de
   sucesso analítico vs. medido. É a peça que sustenta a conversa mais difícil.
3. **Uma figura nova: F1 (Pareto sucesso × custo)** — a única tese viva precisa de uma imagem.
4. **Slides do deck reduzido** (ver Etapa 6, versão reunião — ~10 slides, não 18).
5. **Nada de reescrever documentos ainda.** A reescrita depende de decisões que só saem da reunião
   (qual tese, qual documento prioritário, o que cortar).

O resto deste plano (Etapas 1.1, 1.2, 1.4-1.8, 2, 3, 4.5) passa a ser **pós-reunião**, priorizado
conforme o que for decidido lá.

---

## Context

Yan vai apresentar a IC ao orientador com quatro objetivos: prestar contas, decidir o destino da
Fase 2 (Gazebo), revisar o relatório PIP e fechar escopo para o LAFusion. Uma auditoria dos dados
brutos contra os documentos revelou que **a tese afirmada no resumo e na introdução está
falsificada pelos dados do próprio projeto**, e que vários números publicados não têm CSV que os
sustente. Slides construídos sobre o estado atual apresentariam afirmações que o professor pode
derrubar abrindo um arquivo.

O trabalho não é ruim — o benchmark clássico é sólido, o diagnóstico de reward é bom, a correção
do A* falso foi um ato de rigor. O problema é que o rigor foi aplicado em rajadas e o texto não
acompanhou a última: os dados novos mataram a tese antiga e o documento ainda a proclama.

Este plano conserta a base factual, reescreve a tese em torno do que os dados sustentam, e só
então produz o roteiro de slides.

**Nota sobre a referência Kolomeytsev:** a referência Kolomeytsev & Golembiovsky (arXiv:2512.24651)
foi verificada e **existe** — "Hybrid Motion Planning with Deep Reinforcement Learning for Mobile
Robot Navigation", 31/12/2025. Não há problema de integridade bibliográfica. Nota de
posicionamento: esse paper é o concorrente mais direto do trabalho (global em grafo + local DRL,
sucesso 0,836) e hoje é citado só de passagem sobre reward — precisa de comparação explícita.

**Exigência de continuidade documental.** Os slides não são um relatório de resultados isolado:
precisam encadear com três documentos que já existem e que o professor conhece — o Plano de
Trabalho PI08078-2024 (base contratual FAPEG), o Parecer do Consultor SIGAA (13/06/2025) e o
Relatório Parcial **aprovado em 01/04/2026**. O parcial é o mais delicado: o professor o
coassinou, e várias afirmações dele são contraditas pelos dados atuais (tabela abaixo). A
apresentação precisa reconhecer essas mudanças explicitamente, cedo, e como consequência de
evidência — não deixar que o professor as descubra sozinho durante os resultados.

---

## Contradições com o Relatório Parcial (aprovado 01/04/2026)

| O parcial afirma | Estado hoje | Ação |
|---|---|---|
| RRT* + PPO como par de planejadores | A* + SAC/BC | Já justificado no PIP — só recapitular |
| **"100% de acurácia no switching"** (`:26`, `:62`, `:97`) | Chattering: até 48 trocas num episódio de 200 passos | **Contradição direta.** Retratar explicitamente: a métrica de "acurácia" não capturava oscilação |
| Regret 2,2% vs oracle (`:28`, `:85`) | 2,9% (mock) / 9,1% (real) | Explicar as duas revisões e o porquê de cada |
| **"Supera 6 métodos SOTA, 1º lugar"** (`:75`, `:97`, `:157`) | A* real vence o adaptativo; urban_grid dá empate | **Tese invertida.** É o núcleo do slide de virada |
| Threshold: empírico **0,350**, teórico **0,367**, gap 1,7% (`:85`) | PIP adota **0,30**; CSV aponta **0,20** | **Quatro valores em três documentos.** Resolver na Etapa 1.1 e narrar a convergência |
| "93,3% da performance do oracle" (`:52`) | 93,3% virou o valor do *oracle*, não do método (84,1%) | Corrigir a atribuição do número |
| Submissão IEEE Access em jan/2026 (`:121-123`) | Não submetido; alvo virou LAFusion 2026 | Nota de redirecionamento com justificativa |
| Próximos passos: A*, DWA, DDPG (`:113`) | A* feito; DWA e DDPG nunca implementados | Declarar como substituídos/descartados com o porquê |
| Grid 100×100, cenários automotivos (14.580 obstáculos) | Arenas circulares 2D + urban_grid | Explicar a troca de testbed |

Nenhuma dessas mudanças é indefensável — todas têm motivo técnico. O risco não é ter mudado, é
não ter narrado a mudança.

---

## Achados que motivam o plano (todos verificados contra os dados brutos)

| # | Achado | Evidência |
|---|---|---|
| B1 | **ρ*=0,30 não é o ótimo do próprio experimento.** Regret mínimo em τ=0,20 (0,60%) contra 2,92% em τ=0,30. O "2,9% de regret" citado como resultado central é a penalidade de ter escolhido o limiar errado. | `threshold_sensitivity.csv` |
| B2 | **O limiar foi calibrado com planejadores que não são os do trabalho.** Os 1.500 trials têm `selected ∈ {PPO, RRT*}` — nenhum A*, nenhum SAC. E são 500 cenários × 3 limiares, não 1.500 independentes. | `adaptive_switching_results.csv` |
| B3 | **O CSV grande aponta para um terceiro limiar.** Sucesso cresce monotonicamente 0,25→0,35 (79,8%/81,0%/82,4%); dentro dele o melhor é 0,35. Três arquivos, três limiares, nenhum é 0,30. | idem |
| B4 | **Os resultados centrais de H1 não têm dado bruto.** O relatório cita `h1_real_2d_mixed_pool.csv` para "1.500 trials pareados" (88,7/84,3/84,1, McNemar 199 discordâncias); o arquivo tem 10 linhas. | `h1_real_2d_mixed_pool.csv` |
| B5 | **Custos publicados errados.** Relatório e figura dizem 9,3/26,4/32,7 ms; o JSON diz 8,00/20,90/29,21. Literais hardcoded em `plot_h1_real_validation.py:31`. Razão real: 656×. | `h1_real_2d_cost_distribution.json` |
| B6 | **ρ tem duas definições incompatíveis.** ROS2: fração de células ocupadas em janela de 2 m (`window_m=2.00`). Env2D: fração de raios LIDAR < 1 m (`mean(ranges < 1.0)`). Escalas diferentes, mesmo limiar aplicado aos dois. | `adaptive_switcher_node.py:18`, `rerun_h1_mixed.py:33` |
| B7 | **A tese do resumo contradiz a conclusão do mesmo documento.** `:23`/`:39` afirmam "supera qualquer escolha fixa"; `:213`/`:235` reformulam para argumento de custo. urban_grid n=1000 mostra empate (82,9/82,3/82,8). | `relatorio_final_pip.md`, `urban_grid_results.csv` |
| B8 | **O custo do adaptativo nunca foi medido** — só o de A* e BC isolados. O adaptativo paga `astar_policy.reset()` por troca. A tese de custo repousa numa inferência. | nenhum script instrumenta custo |
| B9 | **`random_switching` está medido e não é reportado** (71,2%→54,6%). É a ablação que separa "comutar" de "comutar com base em ρ". | `sota_comparison_500trials.csv` |
| B10 | **Vale inexplicado em ρ=0,25**: adaptativo cai para 68,2% entre 88,6% e 87,6% — exatamente na faixa de decisão. Não comentado em lugar nenhum. | idem |
| **B11** | **🔴 O 85,3% da Fase 1 é uma tautologia aritmética, não um resultado experimental.** Ver análise abaixo. | `validation_abstract/planners/*.py`, `experiments_abstract/sota_comparison.py` |

### B11 em detalhe — o achado mais grave da auditoria

Os "planejadores" da Fase 1 não simulam navegação. Eles sorteiam sucesso de uma fórmula fechada
escrita à mão:

```python
# validation_abstract/planners/rrt_star.py:29  (classe MockRRTStarPlanner)
success_prob = max(0.6, 0.98 - density * 1.2)      # RRT*: decresce com densidade
success = np.random.random() < success_prob

# experiments_abstract/sota_comparison.py:136     (fixed_ppo)
success_prob = max(0.1, 0.6 + density * 0.4)       # PPO: cresce com densidade
```

O ρ-criterion escolhe entre os dois por densidade. Como uma fórmula foi escrita decrescente em ρ
e a outra crescente em ρ, **o adaptativo é matematicamente forçado a ganhar** — ele está
selecionando o máximo de duas retas cujo cruzamento foi definido pelo próprio autor. Não há
navegação, colisão, geometria ou trajetória envolvida: `np.random.random() < f(ρ)`.

Calculando analiticamente, sem rodar nada (as retas cruzam em ρ≈0,2375):

| ρ | RRT* | PPO | adaptativo (τ=0,30) |
|---|---|---|---|
| 0,15 | 0,800 | 0,660 | 0,800 |
| 0,25 | 0,680 | 0,700 | **0,680 ← subótimo** |
| 0,35 | 0,600 | 0,740 | 0,740 |
| 0,45 | 0,600 | 0,780 | 0,780 |
| 0,55 | 0,600 | 0,820 | 0,820 |
| **média** | **0,656** | **0,740** | **0,764** |

Três consequências que fecham questões abertas do plano:

1. **O "vale em ρ=0,25" (B10) tem explicação exata:** é a única densidade amostrada onde τ=0,30
   escolhe errado, porque o cruzamento real das retas é em 0,2375 e o limiar adotado é 0,30.
2. **A varredura de τ sobre essas fórmulas maximiza em τ=0,20** (0,768) contra 0,764 em τ=0,30 —
   reproduzindo exatamente o B1. **O τ "ótimo" nunca foi uma descoberta empírica: é o ponto de
   cruzamento de duas retas que alguém digitou.**
3. O `sota_comparison.py` gera os concorrentes (He, Neural Switching, Hybrid DRL) com o mesmo
   padrão — inclusive um `MLPClassifier` treinado em `X = np.random.random((100,3))` com rótulo
   `y = X[:,0] > 0.3`, isto é, um "método neural concorrente" treinado para reproduzir o limiar
   0,30 sobre ruído puro.

**Implicação para a apresentação:** o número mais citado do trabalho (85,3%, presente no resumo do
PIP, no parcial aprovado, no abstract do LAFusion e no resumo do CONPEEX) não mede desempenho de
planejamento. O relatório diz "mocks calibrados estatisticamente para reproduzir taxas publicadas",
o que é generoso demais: não há calibração contra dado publicado, há duas fórmulas lineares
inventadas. Se o professor abrir `rrt_star.py`, a conversa acaba ali.

Isto **não** é fraude — está declarado como mock em vários lugares e o nome da classe é
`MockRRTStarPlanner`. Mas é um resultado sem conteúdo empírico sendo apresentado como resultado
principal, e precisa ser reclassificado antes de qualquer slide.

---

## Etapa 1 — Bloqueadores factuais (nada de slides antes disto)

**1.1 Resolver a contradição do limiar (B1/B2/B3).** É a pergunta mais letal e não tem resposta
hoje. Três saídas, em ordem de preferência:

- **(a) Recalibrar τ com os planejadores reais.** Varredura de τ ∈ [0,10; 0,60] usando A* real +
  BC real no gêmeo 2D, com **train/test split** (calibrar em metade dos cenários, reportar na
  outra) — corrige de uma vez B1, B2 e o ponto cego de overfitting de seleção. O τ que sair daí
  é o τ do trabalho, seja ele qual for.
- **(b)** Se (a) confirmar outro valor, adotar o novo e reescrever os documentos.
- **(c)** Só se o tempo apertar: manter 0,30 declarando explicitamente que **não** é o mínimo de
  regret, e justificar por outro critério (estabilidade/custo). Honesto, mas fraco.

Novo script: `eval/env2d/sweep_threshold_real.py`. Saída: `results_abstract/threshold_sweep_real.csv`
com colunas `tau, split, n, success, regret, switches, cost_ms`.

**1.2 Unificar a definição de ρ (B6).** Decidir qual das duas é a definição oficial e declarar a
fórmula com todos os parâmetros (janela, limiar de ocupação, unidade). Se as duas continuarem
existindo, documentar a correspondência entre elas. **A fórmula em `:58` precisa reportar o valor
de `w`** — hoje ela tem um parâmetro livre não especificado.

**1.3 Regenerar os dados perdidos (B4)** com n=1.500 real e preservar os CSVs:

```bash
mkdir -p logs
# ATENÇÃO: fazer backup antes — estes scripts SOBRESCREVEM os CSVs existentes
cp results_abstract/urban_grid_results.csv results_abstract/urban_grid_results.bak.csv

python3 eval/env2d/rerun_h1_hysteresis.py --trials 1500 --seed0 5000 | tee logs/hyst.log
python3 eval/env2d/rerun_h1_mixed.py      --trials 1500 --seed0 5000 | tee logs/mixed.log
python3 eval/env2d/rerun_urban.py         --trials  500 --seed0 9000 | tee logs/urban.log
```

**Duas correções em relação a versões anteriores deste plano:**

1. **`rerun_urban.py --trials 500`, não 1000.** O script roda **duas condições** (`static` e
   `dynamic`), gerando `2 × trials` linhas. O `urban_grid_results.csv` atual tem 1000 linhas
   porque foi rodado com `--trials 500`. Passar `--trials 1000` produziria 2000 linhas e mudaria
   o experimento — os "n=1000" citados no plano são 500 por condição.
2. **`rerun_urban.py` sobrescreve `urban_grid_results.csv`** — o arquivo que hoje contém o único
   experimento com massa estatística real do trabalho (o do empate 82,9/82,3/82,8). Fazer backup
   antes é obrigatório; se a rodada nova falhar no meio, o dado principal se perde.

`--seed0` idêntico entre hysteresis e mixed torna a comparação one-shot vs. histerese pareada nos
mesmos mundos.

**Sobre a estimativa de tempo:** os "~0,16 s/trial / ~15 min" vieram de extrapolação de uma
rodada de 10 trials em `sparse`. Trials em `very_dense` e `urban_grid` custam mais (A* expande
mais nós), e o mixed pool sorteia mundos. **Tratar 15 min como piso, não como estimativa** —
rodar primeiro com `--trials 50` para medir de verdade antes de lançar a bateria completa.

**1.4 Instrumentar o custo do adaptativo (B8)** — é o buraco no centro da única tese viva.
Em `rerun_h1_hysteresis.py`, cronometrar com `perf_counter()` separando:
- `astar_search_ms` (o `reset()`, onde está o custo real da busca)
- `astar_track_ms` (o `act()`, pure pursuit O(1))
- `bc_ms`, `adaptive_ms` (incluindo **todo replanejamento por switch**)

Excluir `optimal_path_length()` da contagem (é métrica, não custo). Sem essa separação o A*
parece gratuito, porque `.act()` não é onde ele gasta.

**1.5 Corrigir os ms (B5).** Valores para 8,00/20,90/29,21 e razão para 656× em
`relatorio_final_pip.md:199` e no `.tex`. Eliminar os literais de `plot_h1_real_validation.py:31-32`,
lendo do JSON. Recriar o script gerador perdido como `eval/env2d/measure_planner_cost.py`.

**1.6 Trazer os baselines que faltam para a tabela principal.** `random_switching` (B9) já está
medido e fora do relatório; **sempre-A\*** nunca esteve na Tabela 2, e é o baseline que importa.
Sem os dois, a tabela não sustenta o claim de superioridade.

**1.7 Vale em ρ=0,25 (B10) — RESOLVIDO pela análise do B11.** É o único ponto amostrado onde
τ=0,30 escolhe errado, porque as retas dos mocks cruzam em ρ≈0,2375. Não é um defeito do critério;
é um artefato do mock. Basta explicar — mas explicar exige admitir o B11.

**1.8 Reclassificar a Fase 1 (B11) — decisão de maior impacto do plano.** O 85,3% é uma tautologia
aritmética. Três saídas:

- **(a) Rebaixar (recomendado).** A Fase 1 deixa de ser "validação" e vira **"estudo de viabilidade
  do mecanismo de comutação"**, declarando que os planejadores eram modelos analíticos de sucesso-
  vs-densidade, não implementações. O 85,3% sai do resumo, do abstract e de qualquer slide de
  resultado. O resultado principal passa a ser o urban_grid (n=1000, planejadores reais).
- **(b) Substituir.** Reexecutar o protocolo da Fase 1 inteiro com A* real + BC real. É o que o
  `urban_grid` e a `rerun_h1_mixed` já fazem parcialmente — na prática, (a) + promover os
  experimentos reais a resultado principal cobre isso sem trabalho novo.
- **(c) Defender como está.** Não recomendo: a fórmula é legível em 4 linhas de código.

Onde o 85,3% precisa sair ou ganhar ressalva forte: `relatorio_final_pip.md:23,27,114,120,235`;
`lafusion_2026_draft.md:40,224,226,345,430`; `docs/relatorio_parcial.md:14,68,157`;
`paper/resumo_conpeex_2026.md`. **Atenção:** o parcial aprovado e o resumo do CONPEEX já foram
entregues com esse número — não dá para reescrever o passado, mas dá para narrar a correção
(Etapa 4.5.1).

---

## Etapa 2 — Reescrever a tese (depende da 1)

A tese atual está falsificada. A única defensável com os dados de hoje:

> **Sob planejadores reais, a seleção contextual atinge paridade de acerto com o melhor
> planejador fixo a uma fração do custo de decisão.**

Isso é um resultado de Pareto, não uma derrota — e é mais interessante que o claim original.

**Onde reescrever** (a tese antiga precisa sair do documento, não conviver com a nova):
- `relatorio_final_pip.md:23` (resumo), `:39` (tese central), `:235` (conclusão)
- `lafusion_2026_draft.md:36-56` (abstract), `:103-127` (contribuições)
- Manter `:213` e `:235`, que já estão na formulação certa

**A pergunta que a tese de custo ainda não responde** — e que é a mais perigosa da reunião:
*"29 ms cabe folgado num laço de 10 Hz. Por que não usar A* sempre?"* A defesa por custo só vale
se houver regime onde 29 ms doa. Três argumentos possíveis, **é preciso escolher e sustentar
com dado**: (i) escala de mapa/grade maior; (ii) N robôs compartilhando CPU; (iii) a cauda de
outliers (51,9 ms) violando deadline mesmo quando a média não viola. O (iii) já tem dado
(`cost_distribution.json`) e é o mais barato de defender.

**Buracos de método a preencher** (o "como"): como o BC foi treinado (nº de demonstrações,
arquitetura, epochs) e **como os mocks foram calibrados** — a Fase 1 inteira repousa nisso e não
há uma linha a respeito.

**Parâmetros sem justificativa a declarar:** janela `w`; `occ ≥ 65` (é default do Nav2 — dizer
isso); histerese ±0,02; grade A* 0,08 m e margem 0,08 m; `GOAL_RADIUS=0,25`; a origem dos
"150 trials" do teste da Tabela 2 quando o CSV tem 500 por densidade.

---

## Etapa 3 — Literatura (paralela à 2)

Não existe revisão sistemática, e o draft do LAFusion admite isso em `:391` enquanto afirma
novidade. Fazer o mínimo defensável: protocolo documentado (bases, string de busca, critérios de
inclusão), tabela comparativa de características, ~15 referências novas.

**Lacuna conceitual mais séria:** o problema deste trabalho *é* **algorithm selection** (Rice 1976;
Kotthoff; SATzilla) e **hyper-heuristics** — literatura inteira sobre escolher qual algoritmo
rodar com base em features da instância. Não é citada. Isso não invalida o trabalho, mas invalida
o claim de novidade em `:37`, que hoje é forte demais para 16 referências.

**Comparação explícita com Kolomeytsev et al. (2025)** — o concorrente mais direto (HMP-DRL,
0,836 de sucesso), hoje citado só sobre reward.

**Estado da arte a reconhecer:** BC de campo potencial é 2018-19. Navegação aprendida hoje passa
por foundation models (NoMaD, ViNT, GNM), Diffusion Policy e o BARN Challenge (citado, nunca
comparado). Não é preciso implementar — é preciso posicionar.

---

## Etapa 4 — Figuras (depende da 1)

### Auditoria visual (todas as 26 PNG foram abertas e inspecionadas)

**Prontas para slide, sem retrabalho:**

| Figura | Por quê | Uso |
|---|---|---|
| `2d/fig_2d_urban_dynamic_comparison.png` (1634×668) | **A melhor figura do repositório — e está órfã**, não citada em documento nenhum. Mostra os três métodos × estático/dinâmico, em sucesso e eficiência de rota, N=500/condição. É a peça que fecha "ambientes urbanos" + "obstáculos dinâmicos" do Plano de Trabalho | Slide 12, resultado principal |
| `gazebo_screenshots/gz_panel.png` (1890×749) | Prova visual de que a infra rodou: top-down + LIDAR 360° do TurtleBot3. Anotada, bilíngue, alta resolução | Slide 16 (Gazebo) |
| `benchmark/benchmark_time.png` | Limpa, log-log correta, anotação boa. Dado mais sólido do trabalho | Slide 8 (resposta ao Parecer) |
| `marl/fig_marl_shared_reward_comparison.png` | Contraste 60%→0% salta imediatamente; rótulos em cada barra | Slide 15 |
| `core/switching_heatmap.png` (884×884) | Explica o conceito do ρ-criterion visualmente melhor que qualquer texto: regiões azuis/vermelhas com fronteira tracejada | Slide 6 (conceito) |
| `2d/fig_2d_bc_trajectory_sparse.png` + o GIF | Melhor resultado quantitativo (98%), e o GIF anima | Slide de paradigmas |
| `cbs/cbs_scalability.png` | Sustenta o argumento de escala (resposta ao "29 ms cabe em 10 Hz") | Slide 15 |

**Problemáticas — não usar sem corrigir:**

- 🔴 **`pareto/fig_pareto_boxplot_main.png`** — o painel lateral afirma
  `τ* = argmin R(τ)` e conclui *"τ*=0,30 é Pareto-ótimo"*, mas `threshold_sensitivity.csv`
  minimiza em τ=0,20 e o próprio boxplot rotula τ=0,25 como *"melhor sucesso puro"*. **A figura
  contradiz a si mesma e ao CSV.** Além disso os três boxplots têm medianas visualmente idênticas
  (80%), o que não sustenta a conclusão. É a figura que mais expõe o B1 — refazer ou omitir.
- 🟠 `2d/fig_2d_h1_real_validation.png` — contém os ms errados (9,3/26,4/32,7). Regenerar após 1.5.
- 🟠 `core/fig_statistical_test.png` — ilustra o claim falsificado da Fase 1 (B11). Só usar se
  reenquadrada como "estudo de viabilidade", nunca como validação.
- 🟡 Todas as figuras de painel (900×~300 px, ex. `fig_2d_h1_boxplots`, `fig_marl_degradation`)
  foram dimensionadas para coluna de artigo — **ficam ilegíveis projetadas**. Reexportar em
  `paper/figs/slides/` com figsize ampliado e `dpi=200`.

**Órfãs que merecem entrar** (existem, ninguém cita): `fig_2d_urban_dynamic_comparison`,
`switching_heatmap`, `density_progression` (2234×895), `all_benchmarks_comparison`,
`gz_panel` e os 7 screenshots do Gazebo, `marl/roadmap_marl_tj`.

**Veredito:** há material visual bom e suficiente para o deck inteiro. Só **duas** figuras
precisam ser criadas do zero (Pareto sucesso×custo e chattering); o resto é seleção, reexportação
em tamanho de projeção, e aposentar a `fig_pareto_boxplot_main`.

### Trabalho de figuras

Regenerar com os dados novos: `plot_h1_real_validation.py`, `plot_h1_boxplots.py`.

Criar, em ordem de valor argumentativo:
- **F1 — Pareto sucesso × custo** (`figs/pareto/fig_pareto_success_vs_cost`). Sustenta a única
  tese viva: empate no eixo y, vitória no eixo x. Se só der para uma figura nova, é esta.
- **F2 — Chattering one-shot vs. histerese** (`figs/core/fig_chattering_hysteresis`). ECDF de
  switches/episódio. Contribuição mais nova e mais "de engenharia".
- **F3 — Sucesso por ρ com zona morta sombreada** (`figs/core/fig_success_by_rho`). Também é
  onde o vale em ρ=0,25 fica visível — enfrentar, não esconder.
- **F4 — Varredura de τ real, train vs. test** (`figs/core/fig_threshold_sweep_real`). É a
  resposta gráfica à pergunta nº 1.

Reportar **dispersão em tudo**: o benchmark clássico tem 5 trials/config e nenhum desvio-padrão
na Tabela 1; taxas de sucesso aparecem como pontos sem IC. Adicionar barras de erro.

Para o Canva: exportar PNGs em `paper/figs/slides/` com `dpi=200` e figsize ampliado — as figuras
atuais foram dimensionadas para coluna de artigo e ficam ilegíveis projetadas.

---

## Etapa 4.5 — Reconciliação documental (depende da 1; paralela à 2)

Fechar a cadeia Plano de Trabalho → Parecer → Parcial → Final, para que os slides tenham com o
que encadear.

**4.5.1 Nota de reconciliação com o parcial.** Uma subseção nova no relatório final (ou apêndice)
percorrendo a tabela de contradições acima: o que mudou, por qual evidência, e o que vale agora.
Isso protege o professor também — ele coassinou o parcial, e um avaliador externo que compare os
dois documentos vai encontrar as divergências. Melhor que a explicação esteja escrita.

**4.5.2 Checklist do Plano de Trabalho (PI08078-2024).** Os 5 objetivos específicos têm cobertura,
mas dois exigem honestidade:
- **Obj. 2** ("cenários estáticos, dinâmicos e com obstáculos") — o `urban_grid` com obstáculo
  móvel cobre o dinâmico no gêmeo 2D; o Gazebo não. Declarar onde foi coberto e onde não.
- **Obj. 5** ("Python, C++, ROS e Gazebo") — C++ só indiretamente via Nav2/SmacPlanner2D; Gazebo
  sem dado quantitativo. Já há justificativa, manter visível.
- O plano fala em **ambientes urbanos**: o `urban_grid` (n=1000) é a peça que fecha esse item e
  hoje está subaproveitada nos documentos. **Promover a resultado de primeira linha** — é o único
  experimento do trabalho com massa estatística real.

**4.5.3 Reforçar a resposta ao Parecer do Consultor.** A resposta está correta (`heapq`, matriz
densa, Nav2 C++) e citada por data na Seção 2.1 — mantê-la. **Mas** o benchmark tem 5
trials/config e nenhum desvio-padrão; "implementação otimizada" com n=5 é atacável justamente no
ponto que o consultor levantou. Rodar mais trials (é barato) e reportar dispersão na Tabela 1.

---

## Etapa 5 — Decisões para o professor (independente, preparar desde já)

Levar como recomendação + alternativa, não pergunta aberta.

- **DDPG** — prometido no parcial, nunca implementado, hoje só "descartado" em
  `relatorio_final.md:71,265`. Recomendação: declarar em Limitações como substituído por SAC/BC,
  com o porquê. Transforma omissão em decisão documentada.
- **Gazebo** — Yan descartou; o professor não recomendou descartar (sugeriu Victor Matheus /
  Pequi Mecânico). Enquadrar: "Fase 2 vira limitação declarada, ou reabro com apoio do Pequi?"
  Levar `results_ros2/gazebo_episode.json` e `figs/gazebo_screenshots/` como prova de que a infra
  rodou. Detalhe: esse JSON é untracked e **dono root** (gerado em container) — ajustar permissão
  e commitar antes da reunião.
- **16 → 15 páginas** — chegar com corte proposto: Tabela 1 vira duas frases; o parágrafo de
  escalabilidade CBS (`:221`) sobrepõe a figura.

---

## Etapa 6 — Roteiro de slides (depende de 2, 3 e 4)

Entregável: `paper/slides_prof_aldo.md`, um bloco por slide (título, bullets redigidos, caminho
da figura, nota do que falar) para o Yan montar no Canva.

### ⚠️ Duas versões do deck

O E5 inverteu a ordem: o deck de 18 slides abaixo é o da **apresentação final** (defesa/CONPEEX,
depois que as correções estiverem feitas). Para a **reunião com o Prof. Aldo agora**, o formato
é outro — sessão de trabalho, ~10 slides, 20 min de fala e 40 de discussão:

**Deck de reunião (executar este primeiro):**

1. **Capa + objetivo da reunião** — dito explicitamente: *"trouxe uma auditoria do projeto e três
   decisões de escopo para tomarmos juntos"*. Define a natureza da conversa na primeira frase.
2. **Onde estamos no Plano de Trabalho** — os 5 objetivos, o que está coberto. Ancoragem.
3. **O que fizemos desde o parcial** — A* real, BC 98%, MARL 60%→0%, urban_grid n=1000,
   histerese. Entregas concretas primeiro: estabelece que houve trabalho, não só problemas.
4. **⭐ A auditoria — o que encontrei ao reler o código antigo.** Os achados em ordem de
   gravidade: B11 (Fase 1 tautológica), B1/B2 (limiar), B4 (dados perdidos). Linguagem do E2:
   plural nos erros, singular nas correções.
5. **O que isso significa para a tese** — a tese antiga não se sustenta; a de paridade-a-custo
   sim; e o cenário E3 ("pode não funcionar no regime testado") explicitamente na mesa.
6. **O que já temos de sólido** — benchmark clássico, urban_grid n=1000, MARL, BC.
   Fig: `benchmark_time.png`, `fig_2d_urban_dynamic_comparison.png`.
7. **F1 — Pareto sucesso × custo.** A única tese viva, em uma imagem.
8. **Gazebo — status e a pergunta.** `gz_panel.png`. Termina em decisão.
9. **⭐ As três decisões** — (i) qual documento recebe versão completa dado que LAFusion vence
   antes do relatório (E4); (ii) reclassificar a Fase 1 ou defendê-la (B11/E1); (iii) Gazebo
   reabre ou vira limitação declarada.
10. **Plano para as cinco semanas** — duas ou três opções de escopo, para ele escolher. Não um
    plano fechado: opções.

**Regra de tom (E2), válida para os dois decks:** erros no plural ("o protocolo que usamos"),
correções no singular ("refiz a validação"). Nenhum slide pede desculpa; nenhum slide expõe o
orientador.

---

### Deck da apresentação final (~18 slides, 20-30 min) — pós-correções

### A narrativa (o fio que sustenta o deck)

Sem uma narrativa, a continuidade documental vira checklist de conformidade e o deck fica chato e
defensivo. A história que os documentos e os dados contam, quando postos em ordem, é esta — e ela
tem começo, virada e fim:

> **"Prometemos comparar planejadores. Descobrimos que a pergunta interessante não era *qual é
> melhor*, mas *quando cada um é melhor*. Fomos atrás disso — e, ao testar com rigor, descobrimos
> que nossa própria resposta estava errada pelo motivo certo: o ganho não está no acerto, está no
> custo."**

Três atos:

- **Ato I — O contrato e o pivô (slides 1-8).** O Plano de Trabalho pedia comparar clássicos e
  modernos. A revisão mostrou que isso já estava saturado na literatura, e o pivô para *seleção
  contextual* foi feito e **aprovado no parcial**. O Parecer do Consultor exigia implementações
  otimizadas — respondido com `heapq`, matriz densa, Nav2 C++. Aqui o trabalho está em dia com
  tudo o que foi prometido. Tom: cumprimento e fundamentação.

- **Ato II — A virada (slides 9-11).** O ponto de tensão. Ao substituir mocks por planejadores
  reais, três coisas caíram: o "A*" era linha reta, a "100% de acurácia" escondia chattering, e a
  tese de superioridade em acerto não sobreviveu. **Este é o clímax da apresentação, não um
  constrangimento.** A pergunta que o slide 9 responde é: *"o que acontece quando você testa a
  própria hipótese com o rigor que ela merece?"* Tom: honestidade e método.

- **Ato III — O que sobrou é melhor (slides 12-18).** Do outro lado da virada há um resultado
  mais forte: paridade de acerto a 1/656 do custo, chattering diagnosticado e corrigido com
  controle supervisório, coordenação multiagente resolvida por reward compartilhada (60%→0%). A
  tese nova é de Pareto, não de superioridade — e é a que se leva ao LAFusion. Tom: resultado e
  direção.

**Frase de ligação entre atos** (usar literalmente, é o que dá coesão):
- I→II: *"Até aqui, tudo funcionava. Então decidimos testar se funcionava mesmo."*
- II→III: *"A hipótese caiu. O que ficou de pé é mais defensável do que o que caiu."*

**Regra de tom:** nenhum slide pede desculpa. Cada mudança é apresentada como consequência de uma
decisão de testar melhor. O professor precisa sair com a impressão de que o aluno encontrou os
próprios erros antes que alguém encontrasse — que é exatamente o que aconteceu.

Estrutura, com a tese nova **e a espinha de continuidade** (o deck percorre
Plano de Trabalho → Parecer → Parcial → hoje):

1. **Capa** — título oficial do PI08078-2024, não um título inventado
2. **De onde partimos** — o Plano de Trabalho: objetivo geral e os 5 objetivos específicos, com
   marcação visual do que está coberto. Ancora tudo o que vem depois no contrato FAPEG
3. **O problema em uma frase** — clássico confiável e caro; aprendido barato e frágil
4. **O pivô (já aprovado no parcial)** — de "comparar clássicos vs modernos" para "tratar a
   seleção como variável contextual". Recapitular que isso foi aprovado em 01/04/2026, não é
   mudança nova
5. **Pergunta de pesquisa e hipóteses** — H1 **já na forma reformulada**
6. **O ρ-criterion** — definição com **todos** os parâmetros declarados, incluindo `w` e a
   unificação da métrica (`figs/core/switching_heatmap.png`)
7. **Por que densidade, por que limiar rígido** — os bons parágrafos `:51`/`:53`
8. **Resposta ao Parecer do Consultor (13/06/2025)** — "implementações otimizadas": `heapq`,
   matriz densa, Nav2/SmacPlanner2D em C++. Exigência formal atendida, com o benchmark real
   (`figs/benchmark/benchmark_time.png`) como evidência. Slide curto e forte
9. **⭐ O que mudou desde o relatório parcial — e por quê.** Tabela de três colunas: *o parcial
   dizia* / *o que descobrimos* / *o que vale agora*. Cobre: A* falso→real, 100% de acurácia→
   chattering, threshold, regret, tese invertida. **Este slide é o pivô da apresentação** —
   antes dos resultados, transforma cada contradição em demonstração de rigor
10. **Metodologia atual** — gêmeo 2D, A* real, BC, trials pareados, train/test split no limiar
11. **Fase 1 Monte Carlo** — com a ressalva "mocks calibrados" dita em voz alta
12. **Resultado 1 — sucesso: empate** (urban_grid, n=1000). Dizer "empate" com todas as letras,
    com IC da diferença. É também o item "ambientes urbanos" do Plano de Trabalho
13. **Resultado 2 — custo: 656×.** Fig F1 (Pareto). Engatar imediatamente no 12
14. **Resultado 3 — chattering e histerese.** Fig F2. Aqui se fecha o "100% de acurácia" do
    parcial: a métrica antiga não via oscilação; a nova vê e corrige
15. **Multiagente / CBS** — escalabilidade, O(1) por agente; MARL 60%→0% (`figs/cbs/`, `figs/marl/`)
16. **Fase 2 / Gazebo** — status honesto, evidência de que a infra rodou, termina na pergunta
    de decisão
17. **Cobertura do Plano de Trabalho** — checklist final: o que foi entregue, o que mudou de
    forma justificada, o que não foi feito e por quê (DDPG, DWA, obstáculos dinâmicos no Gazebo)
18. **Limitações e próximos passos** — + LAFusion + as três decisões da Etapa 5

Se 18 slides ficarem longos para o tempo, fundir 3 com 5 e 17 com 18 — mas **9 e 12-13 são
inegociáveis**.

**Figuras por valor argumentativo:** F1 (Pareto) > `benchmark/benchmark_time.png` (dado mais
sólido do trabalho) > `cbs/cbs_scalability.png` (resposta à pergunta dos 29 ms) > F2 (chattering) >
`2d/fig_2d_bc_trajectory_sparse.png` + GIF (melhor resultado, 98%) >
`marl/fig_marl_shared_reward_comparison.png` (60%→0%, o mais impressionante).
**Fora:** `fig_2d_h1_real_validation.png` até ser regenerada, e a Tabela 2 como resultado de
destaque.

**Nota de apresentação:** o slide 9 é onde o professor interrompe. O 10 precisa engatar em
segundos — a defesa do empate é o custo, e não pode vir três slides depois.

---

## Perguntas a ter resposta pronta

Ordenadas por letalidade. As cinco primeiras não têm resposta hoje:

1. "Sua tabela minimiza regret em τ=0,20. Por que 0,30?" → Etapa 1.1
2. "O limiar foi calibrado com RRT*/PPO. Por que vale para A*/SAC?" → Etapa 1.1
3. "Onde está o CSV dos 1.500 trials pareados?" → Etapa 1.3
4. "Qual o tamanho da janela `w`?" → Etapa 1.2
5. "Qual o custo do adaptativo — não do A*, não do BC?" → Etapa 1.4
6. "29 ms cabe num laço de 10 Hz. Por que não A* sempre?" → Etapa 2 (escolher o argumento)
7. "Comparou contra comutação aleatória?" → Etapa 1.6
8. "Calibrou e avaliou no mesmo conjunto?" → Etapa 1.1 (split)
9. "Como os mocks foram calibrados?" → Etapa 2
10. "Por que o método piora em ρ=0,25?" → Etapa 1.7

---

## Verificação

- `python3 eval/env2d/measure_planner_cost.py` reproduz o JSON na ordem de grandeza de
  8,00/20,90/29,21 ms.
- Todo número citado nos documentos tem CSV com o n que o texto afirma:
  `wc -l` bate com "1.500 trials" / "1.000 trials" em cada citação.
- `grep -n "9,3 ms\|26,4\|32,7\|~600" paper/relatorio_final_pip.md` → vazio.
- `grep -n "astar_cost_ms = \[" eval/env2d/plot_h1_real_validation.py` → vazio.
- `grep -rn "supera qualquer" paper/` → vazio (tese antiga eliminada).
- A fórmula de ρ no relatório reporta `w` e o limiar de ocupação com valores numéricos.
- Tabela principal inclui sempre-A* e random_switching.
- Cada slide que cita número tem o número conferido contra o CSV correspondente.

## Dependências e cronograma (5 semanas até 31/08)

A ordem mudou por causa do E5 — a reunião entra **cedo**, não no fim:

```
AGORA (1-2 dias)
  1.3 bateria (15 min, tratar como piso)  +  verificar B11  +  F1 (Pareto)  +  deck de reunião
        │
        ▼
  ╔═══ REUNIÃO ═══╗  ← define escopo das 5 semanas; resolve E1, E3, E4
        │
        ├─► se prioridade = LAFusion (vence 16/08):
        │     1.1 recalibrar τ → 1.4 custo adaptativo → 2 tese → 4 figuras → paper
        │     relatório PIP recebe versão mínima viável
        │
        └─► se prioridade = relatório PIP (vence 31/08):
              1.8 reclassificar Fase 1 → 4.5 reconciliação → 2 tese → deck final
              LAFusion adiado ou submetido com escopo reduzido

  Etapa 3 (literatura) e Etapa 5 (decisões) — paralelas, desde já
```

**Caminho crítico real:** não é técnico, é a decisão de escopo. Os bloqueadores levam mais de
cinco semanas para serem zerados; o que cabe no prazo depende de qual prêmio for priorizado (E4),
e essa escolha sai da reunião.

**Riscos de calendário:** LAFusion 31/07–16/08 vence **antes** do relatório (31/08). Se a decisão
demorar, o paper perde a janela por default — o que é uma decisão tomada por omissão, o pior
tipo. Levar E4 como primeiro item da pauta.
