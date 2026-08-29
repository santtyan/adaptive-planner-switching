---
name: revisao-critica-relatorio
description: Revisão crítica linha-por-linha de relatórios científicos e artigos (rigor de lógico, gramático sênior e revisor de conferência de computação): decisões arbitrárias sem justificativa, afirmações sem fonte, lacunas de reprodutibilidade, perguntas que uma banca faria e o texto não responde, inconsistências internas, coesão em português, e idiossincrasias de texto gerado por IA (travessão, antítese "X, não Y", tríades compulsivas, adjetivação vazia). Use sempre que o usuário pedir para revisar criticamente, auditar, "achar furos", preparar para banca/prêmio, ou perguntar se a tese/argumentação/narrativa está forte, bem justificada ou pronta para submissão. Use também antes de considerar um relatório final ou artigo pronto para entrega.
---

# Revisão crítica de relatório científico

Esta skill audita um documento já escrito (não escreve do zero) com o padrão de uma banca de
prêmio: adversarial, exaustiva, sem elogiar. O objetivo é achar toda pergunta que alguém pode
fazer e o autor não sabe responder, antes que essa pergunta apareça numa arguição de verdade.

## Passo 1 — rodar o detector determinístico primeiro

```bash
python3 .claude/skills/revisao-critica-relatorio/scripts/check_ai_tics.py <arquivo.tex>
```

Reporta em segundos, sem gastar contexto de LLM: contagem de travessões, padrão antitético
"X, não Y", adjetivos sem número ao lado, negrito abrindo parágrafo, meta-comentário sobre o
próprio processo de correção, referências bibliográficas órfãs nos dois sentidos (citada no
corpo mas sem `\bibitem`, ou `\bibitem` nunca citado), e se hardware está declarado quando o
texto menciona benchmark de tempo/CPU. É um detector, não veredito: cada ocorrência pode ser
legítima, mas toda ocorrência merece uma segunda olhada.

## Passo 2 — releitura integral linha por linha

O script não substitui leitura. Ler o documento inteiro (não amostrar seções) categorizando
achados em:

1. **Decisões arbitrárias sem justificativa.** Todo número, limiar ou parâmetro precisa de uma
   frase "por quê" a menos de 2-3 linhas de distância. `occ ≥ 65` sem dizer que é o default do
   Nav2, ou `N=1500 trials` sem dizer por que 1500 e não 500 ou 3000, são exatamente o tipo de
   coisa que um avaliador pergunta e o autor esquece de ter respondido no texto.
2. **Afirmações sem fonte.** Qualquer frase sobre "a literatura" ou "o estado da arte" precisa
   apontar citação específica. "A literatura trata, em sua maioria, X" sustentado por 3
   referências é o tipo de generalização que convida à pergunta "revisão sistemática ou
   impressão?".
3. **Lacunas no "como".** Hardware (CPU/RAM: sem isso, todo benchmark de tempo é inválido),
   versões de software, seeds, critério de parada, definição operacional de métricas ("sucesso"
   significa o quê exatamente?). Sem esses detalhes o trabalho não é reproduzível.
4. **Perguntas adversariais.** Para cada seção, perguntar: que pergunta um professor cético faria
   aqui, e o texto responde? Ser especificamente adversarial com o achado mais forte do trabalho
   — é ali que mora o ataque mais provável.
5. **Inconsistências internas.** Números que mudam entre seções, referências cruzadas para
   conteúdo que não existe ou diz o oposto, resumo que afirma algo que o corpo depois retrata.
6. **Coesão e fluidez em português.** Anglicismos sem tradução nem itálico consistente (trials,
   baseline, goal, stack, mock, timeout — escolher tradução OU itálico sistemático, não misturar),
   repetição excessiva da mesma expressão, períodos longos demais para leitura corrida,
   personificação vaga ("os cenários revelaram").

## Passo 2b — padrão de gramático sênior da língua portuguesa

Além de coesão/fluidez (item 6 do Passo 2), aplicar rigor de revisor profissional de português,
frase por frase:

- **Crase**: toda ocorrência de "a" antes de palavra feminina checada por substituição (trocar
  por masculino: "ao"/"a"?) e por regência do verbo/preposição anterior ("referente à",
  "em relação à", "devido à" pedem crase; "a partir de" nunca leva).
- **Regência verbal e nominal**: verbos como "implica" (transitivo direto, sem "em"), "assistir"
  (sentido de ver = "assistir a"), "visar" (objetivo = "visar a"), preposição exigida por
  adjetivos ("propenso a", "favorável a") checada uma a uma.
- **Concordância verbal e nominal em frases longas**: sujeito composto ou intercalado por
  aposto/parentético é o ponto onde concordância mais quebra ("as taxas de sucesso, medidas em
  três densidades, revelam" — verbo concorda com "taxas", não com o substantivo mais próximo).
- **Paralelismo sintático em enumerações**: itens de uma lista ou oração com "e"/"ou" devem ter
  a mesma estrutura gramatical (todos substantivos, todos infinitivos, todos adjetivos) — não
  misturar "implementar X, avaliação de Y e testar Z".
- **Colocação pronominal**: próclise após palavra atrativa (não, que, quando, quem), ênclise em
  início de oração; evitar mesóclise (não cabe em prosa científica) e próclise incorreta após
  pausa forte (ponto, ponto e vírgula).
- **Pontuação de orações intercaladas e explicativas**: aposto e oração explicativa entre
  vírgulas (nunca apenas uma vírgula de abertura sem a de fechamento); ponto e vírgula para
  separar orações longas já pontuadas internamente com vírgula, não vírgula simples.
- **Ambiguidade referencial**: pronome ou elipse cujo antecedente pode ser mais de um substantivo
  na frase anterior ("o critério supera o método fixo, que não generaliza" — "que" refere-se a
  qual dos dois?).
- **Registro formal consistente**: nada de coloquialismo, gíria técnica não glossada na primeira
  ocorrência, ou mistura de registro entre seções (uma seção em tom de manual, outra em tom de
  ensaio).

O padrão de aceitação é o de um gramático profissional revisando um texto para publicação, não
"compreensível" ou "sem erro grosseiro": toda frase deveria sobreviver à leitura de um revisor de
prova de editora acadêmica sem marcação.

## Passo 3 — idiossincrasias de texto gerado por IA (além do que o script pega)

- **Antítese "X, não Y" em excesso** (mais de 5-6 no documento): é a assinatura mais delatora de
  texto passado por LLM sem reescrita humana. Cada ocorrência isolada é aceitável; a repetição é
  o problema.
- **Tríades compulsivas**: enumerar sempre em grupos de três ("três fases", "três hipóteses",
  "três grandezas") mesmo quando o número real de itens é outro, ou quando a lista poderia ser
  maior/menor sem perda.
- **Negrito abrindo parágrafo como bullet disfarçado**: `\textbf{Frase de efeito.} Resto do
  parágrafo...` repetido 5+ vezes é estrutura de slide, não de prosa científica corrida.
- **Meta-comentário sobre o próprio processo de edição**: "nesta correção", "na versão revisada",
  "foi retirado", "ao reconferir" — um relatório final apresenta a versão final, não narra a
  história de suas próprias reescritas. Isso sinaliza ao leitor que o documento foi remendado às
  pressas.
- **Hedging excessivo / auto-desculpa**: "resultado mais modesto", "tratado aqui como preliminar
  e não confirmado... permanece válida independentemente desta ressalva" — defender-se antes de
  ser atacado, repetidamente, é tão ruim quanto não se defender.

## Passo 4 — checar se o corte de página não cortou substância

Armadilha específica deste tipo de projeto: comprimir um documento para caber num limite de
página é fácil de fazer cortando prosa redundante, mas fácil de errar cortando substância junto
(estudo de ablação, seção metodológica inteira, nota que contextualiza uma tabela). Antes de
aceitar uma versão comprimida como final:

```bash
wc -w <arquivo_fonte_completo.md> <arquivo_comprimido.tex>
```

Se o comprimido tem menos da metade das palavras do fonte, investigar especificamente: existe
ablação/baseline que sumiu? Existe seção metodológica que só existe na versão longa? Existe nota
de honestidade/limitação que contextualizava um resultado e desapareceu, deixando a claim nua e
mais forte do que os dados sustentam?

## Padrão-ouro de referência (não reinventar métrica que já existe)

Para trabalhos de seleção entre planejadores/algoritmos por característica do problema, o campo
tem vocabulário consolidado desde Rice (1976), *algorithm selection*: **SBS** (Single Best
Solver, o melhor método fixo) e **VBS** (Virtual Best Solver, o oráculo que escolhe o melhor
método por instância). Reportar distância ao VBS e ganho sobre o SBS é mais reconhecido
internacionalmente do que inventar uma métrica de "regret" sem ancorar na literatura — e
geralmente é computável dos mesmos dados pareados que já existem, sem rodar experimento novo.

Em navegação/robótica, os benchmarks canônicos que um revisor espera ver referenciados ou
comparados são BARN (obstáculos densos, mesma faixa de dificuldade deste tipo de trabalho) e,
para métrica de eficiência de caminho, SPL (Success weighted by Path Length) ou SCT (Success
weighted by Completion Time, melhor ajuste quando a tese é sobre custo/tempo).

## Checklist final antes de dizer que está pronto

- [ ] Todo parâmetro numérico tem justificativa a até 2-3 linhas de distância
- [ ] Toda claim sobre literatura tem citação específica, não "a literatura, em sua maioria"
- [ ] Hardware declarado se há qualquer benchmark de tempo
- [ ] Toda referência bibliográfica é citada no corpo; toda citação existe na bibliografia
- [ ] Resumo e Conclusão não afirmam nada que o corpo do texto contradiz ou já retratou
- [ ] Objetivos declarados na Introdução são os mesmos do Plano de Trabalho oficial, não uma
      reformulação que promete algo (ex. "prova teórica") nunca entregue no corpo
- [ ] Zero travessão em prosa (regra deste projeto); antítese "X, não Y" usada com moderação
- [ ] O achado mais forte do trabalho resiste à pergunta adversarial mais óbvia sobre ele
