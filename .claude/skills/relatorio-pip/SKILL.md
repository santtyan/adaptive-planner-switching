---
name: relatorio-pip
description: Regras oficiais do PIP/UFG para o relatório final de Iniciação Científica (formato, estrutura, limites de página/caracteres/PDF) mais o estado real e as armadilhas específicas deste repositório (paper/relatorio_final_pip.md e .tex, tese sobre custo computacional, ρ-criterion A*/BC). Use sempre que o usuário pedir para escrever, revisar, editar, cortar, sincronizar ou submeter o relatório final, o resumo/abstract, ou mencionar SIGAA, PIP/UFG, prazo de 31/08, certificados Diálogos, ou o artigo LAFusion. Use também antes de qualquer edição em paper/relatorio_final_pip.md ou .tex, mesmo sem o usuário citar o nome do documento.
---

# Relatório Final PIP/UFG

Este projeto é uma Iniciação Científica (PIBIC/FAPEG PI08078-2024, EMC/UFG). O relatório final
vai para o SIGAA com prazo **31/08/2026**. Esta skill reúne as regras oficiais do programa e o
que já deu errado neste repositório especificamente, para não repetir.

## Regras oficiais do PIP/UFG (não negociáveis)

- **Formato**: A4, fonte Arial 12pt, espaçamento 1,5.
- **Páginas**: máximo **15**, mas é **referencial** — "admite-se excedê-lo, caso isso seja
  estritamente necessário". Não é reprovação automática. Informações Complementares **não
  contam** para esse limite.
- **Resumo**: até **2.500 caracteres com espaço**. Vai também para os Anais do Seminário PIP —
  não é só uma seção do relatório, é um documento próprio com vida fora dele.
- **Estrutura obrigatória, nesta ordem**: Título (igual ao Plano de Trabalho; se mudou, nota de
  rodapé explicando), Autores (nome completo + sobrescrito numerado + função/unidade/email
  abaixo), Resumo, Apresentação (introdução/justificativa/objetivos), Metodologia (sintética),
  Resultados e Discussão (tabelas/figuras), Conclusão/Considerações Finais, Referências
  Bibliográficas (normas ABNT, admite variação por área).
- **Informações Complementares** (não contam página):
  - **Certificados de participação no Programa Diálogos em Pesquisa e Inovação — OBRIGATÓRIO.**
    Falta disso é risco real de devolução pelo SIGAA.
  - Elemento audiovisual (opcional): 1 arquivo, ≤50 MB, vídeo ≤1 min, link do Google Drive
    institucional compartilhado por ≥1 ano.
  - Outras atividades (opcional): certificados de eventos/cursos relacionados.
- **Entrega**: PDF único, **máximo 2 MB**, para upload no SIGAA.

## O que é específico deste repositório

### `.md` é a fonte da verdade; `.tex` é o que vira PDF

`paper/relatorio_final_pip.md` e `paper/relatorio_final_pip.tex` **não se sincronizam
sozinhos**. Já aconteceu de uma correção de tese entrar só no `.md` (commit `c7c5917`) e o `.tex`
ficar desatualizado por um mês, quase sendo submetido ao SIGAA com uma tese que os próprios
dados do projeto refutam (corrigido em `599098b`, ver `DEVELOPMENT_LOG.md` e a memória de sessão
28/08/2026 no auto-memory do projeto).

**Regra: qualquer correção de conteúdo (número, tese, seção) precisa ser aplicada nos dois
arquivos.** Antes de considerar uma correção terminada, `grep` a mesma string/número nos dois:

```bash
grep -n "<trecho que mudou>" paper/relatorio_final_pip.md paper/relatorio_final_pip.tex
```

E a verificação final (compilar, contar páginas, checar peso do PDF) tem que ser feita no
`.tex`/PDF, nunca só no `.md` — é o `.tex` que o SIGAA recebe.

### A tese atual é sobre custo, não sobre acerto

A formulação antiga ("o ρ-criterion supera qualquer método fixo em taxa de acerto") foi
**refutada** com planejadores reais (A* vence em acerto: 88,2% vs. 84,3%, McNemar
p=5,4×10⁻⁵, n=1.500, `results_abstract/h1_real_2d_mixed_pool.csv`). A tese correta é que o
critério **mantém desempenho próximo ao melhor planejador fixo a uma fração do seu custo
computacional** (~656× mais barato o BC em alta densidade). Qualquer texto que reapareça com
"supera qualquer método fixo em acerto" como conclusão (não como hipótese histórica citada e
refutada) é regressão — sinalizar e corrigir.

O critério escolhe entre **A\*** (clássico) e **BC** (Behavior Cloning, supervisionado). **SAC
não faz parte do critério** — é RL, existe no projeto por exigir o objetivo 3 do plano de
trabalho, mas perde para A*/BC em todo regime testado. Texto que descreve o critério como
selecionando SAC é o mesmo erro que já apareceu no `.tex` desatualizado.

### Nunca copiar número entre documentos sem recalcular

Dois números diferentes já coexistiram descrevendo a mesma medição (ex.: razão de custo
"~600×" no `.tex` velho vs. "~656×" no `.md` corrigido) porque um documento copiou do outro em
vez de recalcular do CSV. Ao citar qualquer estatística no relatório, recalcular direto do CSV
fonte, não copiar de outra seção ou do slide:

```python
import pandas as pd
from scipy.stats import binomtest
d = pd.read_csv('results_abstract/h1_real_2d_mixed_pool.csv')
a, ad = d['astar'], d['adaptive']
b1 = ((a==1)&(ad==0)).sum(); b2 = ((a==0)&(ad==1)).sum()
print(f"A* {a.mean()*100:.1f}% | adaptive {ad.mean()*100:.1f}%")
print(f"McNemar p = {binomtest(b1, b1+b2, 0.5).pvalue:.3g}")
```

Ver `paper/figs/CATALOG.md` para o mapa figura → script gerador → documento que a usa.

## Checklist antes de qualquer edição no relatório

Rodar o verificador de formato primeiro, para saber o estado real antes de decidir o que cortar
ou revisar:

```bash
python3 .claude/skills/relatorio-pip/scripts/check_formato.py
```

Ele reporta: caracteres do resumo vs. limite de 2.500, tamanho do PDF vs. limite de 2 MB,
páginas vs. o referencial de 15, se o `.md` foi tocado depois do `.tex` (sinal de possível
dessincronização) e se os certificados ainda são placeholder.

## Padrão-ouro de escrita científica (aplicado ao caso deste projeto)

Resumo de literatura sobre estrutura IMRaD, hipóteses falseáveis e reprodutibilidade (checklist
NeurIPS), traduzido em regras acionáveis para este relatório especificamente.

### Resultados não interpreta; Discussão/Conclusão interpreta
O erro mais comum em relatórios de IC é misturar as duas coisas. Na Seção "Resultados e
Discussão" deste relatório, os números devem aparecer primeiro (o que foi medido), a
interpretação depois (o que isso significa para a tese) — não intercalados a ponto de a leitura
não conseguir separar dado de argumento.

### Hipótese refutada é achado científico, não fracasso a esconder
A transição "H1 supera em acerto" → "H1 refutada em acerto, confirmada em custo" **já está bem
feita** neste relatório: reporta a hipótese original, o resultado que a refuta, e a hipótese
revisada, com o mesmo rigor estatístico (McNemar, n=1.500) nas duas. Preservar essa estrutura em
qualquer revisão — não suavizar a refutação nem reescrever como se a tese de custo sempre tivesse
sido a única hipótese (isso seria HARKing: apresentar uma hipótese pós-hoc como se fosse a
priori). O texto já marca isso explicitamente ("hipótese inicial... refutada ao testar com
planejadores reais") — manter essa marcação em qualquer corte do resumo.

### McNemar é o teste certo aqui, mas p-valor sozinho não basta
A comparação A*/BC/adaptativo é pareada e binária (sucesso/fracasso, mesmos cenários) — McNemar é
apropriado, já em uso. Com n=1.500, diferenças estatisticamente significativas podem ser
pequenas na prática: o relatório já reporta o efeito em unidade natural primeiro (88,2% vs.
84,3%, um gap de 3,9 pontos percentuais) antes do p-valor — manter essa ordem em qualquer edição,
nunca liderar com o p-valor sozinho.

### Resumo: o que cortar primeiro sob o limite de 2.500 caracteres
Ordem de prioridade ao cortar (do que corta primeiro para o que nunca corta):
1. Digressões metodológicas ("um resultado mais modesto do que...") — primeira candidata.
2. Contexto genérico da introdução (frases sobre robôs autônomos em geral).
3. **Nunca cortar**: o número do resultado principal, a tese revisada, ou a menção de que a
   hipótese original foi testada e refutada — é aí que mora o rigor científico do documento.

### Toda claim tem que bater com o experimento que a testou
Antes de considerar uma edição terminada, perguntar: essa frase afirma algo que o CSV/script
citado realmente mostra? Isso é o item mais comum do checklist de reprodutibilidade do NeurIPS
("claims match results") e é exatamente a categoria do incidente do `.tex` desatualizado.
