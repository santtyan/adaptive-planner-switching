# Resumo CONPEEX 2026 — pronto para colar no PLATEIA
## Edital PROEC Nº 08/2026 | Prazo: 26/06/2026 às 17h | https://23conpeex.plateia.ufg.br/

---

## CHECKLIST DE SUBMISSÃO

- [ ] Criar conta / login no PLATEIA (https://23conpeex.plateia.ufg.br/)
- [ ] Clicar em "Realizar Inscrição" → modalidade **Seminário de Iniciação à Pesquisa – PIP – Iniciação Científica**
- [ ] Preencher Título (copiar do bloco abaixo — CAIXA ALTA, ≤150 chars)
- [ ] Colar o texto do Resumo no campo "Resumo" (1.500–2.500 chars — contar no PLATEIA)
- [ ] Adicionar palavras-chave individualmente (Enter após cada uma)
- [ ] Área de conhecimento: **Ciências Exatas, da Terra e Engenharias**
- [ ] Coautor: Prof. Dr. Aldo André Diaz Salazar (campo orientador)
- [ ] Marcar se deseja concorrer ao Prêmio de Melhores Trabalhos
- [ ] Confirmar submissão e guardar o comprovante

---

## TÍTULO (colar em CAIXA ALTA)

MÉTODOS MODERNOS E CLÁSSICOS PARA PLANEJAMENTO DE TRAJETÓRIA ADAPTATIVO EM NAVEGAÇÃO AUTÔNOMA

---

## RESUMO (corpo — colar no campo "Resumo" do PLATEIA)

*Texto corrido, alinhamento justificado. Contar os chars no PLATEIA antes de submeter. Estimativa: ~2.367 chars.*

---

A tese deste trabalho é que a seleção adaptativa de planejador de trajetória, baseada na densidade local de obstáculos, supera métodos de planejamento fixos em taxa de sucesso em ambientes com densidade variável. Algoritmos determinísticos como Dijkstra e A* oferecem garantias de otimalidade e baixo custo computacional, mas tendem a apresentar desempenho limitado em ambientes densos e não estruturados; políticas de aprendizado por reforço, como o SAC, generalizam melhor nesses contextos, porém introduzem custo desnecessário em espaços livres. Portanto, a seleção do planejador deve ser tratada como uma decisão contextual adaptativa, não como uma escolha estática de projeto. Para verificar essa tese, propõe-se o critério ρ, que calcula a densidade local de obstáculos em torno da posição do robô e define a regra de seleção: A* para ambientes esparsos (ρ < 0,30) e SAC para ambientes densos (ρ ≥ 0,30), com limiar ρ* = 0,30 determinado por análise de sensibilidade sobre taxa de sucesso. A metodologia envolveu a implementação e avaliação de quatro algoritmos clássicos — Dijkstra, A*, Floyd-Warshall e Johnson — em quatro grades (100 a 2.500 nós), com medição de tempo de execução e consumo de memória. Para a validação em ambiente simulado, desenvolveu-se a infraestrutura de integração com ROS2 Humble, Gazebo Classic e TurtleBot3 Waffle via Stable-Baselines3, incluindo o treinamento do agente SAC. Os resultados mostram que Dijkstra atinge 0,08 ms e 4 KB em grades de 100 nós e 2,4 ms e 85 KB em grades de 2.500 nós; A* apresenta tempo similar e maior uso de memória (6,6 KB e 220 KB, respectivamente). Floyd-Warshall consome 39 segundos e 22 MB em uma grade de 900 nós, resultado de crescimento cúbico que inviabiliza sua aplicação em robótica móvel e justifica a escolha de A* como planejador clássico do sistema. A validação do critério em experimentos Monte Carlo, cobrindo condições de densidade esparsa, moderada e densa, demonstrou taxa de sucesso média de 85,3%, superior a todos os métodos de referência testados — o melhor planejador fixo atingiu 76% e o melhor método de comutação concorrente, 78,7% — com regret de 2,9% ante o seletor ideal. Os resultados parciais indicam que o critério ρ captura a fronteira de decisão entre planejadores clássicos e de aprendizado por reforço, com potencial de generalização a cenários multi-agente; a validação com planejadores reais em Gazebo está em andamento. Implementação disponível em github.com/santtyan/adaptive-planner-switching.

---

## PALAVRAS-CHAVE (inserir uma por uma no PLATEIA, pressionar Enter após cada)

1. planejamento de trajetória adaptativo
2. aprendizado por reforço profundo
3. navegação autônoma
4. planejamento híbrido
5. densidade de obstáculos

---

## DADOS DO AUTOR RESPONSÁVEL

- Nome: Yan Santos Leite
- Matrícula: 202302594
- Curso: Engenharia Mecânica — EMC/UFG
- Orientador: Prof. Dr. Aldo André Diaz Salazar — INF/UFG

---

*Observação: o texto do resumo acima não inclui título, nomes ou palavras-chave na contagem de caracteres (conforme item 3.10 do edital). Verificar a contagem final no campo de submissão do PLATEIA antes de confirmar.*
