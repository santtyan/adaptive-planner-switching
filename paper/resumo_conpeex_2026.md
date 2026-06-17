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

## TÍTULO (colar em CAIXA ALTA — 117 chars ✅)

DESENVOLVIMENTO DE FRAMEWORK ADAPTIVO PARA SELEÇÃO DE ALGORITMOS DE PLANEJAMENTO DE TRAJETÓRIA EM NAVEGAÇÃO AUTÔNOMA

---

## RESUMO (corpo — colar no campo "Resumo" do PLATEIA)

*Texto corrido, alinhamento justificado. Contar os chars no PLATEIA antes de submeter. Estimativa: ~2.380 chars.*

---

O planejamento de trajetória em robótica autônoma enfrenta um dilema fundamental: algoritmos determinísticos como Dijkstra e A* oferecem garantias de otimalidade e completude, mas degradam-se em ambientes com alta densidade de obstáculos; políticas de aprendizado por reforço, como SAC (Soft Actor-Critic), adaptam-se melhor a contextos complexos, porém são desnecessariamente custosas em espaços abertos. Este trabalho tem como objetivo desenvolver e validar um framework adaptivo que seleciona automaticamente, em tempo de execução, o planejador mais adequado com base nas características do ambiente. Para a seleção, é proposto o critério ρ (rho), que calcula a densidade local de obstáculos em uma janela ao redor da pose do robô e aplica a política π(ρ) = {A* se ρ < 0,30; SAC se ρ ≥ 0,30}, com limiar ρ* = 0,30 determinado por validação experimental. A metodologia envolveu a implementação e benchmarking de quatro algoritmos clássicos — Dijkstra, A*, Floyd-Warshall e Johnson — em grids de 100 a 2.500 nós, com medição de tempo de execução (timeit) e consumo de memória de pico (tracemalloc), além da integração completa em ROS2 Humble com Gazebo Classic e TurtleBot3 Waffle via Stable-Baselines3. Os resultados mostram que Dijkstra e A* escalam linearmente: 0,07 ms e 3,7 KB para grids de 100 nós; 2,46 ms e 85 KB para 2.500 nós. Floyd-Warshall e Johnson tornam-se inviáveis para planejamento em tempo real: Floyd-Warshall consome 39 segundos e 22 MB em um grid 30×30 (O(n³)), enquanto Johnson demora 854 ms e 57 MB no mesmo cenário. Esses dados justificam empiricamente a escolha de A* como componente clássico do framework. A validação do critério adaptivo em 1.500 experimentos controlados demonstrou taxa de sucesso de 85,3% contra 76% do melhor método fixo, com regret de apenas 2,2% em relação a um seletor oracle ideal. Conclui-se que a seleção adaptiva de planejador baseada em densidade de obstáculos supera métodos fixos em ambientes heterogêneos, com evidência teórica e empírica. A implementação completa está disponível em repositório público, promovendo reprodutibilidade e continuidade da pesquisa.

---

## PALAVRAS-CHAVE (inserir uma por uma no PLATEIA, pressionar Enter após cada)

1. planejamento de trajetória
2. aprendizado por reforço
3. navegação autônoma
4. ROS2
5. switching adaptivo

---

## DADOS DO AUTOR RESPONSÁVEL

- Nome: Yan Santos Leite
- Matrícula: 202302594
- Curso: Engenharia Mecânica — EMC/UFG
- Orientador: Prof. Dr. Aldo André Diaz Salazar — INF/UFG

---

*Observação: o texto do resumo acima não inclui título, nomes ou palavras-chave na contagem de caracteres (conforme item 3.10 do edital). Verificar a contagem final no campo de submissão do PLATEIA antes de confirmar.*
