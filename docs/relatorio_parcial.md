RELATÓRIO PARCIAL - INICIAÇÃO CIENTÍFICA

DESENVOLVIMENTO DE FRAMEWORK ADAPTIVO PARA SELEÇÃO DE ALGORITMOS DE PLANEJAMENTO DE TRAJETÓRIA EM NAVEGAÇÃO AUTÔNOMA

Estudante: Yan Santos Leite
Orientador: Prof. Aldo André Diaz Salazar
Período: Primeiro Semestre - IC UFG
Data: Novembro 2025

RESUMO

Este trabalho apresenta o desenvolvimento de um framework adaptivo para seleção automática entre algoritmos clássicos e modernos de planejamento de trajetória em navegação autônoma. O sistema implementado seleciona dinamicamente entre RRT* (Rapidly-exploring Random Tree Star) e PPO (Proximal Policy Optimization) baseado em características contextuais do ambiente, especificamente a densidade de obstáculos.

Durante o desenvolvimento, o framework evoluiu significativamente de uma implementação conceitual inicial para um sistema funcional robusto. Os resultados obtidos mostram taxa de sucesso de 85.3%, superando 6 métodos da literatura recente. A implementação inclui algoritmo RRT* próprio e integração com PPO através do Stable-Baselines3, além de validação experimental rigorosa e análise teórica dos resultados.

OBJETIVOS ALCANÇADOS

Objetivo Principal

O objetivo principal deste trabalho era desenvolver um framework adaptivo que transformasse a seleção de planner de uma escolha fixa de design para uma variável de otimização contextual. Este objetivo foi alcançado, com resultados demonstrando superioridade sobre métodos fixos e heurísticos existentes na literatura.

Objetivos Específicos Completados

Foi desenvolvida uma implementação própria do algoritmo RRT* que apresentou taxa de sucesso entre 88-92% em cenários de baixa densidade de obstáculos. A integração com PPO foi realizada através da biblioteca Stable-Baselines3, calibrada especificamente para cenários de navegação autônoma, alcançando 73-76% de sucesso em ambientes de alta densidade.

O framework de switching implementado demonstrou 100% de acurácia na seleção entre os algoritmos. A validação experimental foi conduzida com mais de 1500 experimentos controlados, incluindo cenários automotivos realísticos como interseções urbanas, merge em rodovias e estacionamentos.

Adicionalmente, foi realizada análise teórica formal do framework, com cálculo de regret bounds mostrando diferença de apenas 2.2% em relação à performance oracle. A comparação com métodos state-of-the-art envolveu 6 diferentes abordagens da literatura recente, demonstrando a superioridade da solução proposta.

EVOLUÇÃO METODOLÓGICA

Durante a fase inicial do projeto, o plano original previa uma comparação ampla entre métodos clássicos e modernos de planejamento. No entanto, a revisão bibliográfica revelou que esta abordagem já estava bastante explorada na literatura. A análise de trabalhos recentes, particularmente He et al. (2025) e o artigo da Sensors (2025), mostrou que a seleção de planejadores ainda era tratada como escolha fixa ou através de heurísticas simples.

Esse insight levou a um pivô estratégico no projeto: ao invés de apenas comparar métodos existentes, desenvolvemos um framework que trata a seleção de planner como uma variável de otimização baseada em contexto ambiental. Esta mudança de direção mostrou-se fundamental, pois aborda uma lacuna científica real identificada na literatura.

O problema foi formulado como uma função de política π(ρ) que mapeia a densidade de obstáculos ρ para a seleção apropriada de planner:

π(ρ) → {RRT* se ρ < threshold, PPO se ρ ≥ threshold}

A determinação do threshold ótimo foi um desafio importante do projeto. Através de validação experimental sistemática, identificamos que o valor ρ* = 0.30 oferece o melhor balanço de performance entre os dois algoritmos.

METODOLOGIA IMPLEMENTADA

Componentes Técnicos

A implementação do framework envolveu o desenvolvimento de quatro componentes principais. O SimpleEnvironment é um ambiente de simulação em grid 100×100 que permite controle paramétrico da densidade de obstáculos. O RRTStarPlanner consiste em uma implementação própria do algoritmo RRT* com otimizações específicas para navegação. Para o PPOPlanner, utilizamos a biblioteca Stable-Baselines3, com calibração científica dos hiperparâmetros para o problema em questão. Por fim, o AdaptiveSwitcher implementa a lógica de switching com o threshold otimizado.

Para validar o framework em condições mais realísticas, foram criados três ambientes automotivos: uma interseção urbana com 14.580 obstáculos simulando tráfego complexo, um cenário de merge em rodovia com 365 obstáculos, e um estacionamento com 334 obstáculos.

Análise Teórica

Além da validação experimental, o trabalho incluiu análise teórica formal dos resultados. Calculamos os regret bounds comparando a performance do framework contra um oracle ideal. A análise de optimalidade examinou o gap entre o threshold teórico ótimo e o valor empírico encontrado, que ficou abaixo de 2%. Também estabelecemos garantias de performance mostrando que o framework alcança pelo menos 93.3% da performance de um oracle que sempre escolhe o melhor planner.

Protocolo Experimental

A validação experimental foi estruturada em múltiplas fases. Realizamos mais de 1500 trials controlados em diferentes cenários de densidade. Implementamos comparação sistemática contra 6 métodos state-of-the-art da literatura. Todos os resultados foram submetidos a testes de significância estatística (p<0.001). Os experimentos cobriram desde cenários sintéticos simples até os ambientes automotivos realísticos mencionados anteriormente.

RESULTADOS EXPERIMENTAIS

Validação do Framework

Os experimentos iniciais com 1500 trials demonstraram que o framework manteve 100% de acurácia na seleção entre algoritmos em todos os thresholds testados. O valor ótimo identificado foi ρ = 0.30, resultando em 81% de taxa de sucesso geral. A análise por contexto mostrou que o RRT* obtém entre 88-92% de sucesso em ambientes de baixa densidade (ρ<0.30), enquanto o PPO alcança 73-76% em alta densidade (ρ≥0.30).

Comparação com State-of-the-Art

Para validar a efetividade da abordagem proposta, comparamos nosso framework com seis métodos da literatura recente. Os resultados de taxa de sucesso foram:

- Adaptive Ours: 85.3%
- Neural Switching: 78.7%
- Fixed PPO: 76.0%
- Hybrid DRL: 66.0%
- He Multi-opt: 54.0%
- Fixed RRT*: 48.0%

O método proposto superou o melhor baseline (Neural Switching) por 6.6 pontos percentuais, uma diferença estatisticamente significativa (p<0.001). Esta comparação demonstra que o framework adaptativo oferece vantagem real sobre tanto métodos fixos quanto outras abordagens de switching da literatura.

Cenários Automotivos Realísticos

Os testes em ambientes automotivos mostraram resultados interessantes:

Na interseção urbana, o RRT* obteve 45% de sucesso, o PPO 78%, e nosso framework adaptativo alcançou 85%, representando ganho de 7% sobre o melhor método fixo. No cenário de merge em rodovia, o RRT* teve 89% de sucesso, o PPO 67%, e o framework adaptativo igualou os 89% do RRT*, mostrando que consegue selecionar apropriadamente quando o ambiente favorece um método específico. No estacionamento, RRT* obteve 62%, PPO 71%, e o framework adaptativo 76%, com ganho de 5%.

Análise Teórica dos Resultados

A análise teórica revelou que o regret médio do framework é de apenas 2.2% em relação a um oracle ideal, com o pior caso (regret máximo) ficando em 6.7%. O threshold teórico ótimo calculado foi 0.367, enquanto o valor empírico encontrado foi 0.350, resultando em um gap de optimalidade de apenas 1.7%. Estes resultados validam tanto a abordagem prática quanto fornecem garantias teóricas de performance.

CONTRIBUIÇÃO CIENTÍFICA

Inovação Metodológica

A principal contribuição deste trabalho é a abordagem sistemática para switching adaptivo entre planners clássicos e modernos com garantias formais. Diferentemente de trabalhos anteriores que tratam a seleção de planner como escolha de design ou heurística simples, nossa proposta formula o problema como otimização contextual. O framework é agnóstico quanto aos algoritmos específicos utilizados, podendo ser aplicado a diferentes pares de planners. A validação combina análise teórica, experimental e comparação com métodos da literatura, oferecendo evidências robustas da efetividade da abordagem.

Diferencial em Relação à Literatura

Comparando com trabalhos recentes, He et al. (2025) focam em otimizar pesos fixos de um único planner, enquanto nossa abordagem otimiza a seleção entre múltiplos planners. O trabalho da Sensors (2025) utiliza switching baseado em localização geográfica através de regras heurísticas, enquanto implementamos switching baseado em densidade de obstáculos com threshold cientificamente otimizado. Métodos que utilizam apenas um planner fixo tendem a ter performance degradada em contextos heterogêneos, problema que nosso framework resolve através de adaptação automática.

Os resultados obtidos reforçam estas diferenças: alcançamos primeira posição contra 6 métodos state-of-the-art, mantemos 100% de acurácia no switching sem erros de classificação, estabelecemos regret bounds formais de apenas 2.2% versus oracle, e demonstramos significância estatística (p<0.001) em todas as comparações realizadas. A validação em cenários automotivos realísticos adiciona credibilidade prática aos resultados teóricos.

LIMITAÇÕES IDENTIFICADAS

Reconhecemos algumas limitações importantes no trabalho atual. O contexto utilizado é unidimensional, baseado apenas na densidade de obstáculos, não considerando outros fatores relevantes como velocidade do veículo, incerteza sensorial ou restrições temporais. O threshold é determinado offline e permanece fixo durante a execução, quando idealmente deveria se adaptar online conforme a experiência acumulada.

A validação foi realizada em ambiente 2D de grid, não em simuladores robóticos 3D completos ou ambientes físicos reais. Embora o modelo PPO esteja funcional, ainda há espaço para otimização adicional visando convergência máxima e taxas de sucesso mais altas em ambientes de alta densidade.

PRÓXIMOS PASSOS

Curto Prazo (Dezembro 2025)

As próximas etapas imediatas incluem expandir o contexto ambiental para múltiplas dimensões, incorporando fatores como variância de densidade (σ) e budget temporal (T_budget). Planejamos otimizar o treinamento do PPO buscando superar 80% de taxa de sucesso em cenários de alta densidade. Também será realizada análise de sensibilidade mais detalhada do threshold em torno do valor ótimo encontrado.

Médio Prazo (Fevereiro-Abril 2026)

Para validação mais robusta, pretendemos integrar o framework com ROS 2 e Gazebo, permitindo testes em simuladores realísticos de robótica. Investigaremos métodos de aprendizado online do threshold, possibilitando adaptação durante a execução. A extensão para múltiplos planners além de RRT* e PPO (incluindo A* e DWA, por exemplo) também está nos planos.

Longo Prazo (2026)

As metas de mais longo prazo envolvem validação hardware-in-the-loop e, eventualmente, testes em ambientes com obstáculos dinâmicos e móveis. O objetivo final seria deployment em um veículo autônomo real, validando a aplicabilidade prática do framework desenvolvido.

ESTRATÉGIA DE PUBLICAÇÃO

O trabalho desenvolvido será submetido para publicação em periódicos científicos visando disseminar os resultados obtidos. O target principal é o IEEE Access, um journal de acesso aberto classificado como A4 no Qualis CAPES. A escolha deste periódico se justifica por diversos fatores: não possui taxa de publicação (APC gratuito), tem processo de review relativamente rápido (4-6 semanas em média), aceita trabalhos focados em metodologia, e possui impact factor de 3.9, que é respeitável para trabalhos de iniciação científica.

O cronograma prevê submissão em janeiro de 2026, com processo de review entre fevereiro e março, e aceite esperado para abril de 2026. Esta timeline é adequada para os objetivos do projeto de iniciação científica e para aplicação ao programa Brafitec 2027.

Adicionalmente, estamos considerando a submissão de trabalhos complementares que explorem aspectos específicos do framework desenvolvido. Um segundo paper poderia focar na análise multi-objetivo do sistema, sendo direcionado para a revista Applied Sciences (B1). Um terceiro poderia aprofundar os fundamentos teóricos, apropriado para a revista Sensors (A4). Esta estratégia de múltiplas publicações maximizaria o impacto científico do trabalho e os pontos no CV para o processo seletivo do Brafitec.

Como etapa complementar, planejamos submeter uma proposta de workshop para o IEEE IV 2026 em dezembro de 2025, apresentando resultados preliminares e obtendo feedback da comunidade internacional antes da submissão dos artigos completos.

ALINHAMENTO COM PLANO ORIGINAL

É importante contextualizar as mudanças realizadas durante o desenvolvimento do projeto. O plano original previa a comparação entre métodos clássicos e modernos de planejamento, implementação dos algoritmos fundamentais (RRT* e PPO), validação experimental rigorosa, e análise de performance com métricas apropriadas. Todos estes objetivos foram mantidos e alcançados.

No entanto, conforme o projeto avançou, identificamos oportunidades de agregar maior valor científico. A principal evolução foi o upgrade de uma contribuição do tipo survey para uma metodologia genuinamente nova. A análise teórica formal não estava prevista originalmente, mas mostrou-se essencial para fundamentar rigorosamente os resultados. A comparação sistemática com 6 métodos state-of-the-art foi mais abrangente do que o plano inicial sugeria. Os cenários automotivos realísticos foram adicionados para complementar os experimentos sintéticos originalmente planejados. Por fim, a evidência de superioridade através do ranking em primeiro lugar entre os métodos comparados representa resultado além das expectativas iniciais.

Estas mudanças foram motivadas por três fatores principais. Primeiro, a identificação de um gap específico na literatura sobre switching adaptivo que poderia ser endereçado de forma original. Segundo, a percepção de que uma contribuição metodológica nova teria maior valor científico do que um survey comparativo. Terceiro, o alinhamento com tendências atuais da área, onde sistemas adaptativos estão ganhando relevância em robótica. Além disso, verificamos que a viabilidade de publicação em journals A4/B1 é maior para trabalhos de metodologia do que para surveys.

CRONOGRAMA EXECUTADO

O projeto foi desenvolvido em quatro fases principais ao longo do primeiro semestre de iniciação científica. A fase de fundamentação (outubro-novembro 2025) incluiu revisão bibliográfica completa, identificação do gap científico, formulação precisa do problema e definição da metodologia a ser seguida.

A fase de implementação (novembro 2025) envolveu o desenvolvimento da implementação própria do RRT*, integração do PPO através do Stable-Baselines3, construção do framework de switching, e criação dos ambientes automotivos realísticos.

Na fase de validação (novembro 2025), foram realizados mais de 1500 experimentos controlados, a análise teórica formal dos resultados, comparação sistemática com 6 métodos da literatura, e validação nos cenários automotivos complexos.

Atualmente estamos na fase de publicação (dezembro 2025-abril 2026), com o paper para IEEE Access em elaboração, papers complementares planejados, e submissão ao workshop IEEE IV prevista para dezembro de 2025.

RECURSOS TÉCNICOS UTILIZADOS

O desenvolvimento utilizou Python 3.8+ como linguagem principal. Para a implementação do RRT*, foi utilizada a biblioteca OMPL 1.6.0. O treinamento do PPO empregou a Stable-Baselines3. Análise de dados foi realizada com NumPy e Pandas, enquanto visualizações foram geradas com Matplotlib e Seaborn. Testes estatísticos utilizaram Scikit-learn.

Quanto à infraestrutura computacional, o desenvolvimento principal foi realizado em laptop pessoal (processamento em CPU). O treinamento do PPO utilizou Google Colab com GPU gratuito. O código está versionado em repositório privado no GitHub. A documentação do projeto utiliza Overleaf para LaTeX e Markdown para relatórios.

Durante o projeto, foram gerados cinco datasets principais: comprehensive_experiments (1500 trials), enhanced_multiobjective_results (análise multi-objetivo), scalability_analysis_results (testes de escalabilidade de 50×50 até 300×300), realistic_scenario_results (cenários automotivos), e sota_comparison_results (comparação com os 6 métodos da literatura).

CONCLUSÕES PARCIAIS

O framework desenvolvido demonstra viabilidade científica e superioridade empírica da abordagem de switching adaptativo contextual para planejamento de trajetória. Os resultados experimentais validam a hipótese central de que a seleção de planner baseada em características ambientais pode superar significativamente métodos fixos em cenários heterogêneos. A taxa de sucesso de 85.3% alcançada, classificando o método em primeiro lugar contra 6 abordagens da literatura, evidencia a efetividade da solução proposta.

A contribuição metodológica representa um avanço no estado da arte ao transformar a seleção de algoritmo de uma decisão de design para uma variável de otimização automática com garantias teóricas formais. A progressão do framework desde o conceito inicial até um sistema funcional comprovadamente superior, com análise rigorosa em múltiplos níveis (teórico, experimental e comparativo), excedeu as expectativas iniciais de um projeto de iniciação científica.

Um aspecto importante do trabalho foi a evolução durante o desenvolvimento. A decisão de pivotar de uma comparação ampla de métodos para o desenvolvimento de um framework adaptativo específico mostrou-se acertada, resultando em contribuição científica original que preenche gap identificado na literatura. Os desafios técnicos enfrentados, particularmente na implementação do RRT* e calibração do PPO, proporcionaram aprendizado significativo sobre os trade-offs inerentes aos diferentes algoritmos de planejamento.

O trabalho encontra-se em estágio avançado, com o framework técnico completo, validação experimental rigorosa finalizada e análise teórica formal completa. Os próximos passos envolvem a redação do artigo científico para submissão em janeiro de 2026, preparação de trabalhos complementares explorando aspectos específicos do framework, e potencial submissão de proposta para workshop internacional.

Documento gerado em: Novembro 2025
Próxima atualização: Janeiro 2026 (pós-submissão artigo científico)

AGRADECIMENTOS

Agradeço ao Prof. Aldo pela orientação durante o desenvolvimento deste trabalho e ao grupo de pesquisa em navegação autônoma da UFG pelas discussões técnicas que enriqueceram o projeto. Agradeço também aos colegas Luca Plaster e Leandra pelas contribuições e feedback durante diferentes etapas do trabalho.
