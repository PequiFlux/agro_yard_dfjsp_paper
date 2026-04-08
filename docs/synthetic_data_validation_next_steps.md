# Próximos Passos para Validação e Melhoria do Dataset Sintético

Este documento consolida próximos passos recomendados pela literatura para fortalecer a validade e a utilidade do release `v1.1.0-observed` como benchmark sintético e como dataset seed para futuras gerações com G2MILP.

O ponto central é separar cinco dimensões que costumam ser confundidas:

- fidelidade ao processo real
- diversidade e cobertura do espaço de instâncias
- ausência de memorização ou casos quase duplicados
- utilidade downstream para tarefas operacionais
- utilidade como benchmark algorítmico

O release atual já está forte em integridade estrutural, executabilidade, diversidade inicial do espaço de instâncias e smoke test orientado a solver. Os itens abaixo indicam o que ainda pode elevar o nível metodológico do benchmark.

A busca adicional na literatura reforça quatro mensagens:

- avaliações puramente visuais ou só com marginais costumam ser insuficientes
- é melhor combinar testes globais, locais, downstream e estruturais
- dados sintéticos relacionais exigem checks além de uma tabela achatada
- benchmarking algorítmico confiável pede não só casos válidos, mas também casos informativos, discriminantes e reprodutíveis

## 1. Adicionar validação holdout-based contra dados reais

Se existir mesmo um subconjunto real pequeno e confidencial, este é o próximo passo de maior impacto.

Objetivo:

- verificar se o sintético não é apenas coerente internamente, mas também aderente ao processo real
- comparar o sintético contra um `holdout real`, não apenas contra o conjunto usado para calibrar a geração

Como aplicar neste benchmark:

- comparar distribuições marginais de chegada, carga, prazos observados, tempos por estágio e tempos de fila
- comparar dependências bivariadas e trivariadas por `commodity`, `moisture`, `shift`, `regime`
- comparar caudas `p90/p95/p99` para `queue_time`, `flow_time`, `due_margin`
- medir distância de vizinhança entre registros sintéticos e `train real` versus `holdout real`

Por que isso importa:

- sem `holdout real`, o benchmark pode ser forte como artefato metodológico e ainda assim não ter fidelidade empírica suficiente ao sistema operacional real

## 2. Separar fidelity, diversity e authenticity

Os próximos testes devem distinguir três propriedades diferentes:

- o dado parece real
- o dado cobre bem o suporte do problema
- o dado não replica em excesso exemplos específicos

Como aplicar neste benchmark:

- calcular `alpha-precision`, `beta-recall` e `authenticity`
- rodar essas métricas no nível de `job`
- repetir no nível de linhas de `eligible_machines.csv`, onde os padrões condicionais são mais ricos

Por que isso importa:

- marginais podem parecer corretas enquanto o dataset cobre mal regiões do espaço operacional
- um dataset pode parecer diverso e ainda assim memorizar em excesso algumas estruturas

## 3. Adicionar testes formais de duas amostras e diagnósticos de razão de densidade

Além de gráficos e comparações descritivas, vale acrescentar testes formais para checar se o sintético e a referência real diferem de maneira estatisticamente detectável.

Como aplicar neste benchmark:

- usar `MMD` para testar diferença entre distribuições de registros no nível de `job`
- usar `Classifier Two-Sample Tests` para detectar diferenças entre real e sintético que não aparecem claramente em marginais
- usar estimativas de `density ratio` para identificar onde o sintético está muito concentrado, muito rarefeito ou deslocado em relação à referência

Por que isso importa:

- `MMD` adiciona um teste não paramétrico global
- `C2ST` costuma ser útil quando as diferenças são sutis e multivariadas
- `density ratio` ajuda a localizar precisamente em quais regiões do espaço o gerador está errando

Base na literatura:

- `MMD`: Gretton et al. (2012)
- `C2ST`: Lopez-Paz & Oquab (2016)
- `density ratio`: Volker, de Wolf & van Kesteren (2024)

Como isso melhora o notebook:

- adicionar uma seção de testes formais `real vs synthetic`
- exportar `p-values`, medidas de efeito e um mapa de regiões com maior desvio de densidade

Observação:

- se não houver dado real, esses testes ainda podem ser úteis entre versões do próprio benchmark, por exemplo `core v1.0.0` versus `observed v1.1.0`, mas isso mede transformação interna, não realismo empírico

## 4. Validar a estrutura relacional e multiarquivo de forma explícita

O seu benchmark não é uma única tabela. Ele é um conjunto relacional com `jobs`, `operations`, `eligible_machines`, `events`, `fifo_schedule`, `metrics` e catálogos agregados.

Como aplicar neste benchmark:

- medir preservação de cardinalidades entre tabelas
- medir consistência de chaves e dependências cruzadas
- medir se relações condicionais entre tabelas continuam plausíveis após síntese ou novas derivações
- comparar não só colunas isoladas, mas também subestruturas como `(job -> operações -> máquinas elegíveis -> agenda FIFO -> métricas)`

Por que isso importa:

- benchmarks relacionais podem parecer corretos em cada arquivo separadamente e ainda assim degradar a coerência estrutural do conjunto
- a literatura mais recente mostra que avaliação relacional precisa ir além de flattening em uma tabela única

Base na literatura:

- Hudovernik, Jurkovič & Štrumbelj (2024)

Como isso melhora o notebook:

- adicionar painéis de consistência relacional
- adicionar métricas de preservação de joins e dependências entre tabelas
- adicionar um resumo de integridade relacional para futuras derivações G2MILP

## 5. Medir utilidade downstream com TSTR/TRTS

A literatura de dados sintéticos usa com frequência avaliações do tipo:

- `TSTR`: train on synthetic, test on real
- `TRTS`: train on real, test on synthetic

Como aplicar neste benchmark:

- treinar tarefas auxiliares como previsão de `overwait`, `flow_time` alto, congestionamento por janela e risco de atraso
- comparar performance quando o treino usa sintético e o teste usa dados reais
- comparar assimetria entre `TSTR` e `TRTS`

Por que isso importa:

- aparência estatística sozinha não garante que o sintético preserve relações úteis para modelagem

Observação:

- este passo depende de existir algum conjunto real, ainda que pequeno

## 6. Medir fidelidade semântica com explicabilidade

Uma direção recente é verificar se modelos treinados no sintético usam sinais parecidos com os observados no real.

Como aplicar neste benchmark:

- treinar modelos auxiliares para prever `overwait`, `flow_time`, gargalo de fila ou risco de atraso
- comparar importância de variáveis entre treino em real e treino em sintético
- usar uma métrica como `SHAP Distance` para checar se `priority`, `appointment`, `commodity`, `moisture`, `shift` e congestionamento preservam papéis semelhantes

Por que isso importa:

- o sintético pode acertar distribuições e ainda errar a lógica causal ou semântica que gera o comportamento operacional

## 7. Usar um framework de avaliação mais padronizado

Outro ponto forte da busca foi a recomendação de padronizar a avaliação com uma grade mais explícita de critérios.

Como aplicar neste benchmark:

- organizar os checks em blocos formais: `fidelity`, `utility`, `privacy`, `structure`, `benchmark usefulness`
- adotar uma rubrica clara para decidir o que é `pass`, `warning` e `fail`
- deixar explícito quais métricas são diagnósticas e quais são critérios de aceitação do release

Por que isso importa:

- frameworks como `SynthEval` mostram valor em consolidar múltiplas métricas sob um protocolo único
- a proposta `CAIR` é útil como disciplina para avaliar se as métricas escolhidas são comparáveis, aplicáveis, interpretáveis e representativas

Base na literatura:

- Lautrup et al. (2024), `SynthEval`
- Hyrup et al. (2023/2025), `CAIR`

Como isso melhora o notebook e o repo:

- transformar o notebook em um protocolo auditável de release
- criar um pequeno scorecard por release derivada
- diferenciar claramente checks obrigatórios de checks exploratórios

## 8. Expandir PCA + kNN para instance space analysis orientada a benchmark

O notebook já faz um bom primeiro passo com `PCA`, vizinho mais próximo e screening de casos `duplicate-like`.

O próximo nível seria:

- mapear `solver footprints` no espaço de instâncias
- identificar regiões fáceis, intermediárias e difíceis
- localizar lacunas do espaço que hoje estão subamostradas
- gerar novas instâncias de forma direcionada para preencher essas lacunas

Como aplicar neste benchmark:

- sobrepor desempenho de solver ao gráfico de espaço de instâncias
- mostrar onde exato, híbrido e metaheurístico se separam
- usar isso como guia para futuras derivações G2MILP

Por que isso importa:

- um benchmark melhor não é apenas “maior”; ele cobre melhor o espaço relevante de instâncias

## 9. Gerar instâncias graded e discriminating

Além das famílias `XS/S/M/L` e `balanced/peak/disrupted`, vale produzir instâncias deliberadamente úteis para comparação de algoritmos.

Dois objetivos distintos:

- `graded instances`: instâncias que formam uma escada clara de dificuldade
- `discriminating instances`: instâncias que separam com nitidez famílias de métodos

Como aplicar neste benchmark:

- gerar famílias em que casos pequenos fechem com exato
- produzir uma zona intermediária onde o trade-off tempo versus qualidade fique claro
- produzir regiões onde a metaheurística tenha vantagem operacional nítida

Por que isso importa:

- isso melhora o benchmark como instrumento de comparação, não apenas como coleção de cenários

## 10. Fortalecer a parte solver-oriented com performance profiles e estabilidade de ranking

Para benchmarking de otimização, médias simples de tempo ou gap costumam esconder comportamento importante.

Como aplicar neste benchmark:

- adicionar `performance profiles`
- adicionar curvas `fixed-budget`
- adicionar curvas `fixed-target`
- comparar solvers ou variantes sob múltiplos orçamentos e múltiplas sementes
- medir estabilidade do ranking entre seeds, budgets e subconjuntos de instâncias
- reportar quando diferenças entre métodos são pequenas demais para suportar afirmações fortes

Por que isso importa:

- essas visualizações mostram robustez, dominância e sensibilidade de forma mais defensável do que médias agregadas
- boas práticas de benchmarking recomendam reprodutibilidade, análise cuidadosa e interpretação prudente de pequenas diferenças de ranking

Base na literatura:

- Dolan & Moré (2002)
- Bartz-Beielstein et al. (2020)

## 11. Reforçar validação de caudas e segmentos raros

O dataset já preserva monotonicidade em `mean_flow`, `p95_flow` e fila média. O próximo passo é ir além de médias e percentis gerais.

Como aplicar neste benchmark:

- validar `p95/p99` de `queue_time` e `flow_time` por regime
- validar segmentos raros como `URGENT`, `WET`, janelas próximas de downtime, commodities menos frequentes
- medir relações condicionais entre atraso, fila e regime nesses subconjuntos

Por que isso importa:

- benchmarks sintéticos frequentemente acertam o centro da distribuição e erram justamente os eventos operacionais mais críticos

## 12. Adicionar auditoria de privacidade apenas se houver claim de privacidade

Se o benchmark vier a ser apresentado também como mecanismo de proteção de dados sensíveis, a validação precisa mudar.

Como aplicar:

- não depender apenas de distância ao vizinho mais próximo
- testar ataques de membership inference
- usar abordagens como DOMIAS ou auditorias equivalentes

Por que isso importa:

- métricas ingênuas de proximidade podem subestimar risco real de vazamento

Observação:

- se o objetivo é apenas benchmark algorítmico, esta seção pode ficar fora do protocolo principal

## 13. Fortalecer reprodutibilidade e governança do benchmark

O benchmark já está bem organizado, mas a literatura de benchmarking em otimização enfatiza que protocolo e reprodutibilidade também são parte da validade.

Como aplicar neste benchmark:

- congelar orçamento de tempo, seeds, hardware e critérios de parada nas comparações
- versionar explicitamente o conjunto de instâncias usado em cada experimento
- registrar mudanças de gerador, hiperparâmetros e auditorias de forma padronizada
- evitar misturar, em uma mesma tabela, resultados de protocolos diferentes

Por que isso importa:

- melhora confiança na comparação entre métodos
- evita que pequenas mudanças de harness sejam confundidas com ganho real de solver
- facilita auditoria pública, reprodução e uso em banca ou artigo

## Ordem sugerida de execução

### Pode ser feito agora, sem dados reais

- testes formais `MMD/C2ST` entre versões do benchmark
- `density ratio` para localizar regiões mal cobertas do espaço sintético
- validação relacional explícita entre arquivos
- protocolar scorecards no estilo `SynthEval/CAIR`
- performance profiles e curvas `fixed-budget/fixed-target`
- instance space analysis com footprints de solver
- expansão para instâncias `graded` e `discriminating`
- validação mais forte de caudas e segmentos raros
- reforço de reprodutibilidade e estabilidade de ranking

### Depende de algum conjunto real de referência

- validação holdout-based
- `alpha-precision`, `beta-recall`, `authenticity`
- `TSTR/TRTS`
- fidelidade semântica com `SHAP Distance`

### Só faz sentido se houver claim de privacidade

- membership inference
- DOMIAS ou auditoria equivalente

## Backlog implementável no repositório

Se a ideia for transformar essas recomendações em entregas concretas no repo, eu priorizaria este backlog:

### Prioridade alta

- adicionar ao notebook uma seção de `MMD/C2ST` e `density ratio`
- adicionar uma seção de integridade relacional multiarquivo
- adicionar `performance profiles` para os smoke tests e futuras baterias de solver
- adicionar um scorecard por release com status `PASS/WARNING/FAIL`

### Prioridade média

- adicionar `solver footprints` sobre o espaço `PCA + kNN`
- medir estabilidade de ranking entre budgets e seeds
- gerar novas instâncias em lacunas do espaço já identificado

### Prioridade condicionada a dados reais

- criar pipeline de `holdout-based validation`
- adicionar `TSTR/TRTS`
- adicionar `SHAP Distance`

## Interpretação metodológica recomendada para o TCC

Com o estado atual do release, já é defensável afirmar que:

- o benchmark é estruturalmente íntegro
- o baseline FIFO é executável contra o schema
- a camada observacional reduziu determinismo excessivo
- o espaço de instâncias não colapsa em duplicatas triviais
- o benchmark já é informativo para comparação algorítmica inicial

Mas, sem um `holdout real`, a formulação mais correta é:

- o dataset é um benchmark sintético forte e útil
- ele ainda não prova, por si só, fidelidade empírica completa ao processo real

## Referências

1. Alaa, A., Van Breugel, B., Saveliev, E., & van der Schaar, M. (2022). *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models*. Proceedings of the 39th International Conference on Machine Learning. https://arxiv.org/abs/2102.08921
2. Platzer, M., & Reutterer, T. (2021). *Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data*. https://arxiv.org/abs/2104.00635
3. Esteban, C., Hyland, S. L., & Rätsch, G. (2017). *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs*. https://arxiv.org/abs/1706.02633
4. Yu, W., Mousavi, S. F., et al. (2025). *SHAP Distance: An Explainability-Aware Metric for Evaluating the Semantic Fidelity of Synthetic Tabular Data*. https://arxiv.org/abs/2511.17590
5. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). *A Kernel Two-Sample Test*. Journal of Machine Learning Research, 13, 723-773. https://www.jmlr.org/beta/papers/v13/gretton12a.html
6. Lopez-Paz, D., & Oquab, M. (2016). *Revisiting Classifier Two-Sample Tests*. https://arxiv.org/abs/1610.06545
7. Volker, T. B., de Wolf, P.-P., & van Kesteren, E.-J. (2024). *A density ratio framework for evaluating the utility of synthetic data*. https://arxiv.org/abs/2408.13167
8. Lautrup, A. D., Hyrup, T., Zimek, A., & Schneider-Kamp, P. (2024). *SynthEval: A Framework for Detailed Utility and Privacy Evaluation of Tabular Synthetic Data*. https://arxiv.org/abs/2404.15821
9. Hyrup, T., Lautrup, A. D., Zimek, A., & Schneider-Kamp, P. (2023). *Sharing is CAIRing: Characterizing Principles and Assessing Properties of Universal Privacy Evaluation for Synthetic Tabular Data*. https://arxiv.org/abs/2312.12216
10. Hudovernik, V., Jurkovič, M., & Štrumbelj, E. (2024). *Benchmarking the Fidelity and Utility of Synthetic Relational Data*. https://arxiv.org/abs/2410.03411
11. van Breugel, B., et al. (2023). *Membership Inference Attacks against Synthetic Data through Overfitting Detection*. https://arxiv.org/abs/2302.12580
12. Yao, X., et al. (2025). *The DCR Delusion: Measuring the Privacy Risk of Synthetic Data*. https://arxiv.org/abs/2505.01524
13. Ward, J., Lin, X., Wang, C.-H., & Cheng, G. (2025). *Synth-MIA: A Testbed for Auditing Privacy Leakage in Tabular Data Synthesis*. https://arxiv.org/abs/2509.18014
14. Smith-Miles, K., Baatar, D., Wreford, B., & Lewis, R. (2014). *Towards objective measures of algorithm performance across instance space*. Computers & Operations Research, 45, 12-24. https://orca.cardiff.ac.uk/id/eprint/53966/
15. Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). *Automated Algorithm Selection: Survey and Perspectives*. Evolutionary Computation, 27(1), 3-45. https://arxiv.org/abs/1811.11597
16. Kletzander, L., Musliu, N., & Raidl, G. R. (2021). *Instance space analysis for a personnel scheduling problem*. Annals of Operations Research, 306, 107-140. https://link.springer.com/article/10.1007/s10472-020-09695-2
17. Dang, N.-T., Akgün, Ö., Espasa, J., Miguel, I., & Nightingale, P. (2022). *A Framework for Generating Informative Benchmark Instances*. https://arxiv.org/abs/2205.14753
18. Dolan, E. D., & Moré, J. J. (2002). *Benchmarking Optimization Software with Performance Profiles*. Mathematical Programming, 91, 201-213. https://arxiv.org/abs/cs/0102001
19. Bartz-Beielstein, T., et al. (2020). *Benchmarking in Optimization: Best Practice and Open Issues*. https://arxiv.org/abs/2007.03488
20. Murad, M., & Ruocco, M. (2025). *Pre-Tactical Flight-Delay and Turnaround Forecasting with Synthetic Aviation Data*. https://arxiv.org/abs/2508.02294
