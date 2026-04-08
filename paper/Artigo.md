# Documento de desenho experimental e metodológico para o artigo

Este desenho foi montado para converter a sua revisão sistemática e o repositório `agro_yard_dfjsp_benchmark_go` em um artigo computacional forte para o SBPO. A revisão já delimitou a lacuna correta: a literatura é rica em comparação algorítmica, mas ainda fraca quando se exige validade operacional, acoplamento entre subsistemas, resposta adaptativa e grounding empírico; no corpus retido, 32 estudos são computacionais e apenas 3 usam dados reais de forma mais direta, e a agenda final aponta como prioridades modelos integrados, comparação entre horizonte rolante e controle disparado por eventos, e validação com digital twins ou dados reais. O repositório já entrega exatamente o artefato para esse passo: um benchmark sintético observacional com 36 instâncias, escalas `XS/S/M/L`, regimes `balanced/peak/disrupted`, baseline FIFO, eventos `JOB_VISIBLE`, `JOB_ARRIVAL`, `MACHINE_DOWN` e `MACHINE_UP`, trilhas de auditoria e um backlog explícito de ISA, performance profiles, `MMD/C2ST`, `density ratio`, `solver footprints` e instâncias `graded/discriminating`.    ([GitHub][1])

O enquadramento correto deste artigo também respeita o estágio do PequiFlux: o produto amplo ainda é descrito como uma trilha mais longa, saindo de TRL 1 para protótipo beta em ambiente controlado, com gêmeo digital, replay offline, benchmarking e futura validação em campo. Logo, o claim do paper deve ser de **benchmarking, explicação metodológica e seleção de métodos**, e não de impacto operacional observado em planta real. 

Metodologicamente, o artigo deve combinar três camadas. A primeira é benchmarking clássico de métodos de PO. A segunda é **Instance Space Analysis (ISA)** para visualizar o espaço de instâncias e explicar onde cada método ganha. A terceira é uma camada moderna de avaliação do benchmark sintético, com diagnósticos de cobertura, discriminabilidade e integridade relacional. ISA foi formulada justamente para avaliar forças e fraquezas algorítmicas ao longo do espaço de instâncias; ela já foi aplicada com sucesso a problemas de otimização e escalonamento, como car sequencing, e a literatura de FJSP já mostra que diferentes solvers têm forças complementares e podem ser escolhidos com base em features da instância. Além disso, a literatura recente de scheduling dinâmico mostra que sistemas que escolhem regras de despacho de forma periódica ou em tempo real são uma linha atual e competitiva. ([ScienceDirect][2])

## 1. Título sugerido

**Além do FIFO em pátios agroindustriais: análise do espaço de instâncias e seleção de métodos em um benchmark D-FJSP observacional**

## 2. Questão de pesquisa

**Como caracterizar o espaço de instâncias de um benchmark D-FJSP agroindustrial observacional e usar essa caracterização para comparar, explicar e selecionar métodos de orquestração de caminhões — do FIFO a políticas reativas e famílias exata/híbrida/metaheurística — sob regimes `balanced`, `peak` e `disrupted`?**

### Subquestões operacionais

1. Quais características de instância mais explicam dificuldade, robustez e ganho relativo ao FIFO?
2. Em que regiões do espaço de instâncias métodos exatos, híbridos, metaheurísticos e políticas reativas são preferíveis?
3. O ISA revela lacunas de cobertura do benchmark e orienta geração de instâncias-filhas mais informativas?
4. Um seletor baseado em features consegue reduzir o arrependimento em relação ao melhor método fixo?

## 3. Tese do artigo

A tese central do artigo deve ser esta:

**não existe um método dominante em todo o benchmark; o método mais adequado depende da região do espaço de instâncias e do regime operacional, e essa dependência pode ser explicada por ISA, footprints de solver e modelos simples de seleção por features.**

Essa tese é coerente com a literatura de ISA, com a literatura de algorithm selection, com o estudo de FJSP por seleção de solver e com a sua própria revisão, que já conclui que o ponto relevante hoje não é “qual método ganha em média”, mas até onde o ganho continua crível quando a operação volta a ficar dinâmica, acoplada e incerta. ([ScienceDirect][2]) 

## 4. Delimitação do claim

O artigo **não** deve ser apresentado como prova de redução de filas em operação real. O claim correto é:

**“Apresentamos um protocolo auditável para comparar e selecionar métodos de orquestração de caminhões em um benchmark sintético observacional de pátio agroindustrial, mostrando como diferentes regiões do espaço de instâncias favorecem diferentes famílias de métodos.”**

Isso é importante porque o próprio repositório afirma que a base continua sintética e que o ganho metodológico não é “virar dado real”, mas sair de um benchmark excessivamente limpo para um seed mais útil em testes de robustez, comparação de métodos e geração de instâncias-filhas. A própria documentação também recomenda, para resultados mais fortes de fidelidade, holdout real, `TSTR/TRTS`, `authenticity` e `SHAP Distance`, mas apenas quando houver algum conjunto real de referência. ([GitHub][1])

## 5. Resumo executivo do artigo

O artigo deve ser um **estudo computacional orientado a decisão**, com duas contribuições fortes. A primeira contribuição é um benchmark comparativo entre métodos end-to-end para o problema de orquestração de caminhões em um pátio agroindustrial modelado como D-FJSP observacional. A segunda contribuição é um pipeline de entendimento do benchmark por meio de ISA, clustering, footprints de solver e classificação de instâncias, mostrando onde cada método é mais forte e onde o benchmark ainda precisa ser expandido.

A narrativa do artigo deve seguir esta lógica: a revisão sistemática identifica a lacuna; o benchmark fornece o ambiente auditável; os experimentos mostram diferenças reais entre métodos; o ISA explica essas diferenças; e a análise de cobertura orienta a próxima geração de instâncias.  ([GitHub][3])

## 6. Base empírica disponível e como usá-la

O benchmark atual já traz 36 instâncias, com 3 réplicas por combinação escala×regime, cobrindo `XS`, `S`, `M`, `L` e os regimes `balanced`, `peak` e `disrupted`. Cada job passa pelas quatro operações `WEIGH_IN`, `SAMPLE_CLASSIFY`, `UNLOAD` e `WEIGH_OUT`, e cada instância já inclui estrutura completa do problema, elegibilidade por máquina, precedências, indisponibilidades, eventos cronológicos, baseline FIFO, métricas agregadas por job e arquivos de auditoria da camada observacional. O catálogo do benchmark também já sugere a trilha por escala: `exact` em `XS/S`, `hybrid` em `M` e `metaheuristic` em `L`. ([GitHub][1])

Além disso, o repositório já passou pelos checks estruturais principais, incluindo `36/36` instâncias com `PASS`, consistência relacional, baseline FIFO reconciliado, checks de duplicata/near-duplicate no espaço de instâncias e smoke test com casos pequenos ótimos e gaps não triviais nos casos maiores. O ponto de cautela é que um diagnóstico auxiliar de monotonicidade de congestionamento ainda aparece como `False`, então a seção de ameaças à validade do artigo deve relatar isso de forma honesta. ([GitHub][1])

### Artefatos do repositório que entram diretamente no artigo

Use diretamente estes arquivos e scripts como base metodológica:

* `catalog/benchmark_catalog.csv`
* `catalog/instance_family_summary.csv`
* `catalog/schema_dictionary.csv`
* `instances/*/jobs.csv`
* `instances/*/operations.csv`
* `instances/*/precedences.csv`
* `instances/*/eligible_machines.csv`
* `instances/*/machine_downtimes.csv`
* `instances/*/events.csv`
* `instances/*/fifo_schedule.csv`
* `instances/*/fifo_job_metrics.csv`
* `instances/*/fifo_summary.json`
* `instances/*/job_noise_audit.csv`
* `instances/*/proc_noise_audit.csv`
* `instances/*/job_congestion_proxy.csv`
* `tools/validate_observed_release.py`
* `tools/validate_benchmark.py`
* `tools/exact_solver_smoke.py`
* `output/jupyter-notebook/instance_validation_analysis_artifacts/*` ([GitHub][1])

## 7. Formulação resumida do problema

O artigo deve formalizar o problema como um D-FJSP observacional de pátio agroindustrial. Em termos práticos:

* cada job representa um caminhão;
* cada job percorre uma cadeia fixa de quatro operações;
* cada operação tem elegibilidade de máquina e tempo de processamento dependente da tripla `(job, op, machine)`;
* existem precedências lineares e downtimes de máquina;
* o sistema evolui sob eventos `JOB_VISIBLE`, `JOB_ARRIVAL`, `MACHINE_DOWN`, `MACHINE_UP`;
* as decisões são de chamada, sequenciamento, atribuição a recurso e reotimização sob visibilidade parcial.

A função-objetivo do artigo deve priorizar validade operacional e não apenas makespan. O centro da avaliação deve ser redução de cauda de fila e de tempo de fluxo, com runtime e estabilidade do plano como contrapesos.

## 8. Métodos que entram no artigo

O artigo deve comparar **métodos completos**, e não apenas “solvers isolados”. O melhor desenho é este.

### M0. FIFO-Replay

É o baseline obrigatório. Deve ser reimplementado no seu engine de replay e reproduzir os artefatos oficiais do benchmark. Sem isso, o resto perde credibilidade. O próprio repositório já entrega `fifo_schedule.csv`, `fifo_job_metrics.csv` e `fifo_summary.json`. ([GitHub][1])

### M1. Heurística estática orientada por prioridade e folga

Aqui entra uma regra simples e explicável, por exemplo **Weighted Slack** com desempate por ordem de chegada. Ela serve como baseline não trivial, barata computacionalmente e fácil de explicar.

### M2. Horizonte rolante periódico com reparo exato

A cada intervalo fixo `Δ`, o sistema reotimiza o conjunto visível e ainda não iniciado, congelando operações em execução. Para `XS/S`, use reparo exato com CP-SAT, Gurobi ou o backend leve já próximo ao smoke test; para `M`, use subproblema limitado por tempo.

### M3. Reotimização disparada por eventos com reparo híbrido

O mesmo subproblema de M2, mas disparado pelos eventos `JOB_VISIBLE`, `JOB_ARRIVAL`, `MACHINE_DOWN` e `MACHINE_UP`. A revisão já aponta essa comparação como uma pergunta central; ela também conversa com o que há de mais promissor para o agronegócio, especialmente integração entre portaria, pesagem, amostragem, descarga e destinação.  

### M4. Metaheurística de grande escala para `L` (opcional forte)

Se houver fôlego de implementação, use ALNS/LNS, Tabu ou outro reparo metaheurístico para as instâncias `L`. Se não houver, mantenha M3 como principal método grande-porte e deixe M4 como extensão.

### Mref. Referência exata para `XS/S`

O repositório já sugere `exact` para `XS/S`, `hybrid` para `M` e `metaheuristic` para `L`. Então o mais forte é usar um método exato como referência de qualidade apenas nas menores escalas, não como competidor de produção em todas as instâncias. ([GitHub][1])

## 9. Pipeline de entendimento das instâncias

Aqui está o coração diferencial do artigo.

### 9.1 Matriz de features das instâncias

Monte uma matriz `X` com features por instância. Divida em seis grupos.

**Grupo A — tamanho e estrutura**
`n_jobs`, `n_machines`, densidade de elegibilidade, média de máquinas elegíveis por operação, variância de flexibilidade, carga mínima por estágio, índice de gargalo por estágio.

**Grupo B — estrutura temporal**
span de chegadas, estatísticas de `reveal_time - arrival_time`, contagem de eventos por horizonte, média e variância do inter-event time, fração de jobs visíveis no início.

**Grupo C — prioridade e prazo**
média, desvio, p10 e p90 do due slack; entropia de classes de prioridade; taxa de appointment; fração de jobs urgentes.

**Grupo D — recurso e confiabilidade**
fração do horizonte bloqueada por downtime, concentração de downtime por máquina, saturação mínima por estágio, índice de escassez do recurso crítico.

**Grupo E — commodity e qualidade**
entropia de commodity, média e dispersão de umidade, share de cargas “wet”, densidade de compatibilidade `job × machine`.

**Grupo F — camada observacional e probes**
média e p95 do delta de `proc_time`, média e p95 do delta de due slack, congestion proxy, métricas baratas vindas do próprio FIFO (`fifo_mean_flow`, `fifo_p95_flow`, `fifo_makespan`). O survey de algorithm selection destaca justamente a importância de features informativas e baratas de computar. ([arXiv][4])

### 9.2 ISA propriamente dita

Use **ISA** como estrutura central de análise. A ISA foi criada para avaliar a diversidade de instâncias e modelar a relação entre propriedades estruturais e desempenho algorítmico; trabalhos recentes mostram sua utilidade para visualizar regiões fáceis, difíceis e discriminantes em problemas de otimização e scheduling. ([ScienceDirect][2])

O pipeline recomendado é:

1. padronização robusta das features;
2. filtro de correlação e eliminação de redundância;
3. projeção linear com **PCA** para a figura base, porque é reprodutível e interpretável;
4. projeção não linear com **UMAP** para recuperar estrutura local e global mais rica;
5. densidade local com kNN;
6. clustering/outlier detection com **HDBSCAN**;
7. sobreposição de desempenho dos métodos no espaço projetado;
8. construção de **solver footprints**;
9. identificação de regiões fáceis, difíceis e discriminantes;
10. proposta de expansão do benchmark a partir das lacunas observadas. ([arXiv][5])

### 9.3 Solver footprints

A figura mais valiosa do artigo não é um boxplot; é um mapa do espaço de instâncias com as regiões em que cada método fica “próximo do melhor”. Para isso:

* defina, para cada instância, a utilidade normalizada de cada método;
* marque como pertencente ao footprint de um método as instâncias em que ele está a até `ε` do melhor método;
* desenhe contornos por KDE, alpha-shapes ou envoltórias suaves;
* sobreponha regime, escala e dificuldade.

Isso torna o paper interpretável, explica “onde cada método ganha” e evita a pobreza analítica de médias globais. A ISA foi criada exatamente para esse tipo de leitura. ([Orca][6])

### 9.4 Classificação de instâncias para os métodos

Aqui entram duas tarefas.

**Tarefa A — dificuldade da instância**
Rotule cada instância como `easy`, `medium` ou `hard` com base na utilidade do melhor método disponível sob orçamento fixo.

**Tarefa B — melhor família de método**
Rotule cada instância com `best_method = argmin U(i,m)`, onde `U(i,m)` é a utilidade agregada do método `m` na instância `i`.

A forma mais limpa de definir `U(i,m)` é min-max normalizar as métricas por instância e usar, por exemplo:

`U(i,m) = 0.45*p95_flow_norm + 0.25*mean_flow_norm + 0.15*makespan_norm + 0.10*weighted_tardiness_norm + 0.05*runtime_norm`.

Isso força o artigo a privilegiar validade operacional e a tratar runtime como restrição importante, mas não dominante.

### 9.5 Modelos para classificação e seleção

Como o benchmark pai tem apenas 36 instâncias, a seleção instance-level deve ser tratada como **explicativa e exploratória**, não como produto final pronto. O conjunto mínimo de modelos é:

* árvore de decisão rasa, para mapa explicável;
* random forest, para baseline forte;
* XGBoost ou LightGBM, para melhor poder preditivo;
* kNN ou classificador sobre coordenadas UMAP, como baseline geométrico.

Use **SHAP** para explicar os atributos que levam uma instância a cair no território de cada método. Isso fecha o ciclo: ISA mostra a geografia, o classificador dá a fronteira, e o SHAP explica a fronteira. ([arXiv][7])

### 9.6 Exportação ASlib-style

Um diferencial muito forte é exportar seu benchmark para um cenário **ASlib-like**, com:

* `features.csv`
* `performance.csv`
* `runstatus.csv`
* `feature_costs.csv`
* `cv.arff` ou partições equivalentes

Isso coloca o benchmark dentro do padrão clássico de algorithm selection e facilita comparações futuras. ([ScienceDirect][8])

### 9.7 Extensão premium: classificação em nível de estado

Se houver tempo, faça também uma segunda tarefa de seleção, **não por instância, mas por estado da operação**. Cada época de decisão do replay vira uma amostra: fila visível, recursos livres, número de máquinas paradas, mix de umidade, prioridade, congestão etc. O label é o melhor método ou a melhor regra daquele instante. Essa extensão aproxima o trabalho da literatura recente de **dynamic dispatching rule selection**, que já compara seleção periódica e em tempo real em job shop dinâmico. ([ScienceDirect][9])

## 10. Experimentos

## E0. Reprodutibilidade e auditoria do benchmark

Objetivo: mostrar que o seu pipeline lê e replica corretamente o benchmark.

O que fazer: rodar `validate_observed_release.py`, `validate_benchmark.py` e reproduzir o FIFO oficial no seu próprio replay. Salve uma tabela com diferença absoluta e relativa entre `fifo_schedule`, `fifo_job_metrics`, `fifo_summary` oficiais e os produzidos pelo seu código. Isso entra como Figura ou Tabela de auditoria. ([GitHub][1])

## E1. Comparação principal entre métodos

Objetivo: comparar M0, M1, M2, M3 e, se existir, M4.

Protocolo: use a réplica `01` de cada uma das 12 famílias escala×regime como calibração e as réplicas `02` e `03` como teste. Isso dá 12 instâncias para tuning e 24 para avaliação final, mantendo cobertura de todas as famílias. ([GitHub][1])

Hipóteses:

* políticas reativas superam FIFO em `p95_flow` e `mean_flow`;
* os ganhos são maiores em `peak` e `disrupted`;
* `XS/S` tendem a favorecer reparo exato;
* `M` tende a favorecer híbridos;
* `L` tende a favorecer heurística/metaheurística ou reparo mais agressivamente limitado por orçamento.

## E2. Periódico versus disparado por eventos

Objetivo: responder a pergunta mais publicável da revisão.

Compare dois mecanismos com o mesmo subproblema interno:

* M2 periódico com `Δ ∈ {15, 30}` min;
* M3 disparado por evento.

Métricas: `p95_flow`, número de replans, latência de decisão, instabilidade do plano e custo operacional do replanejamento.

Justificativa: a sua revisão identifica essa comparação como fronteira importante, e a literatura recente de seleção dinâmica de regras em job shop mostra justamente que seleção periódica e em tempo real podem ter trade-offs diferentes, com robustez periódica e vantagem contextual do real-time dependendo do cenário.  ([ScienceDirect][9])

## E2b. Sensibilidade computacional a budget, threads e paralelismo

Objetivo: mostrar que o custo computacional não é um detalhe de implementação, mas parte do protocolo experimental e do comportamento observado dos métodos `M2`, `M3` e `Mref`.

Este experimento deve ser tratado como **estudo de sensibilidade computacional**, separado da comparação principal entre métodos. A comparação principal continua com configuração fixa de hardware e solver, enquanto este bloco mede como mudanças no orçamento computacional alteram tempo, gap, incumbent e qualidade operacional da solução. Isso é importante porque, em métodos baseados em MIP, `runtime`, `MIPGap`, política de threads e paralelismo externo podem alterar tanto o custo quanto a solução obtida sob `time limit`.

O desenho recomendado é um fatorial simples com três eixos. O primeiro eixo é `TimeLimit`, com pelo menos três orçamentos por escala: `short`, `medium` e `long`. O segundo eixo é `Threads ∈ {1, 2, 4, 8}` no solver. O terceiro eixo é o modo de execução do benchmark: `sequencial` por instância versus `paralelo` em lote, preservando a mesma configuração interna do solver. O objetivo não é “achar o melhor hardware”, mas medir robustez, saturação e sensibilidade do ranking dos métodos quando o orçamento computacional muda.

Para deixar esse experimento publicável e fácil de reproduzir, o paralelismo externo também deve ser parametrizado explicitamente. Em vez de registrar apenas “sequencial” ou “paralelo”, teste `n_workers ∈ {1, 2, 4, 8}` na campanha experimental, sempre documentando o hardware usado e a combinação `Threads × n_workers`. Isso permite separar dois efeitos que costumam ser confundidos: `i)` aceleração interna do solver via multithread; e `ii)` ganho de throughput do pipeline ao resolver múltiplas instâncias ou múltiplos replans em paralelo.

As hipóteses aqui devem ser explícitas. A primeira é que mais threads reduzem `runtime` até um ponto de saturação, mas não necessariamente preservam a mesma solução sob orçamento curto. A segunda é que o paralelismo externo melhora throughput de campanha experimental, mas pode degradar tempo por instância quando há contenção de CPU e memória. A terceira é que a vantagem relativa entre `M2` e `M3` pode depender do orçamento, especialmente em `M/L`, o que justifica tratar orçamento computacional como parte do método e não como detalhe pós-hoc.

As métricas mínimas desse experimento devem incluir: `runtime_sec`, `wall_clock_batch_sec`, `speedup_vs_1_thread`, `speedup_vs_1_worker`, `replan_count`, `solver_status`, `MIPGap`, `ObjVal`, `BestBd` quando disponível, utilidade agregada `U(i,m)`, `p95_flow_time`, `mean_flow_time` e um indicador de estabilidade da solução entre configurações computacionais. Uma forma simples de medir estabilidade é comparar, para a mesma instância e método, a fração de operações cujo recurso ou ordem muda quando `Threads`, `n_workers` ou `TimeLimit` são alterados.

O protocolo precisa ser claramente separado em dois níveis. No **protocolo principal**, fixe `Threads` e `TimeLimit` por escala, por exemplo `Threads=1` ou `Threads=2`, para garantir reprodutibilidade e comparabilidade do paper. No **protocolo de sensibilidade**, varie `Threads`, `TimeLimit` e modo de execução para quantificar elasticidade computacional. Assim, o artigo não confunde mérito algorítmico com abundância arbitrária de CPU.

Este experimento deve responder a cinco perguntas: `i)` quão sensível é a qualidade da solução ao orçamento de tempo; `ii)` em que ponto mais threads deixam de trazer ganho material; `iii)` em que ponto mais `workers` paralelos deixam de melhorar throughput e passam a gerar contenção; `iv)` se o ranking entre `M2`, `M3` e `Mref` muda quando o orçamento muda; e `v)` se o benchmark continua reproduzível em ambiente de CPU comum ou passa a depender de configuração agressiva de paralelismo. Para publicação, esse bloco é muito valioso porque antecipa uma objeção natural de reprodutibilidade e deixa claro que tempo de solução é parte da evidência experimental.

Além das médias agregadas por método e escala, o artefato computacional deve produzir **visões por instância**. No protocolo atual do notebook, isso aparece como: `i)` um heatmap de `runtime_sec` por `instância × método` em escala logarítmica para a comparação principal; `ii)` um heatmap de `runtime_sec` por `instância × (método, threads)` no budget `medium`; `iii)` um heatmap de `utility` por `instância × (método, threads)` no budget `medium`; e `iv)` um gráfico de dispersão com pontos mostrando o `runtime_sec` de cada execução e a condição em que ela terminou, por exemplo `optimal`, `feasible_time_limit`, `infeasible`, `error` ou outro `solver_status` equivalente. Nesse gráfico, a posição horizontal pode representar a configuração experimental ou a própria instância, a posição vertical o tempo de execução, e a cor ou o marcador o status final da execução. Essas visualizações são importantes porque evitam que a análise de custo computacional fique escondida por médias globais e mostram explicitamente heterogeneidade entre instâncias `XS/S/M/L`, além de deixar claro não apenas quanto tempo cada experimento levou, mas também onde ele terminou do ponto de vista computacional.

Como artefato suplementar, o paper deve mencionar que a implementação é entregue em um **notebook executável e reprodutível**, com tabelas e figuras renderizadas inline, para facilitar inspeção do pipeline completo sem depender de pós-processamento manual. Isso não substitui as figuras finais do artigo, mas fortalece transparência e auditoria reprodutível do estudo.

## E3. ISA, clustering e footprints

Objetivo: transformar desempenho em entendimento.

O que gerar:

* PCA com cores por regime;
* UMAP com cores por melhor método;
* HDBSCAN com clusters e outliers;
* mapa de footprints por método;
* mapa de dificuldade;
* regiões subamostradas.

Esse experimento precisa responder: “quais regiões do espaço favorecem exato, híbrido, metaheurística ou política reativa?” e “os rótulos `balanced/peak/disrupted` explicam tudo ou ainda existem subestruturas escondidas?” A resposta esperada é que os rótulos oficiais expliquem parte, mas não tudo. ([ScienceDirect][2])

## E4. Seletor de métodos

Objetivo: mostrar que a classificação das instâncias em regiões do espaço tem utilidade prática.

Modelos: árvore, random forest, gradient boosting.

Baselines:

* melhor método fixo global;
* seletor por escala simples (`XS/S→exact`, `M→hybrid`, `L→metaheuristic`);
* seletor por `scale+regime`.

Métricas:

* acurácia top-1;
* balanced accuracy para dificuldade;
* regret em relação ao oráculo;
* diferença para o melhor método fixo.

Com 36 instâncias, trate o resultado como explicativo. O seletor só vira claim forte quando você tiver estados de replay ou instâncias-filhas adicionais.

## E5. Validade do benchmark sintético

Objetivo: fortalecer o paper e mostrar maturidade metodológica.

O que fazer agora, sem dado real:

* `MMD` entre `core v1.0.0` e `observed v1.1.0` ou entre pais e filhos;
* `C2ST` para detectar diferenças multivariadas;
* `density ratio` para localizar regiões mal cobertas;
* scorecard no estilo `SynthEval/CAIR`;
* integridade relacional multiarquivo;
* identificação de duplicatas exatas e `duplicate-like`;
* checagem específica de caudas e segmentos raros. ([GitHub][3])

Observação crítica: sem holdout real, esses testes servem para robustez do benchmark e comparação entre versões, não para provar realismo empírico completo. ([GitHub][3])

## E6. Instâncias `graded` e `discriminating` (extensão)

Objetivo: preparar paper 2 ou apêndice forte.

A documentação do repo já recomenda expansão para famílias `graded` e `discriminating`. Isso deve ser feito depois de E3, usando justamente as lacunas do ISA:

* `graded`: uma escada clara de dificuldade;
* `discriminating`: regiões em que o ranking entre métodos se separa com nitidez. ([GitHub][3])

## 11. Métodos recentes que devem aparecer no artigo

Use esta combinação, porque ela dá entendimento “completo” das instâncias e dos métodos.

### 11.1 ISA

É o método central.

### 11.2 UMAP

Entra como projeção não linear principal, junto com PCA, porque ajuda a revelar estrutura local e regiões de transição entre famílias de instâncias. ([arXiv][5])

### 11.3 HDBSCAN

Entra para identificar clusters de densidade variável e outliers sem obrigar número fixo de clusters. É particularmente útil para separar regiões raras e detectar instâncias estranhas ou de fronteira. ([joss.theoj.org][10])

### 11.4 Solver footprints

É o elo entre ISA e benchmarking de método.

### 11.5 Empirical hardness models

Treine um modelo de regressão/classificação para prever dificuldade ou utilidade por método. Isso transforma ISA em ferramenta preditiva.

### 11.6 SHAP

Use SHAP para explicar os modelos de hardness/selection. Assim você consegue responder “por que este método ganha aqui?”. ([arXiv][7])

### 11.7 Performance profiles e curvas fixed-budget/fixed-target

Essas figuras são obrigatórias porque médias simples escondem dominância, robustez e sensibilidade a orçamento. ([arXiv][11])

### 11.8 MMD, C2ST e density ratio

Eles não classificam métodos de otimização, mas classificam a qualidade do benchmark como artefato experimental. Para um paper que quer ser metodologicamente forte, isso ajuda muito. ([JMLR][12])

### 11.9 SynthEval e CAIR

Use como disciplina de scorecard, não como fim em si. Eles são úteis para organizar checks em blocos claros e evitar diagnóstico ad hoc. ([arXiv][13])

## 12. Métricas do artigo

As métricas primárias devem ser:

* ganho relativo vs FIFO em `p95_flow_time`;
* ganho relativo vs FIFO em `mean_flow_time`;
* ganho relativo vs FIFO em `makespan`;
* `weighted tardiness` ou atraso ponderado em relação a `completion_due_min`;
* `p95` da latência de reotimização.

As métricas secundárias devem ser:

* fração de jobs acima do limite de espera;
* instabilidade do plano entre replans;
* número de replans por instância;
* `solver_status`, `MIPGap`, `ObjVal` e `BestBd` para métodos baseados em solver;
* `wall_clock_batch_sec` para campanhas sequenciais vs paralelas;
* `runtime_sec` por instância e por método, reportado também em visualização matricial;
* sensibilidade da solução a `Threads` e `TimeLimit`;
* `runtime_sec` e `utility` por instância sob variação de `Threads`, para detectar saturação heterogênea;
* índice de inversão FIFO;
* regret do seletor;
* cobertura do footprint por método;
* separabilidade entre clusters e estabilidade do clustering.

### Definições úteis

**Instabilidade do plano**
Fração de operações ainda não iniciadas que mudam de máquina ou posição entre duas reotimizações consecutivas.

**Índice de inversão FIFO**
Fração de pares comparáveis `(j,k)` com mesma prioridade em que o método atende `k` antes de `j`, embora `j` tenha chegado antes.

**Regret do seletor**
`Regret(i) = U(i, selector(i)) - min_m U(i,m)`.

## 13. Análise estatística

Para comparação de métodos no conjunto de teste, use:

* Friedman para verificar diferença global entre métodos;
* Wilcoxon pareado com correção de Holm para comparações pós-hoc;
* tamanho de efeito;
* intervalo bootstrap para mediana do ganho relativo;
* `performance profiles`;
* curvas `fixed-budget` e `fixed-target`.

Para selector e hardness:

* blocked cross-validation;
* balanced accuracy;
* macro-F1;
* regret médio e mediano;
* matriz de confusão entre `easy/medium/hard` e `best_method`.

Essas escolhas são consistentes com boas práticas de benchmarking em otimização e com a recomendação explícita do repositório para estabilidade de ranking, performance profiles e budgets múltiplos. ([Springer Nature Link][14])

Para o experimento de sensibilidade computacional, use uma análise complementar estratificada por escala e método. A forma mais limpa é reportar medianas, intervalos bootstrap e perfis de desempenho condicionados ao orçamento, além de gráficos de saturação `qualidade × threads` e `qualidade × time limit`. Se houver repetição com mesma configuração, reporte também variabilidade intra-configuração, porque em MIP sob limite de tempo a diferença entre incumbentes obtidos em configurações distintas é parte da conclusão.

Quando o orçamento computacional variar, complemente as curvas agregadas com heatmaps por instância. Isso ajuda a separar saturação global de saturação localizada: uma configuração pode parecer estável em média, mas ainda degradar fortemente um subconjunto pequeno de instâncias `M/L`. Para o artigo, isso é especialmente útil porque o benchmark tem apenas 36 instâncias-pai, então esconder variabilidade intra-conjunto é metodologicamente arriscado.

## 14. Figuras e tabelas que o artigo deve ter

**Tabela 1.** Benchmark e famílias de instância.
**Tabela 2.** Definição dos métodos e budgets.
**Tabela 3.** Grupos de features.
**Tabela 4.** Resultados agregados por método.
**Tabela 5.** Resultados do seletor e regret.
**Tabela 6.** Sensibilidade computacional a `Threads`, `TimeLimit` e modo de execução.
**Tabela 7.** Resumo por instância para o budget `medium`, com `runtime_sec`, `utility`, `solver_status` e `MIPGap` por configuração de `Threads`.

**Figura 1.** Pipeline do artigo.
**Figura 2.** PCA do espaço de instâncias.
**Figura 3.** UMAP com melhor método por região.
**Figura 4.** HDBSCAN com clusters e outliers.
**Figura 5.** Solver footprints.
**Figura 6.** Heatmap ganho vs FIFO por escala×regime.
**Figura 7.** Heatmap de `runtime_sec` por `instância × método` na comparação principal.
**Figura 8.** Performance profiles.
**Figura 9.** SHAP summary do seletor.
**Figura 10.** Scorecard de benchmark sintético.
**Figura 11.** Trade-off periódico vs evento.
**Figura 12.** Curvas `qualidade × budget` para `M2`, `M3` e `Mref`.
**Figura 13.** Curvas de saturação por número de threads.
**Figura 14.** Heatmap de `runtime_sec` por `instância × (método, threads)` no budget `medium`.
**Figura 15.** Heatmap de `utility` por `instância × (método, threads)` no budget `medium`.
**Figura 16.** Throughput sequencial versus paralelo por campanha.

## 15. Escopo mínimo publicável e escopo estendido

### Escopo mínimo publicável

1. Reproduzir FIFO.
2. Implementar M1, M2 e M3.
3. Rodar comparação nas 36 instâncias.
4. Gerar PCA/UMAP/HDBSCAN.
5. Gerar solver footprints.
6. Gerar performance profiles.
7. Escrever discussão de validade e threat model do benchmark.

Com isso sozinho, já existe paper.

### Escopo estendido de alto valor

1. Seletor instance-level.
2. Scorecard `MMD/C2ST/density ratio`.
3. Sensibilidade computacional a `Threads`, `TimeLimit` e paralelismo externo.
4. Estado-level rule selection.
5. Geração de instâncias `graded/discriminating`.

## 16. Estrutura do artigo em 12 páginas

**1. Introdução**
Problema, lacuna da revisão, contribuição do paper.

**2. Benchmark e formulação do problema**
Descrição do D-FJSP observacional e dos artefatos disponíveis.

**3. Métodos comparados**
FIFO, estático, periódico, evento, referência exata/híbrida/metaheurística.

**4. Features, ISA e classificação de instâncias**
Feature engineering, PCA, UMAP, HDBSCAN, footprints, selector.

**5. Protocolo experimental**
Split, budgets, métricas, estatística e sensibilidade computacional.

**6. Resultados de desempenho operacional**
Ganho vs FIFO e análise por regime.

**7. Resultados de ISA e footprints**
Regiões, clusters, dureza, lacunas.

**8. Resultados de seleção de métodos**
Regret, acurácia, SHAP.

**9. Benchmark quality e ameaças à validade**
Sintético, relacional, caudas, cautelas.

**10. Implicações para o agronegócio**
O que parece transferível e o que ainda depende de dado real/digital twin.

**11. Conclusão**
Mensagem principal e próximos passos.

## 17. Tarefas concretas no repositório

Crie os seguintes entregáveis.

### Prioridade alta

* `tools/extract_instance_features.py`
* `tools/run_policy_benchmark.py`
* `tools/build_method_matrix.py`
* `notebooks/01_fifo_reproduction.ipynb`
* `notebooks/02_method_benchmark.ipynb`
* `notebooks/03_instance_space_isa.ipynb`
* `catalog/instance_features.csv`
* `catalog/method_performance_matrix.csv`
* `output/article_figures/*`

### Prioridade média

* `tools/export_aslib_scenario.py`
* `tools/run_selector_cv.py`
* `notebooks/04_selector_shap.ipynb`
* `catalog/aslib_scenario/*`
* `catalog/scorecard_release_sbpo.csv`

### Prioridade condicional

* `tools/compare_versions_mmd_c2st_density.py`
* `tools/generate_child_instances_g2milp.py`
* `notebooks/05_state_level_selector.ipynb`

## 18. Riscos e como controlar

O principal risco é colapso de escopo. Os próprios documentos do time já decidiram, no contexto da hackathon, que levar o PequiFlux completo era inviável e que o recorte precisava ser mais enxuto. Para o artigo, a lição é a mesma: não misture benchmark D-FJSP com o storyline do Yard Copilot multimodal; mantenha o paper como paper de benchmarking e compreensão do espaço de instâncias.  

O segundo risco é overfitting do seletor, porque 36 instâncias parent ainda são poucas para claims preditivos fortes. Por isso, o seletor instance-level deve ser tratado como componente explicativo. O claim forte do paper continua sendo a comparação entre métodos e a leitura por ISA; o seletor mais robusto fica para a expansão com estados de replay ou instâncias-filhas. ([GitHub][1])

O terceiro risco é vender a base como se fosse “realista o suficiente para substituir dado operacional”. O próprio repositório proíbe essa leitura. O texto deve dizer que a base é **benchmark sintético observacional auditável**, útil para comparação de métodos e geração de hipóteses, mas não prova fidelidade empírica completa. ([GitHub][1])

O quarto risco é misturar mérito algorítmico com abundância de recursos computacionais. Se `Threads`, `TimeLimit` e paralelismo externo variarem entre métodos sem controle explícito, o artigo fica vulnerável a uma crítica metodológica básica: o ranking observado pode refletir orçamento de CPU e não apenas qualidade do método. Por isso, o protocolo principal deve fixar configuração computacional, e a variação de `Threads`, `TimeLimit` e modo de execução deve aparecer em experimento separado de sensibilidade computacional, com descrição completa do hardware e da configuração do solver.

## 19. O que não deve ser afirmado no artigo

Não diga:

* “o método reduz filas em plantas reais”;
* “o benchmark substitui dado operacional”;
* “o seletor está pronto para produção”;
* “o agronegócio já está resolvido pela literatura”.

Diga:

* “o artigo compara métodos em benchmark sintético observacional”;
* “o ISA explica onde cada método é forte”;
* “o benchmark já é informativo para comparação algorítmica inicial”;
* “os resultados geram hipóteses transferíveis e orientam validação futura com digital twin ou dado real”.  ([GitHub][3])

## 20. Template de abstract

**Resumo proposto com placeholders**

Este artigo investiga a orquestração de caminhões em um benchmark sintético observacional de pátio agroindustrial modelado como um D-FJSP. Comparamos quatro famílias de métodos — FIFO, regra estática orientada por prioridade, horizonte rolante periódico e reotimização disparada por eventos — complementadas por referências exatas/híbridas em escalas menores. Além da comparação operacional, propomos um pipeline de análise do espaço de instâncias combinando features estruturais, temporais e observacionais com PCA, UMAP, clustering e solver footprints. Os resultados mostram que [preencher], com ganhos mais fortes em [preencher] e trade-offs claros entre qualidade operacional e custo computacional. A análise do espaço de instâncias revela que [preencher], identificando regiões em que métodos exatos, híbridos e reativos são preferíveis, bem como lacunas de cobertura do benchmark. Como contribuição metodológica, o artigo apresenta um protocolo auditável para benchmarking, entendimento de dificuldade e seleção de métodos em cenários de recebimento agroindustrial, sem confundir evidência sintética com validação empírica em campo.

## 21. Frase final de contribuição para a introdução

**“A contribuição deste artigo não é propor mais um método vencedor em média, mas mostrar, em benchmark D-FJSP observacional e auditável, como diferentes regiões do espaço de instâncias favorecem diferentes famílias de métodos de orquestração de caminhões.”**

## 22. Base bibliográfica essencial para este desenho

Para ISA e dureza: Smith-Miles et al. sobre desempenho algorítmico ao longo do espaço de instâncias; Smith-Miles 2025 sobre metodologias de hardness; Sun et al. 2024 em car sequencing; Kletzander et al. 2021 em personnel scheduling. Para seleção automática de algoritmos: Kerschke et al. 2019 e ASlib. Para FJSP: Müller et al. 2022 mostram complementaridade entre solvers CP e seleção por features. Para scheduling dinâmico: Marques et al. 2025 sobre seleção dinâmica de regras. Para benchmarking: Dolan & Moré 2002, Bartz-Beielstein et al. 2020 e Dang et al. 2022 sobre instâncias graded/discriminating. Para avaliação do benchmark sintético: Gretton 2012, Lopez-Paz & Oquab 2016, Volker et al. 2024, Lautrup et al. 2025, Hyrup et al. 2024 e Hudovernik et al. 2024. ([ScienceDirect][2])

O melhor caminho agora é executar primeiro o **núcleo mínimo publicável**: E0, E1, E3 e performance profiles. Isso já gera um artigo crível, alinhado com a lacuna da revisão e com o estado real do repositório.

[1]: https://github.com/PequiFlux/agro_yard_dfjsp_benchmark_go "GitHub - PequiFlux/agro_yard_dfjsp_benchmark_go: Official seed dataset for the Agro Yard D-FJSP GO Benchmark v1.1.0-observed, frozen parent release for G2MILP-derived instances. · GitHub"
[2]: https://www.sciencedirect.com/science/article/pii/S0307904X2500040X "https://www.sciencedirect.com/science/article/pii/S0307904X2500040X"
[3]: https://github.com/PequiFlux/agro_yard_dfjsp_benchmark_go/blob/main/docs/synthetic_data_validation_next_steps.md "agro_yard_dfjsp_benchmark_go/docs/synthetic_data_validation_next_steps.md at main · PequiFlux/agro_yard_dfjsp_benchmark_go · GitHub"
[4]: https://arxiv.org/abs/1811.11597 "https://arxiv.org/abs/1811.11597"
[5]: https://arxiv.org/abs/1802.03426 "https://arxiv.org/abs/1802.03426"
[6]: https://orca.cardiff.ac.uk/id/eprint/53966/?utm_source=chatgpt.com "Towards objective measures of algorithm performance across ..."
[7]: https://arxiv.org/abs/1705.07874 "https://arxiv.org/abs/1705.07874"
[8]: https://www.sciencedirect.com/science/article/pii/S0004370216300388?utm_source=chatgpt.com "ASlib: A benchmark library for algorithm selection"
[9]: https://www.sciencedirect.com/science/article/pii/S0360835225006175 "https://www.sciencedirect.com/science/article/pii/S0360835225006175"
[10]: https://joss.theoj.org/papers/10.21105/joss.00205 "https://joss.theoj.org/papers/10.21105/joss.00205"
[11]: https://arxiv.org/abs/cs/0102001?utm_source=chatgpt.com "Benchmarking Optimization Software with Performance ..."
[12]: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf "https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf"
[13]: https://arxiv.org/abs/2312.12216 "https://arxiv.org/abs/2312.12216"
[14]: https://link.springer.com/article/10.1007/s101070100263?utm_source=chatgpt.com "Benchmarking optimization software with performance profiles"
