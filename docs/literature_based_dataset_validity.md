# Validade do Dataset Segundo a Literatura

## Resposta curta

Segundo a literatura, um dataset sintético ou derivado como este **não deve ser declarado "válido" por um único teste**. A validação precisa combinar pelo menos quatro blocos:

1. **validade estrutural e relacional**
2. **fidelidade estatística e semântica**
3. **diversidade, cobertura e ausência de quase-duplicatas**
4. **utilidade downstream para a tarefa-alvo**

Aplicando esse padrão ao release atual, a leitura metodologicamente correta é:

- o dataset está **fortemente validado como benchmark sintético auditável e executável**
- o dataset está **parcialmente validado em realismo estatístico interno**
- o dataset **ainda não está validado em fidelidade externa ao processo real**, porque não há comparação contra `holdout` real

Em outras palavras: hoje já existe evidência forte para afirmar que a base é **coerente, consistente, diversa e útil como benchmark seed para G2MILP**, mas ainda **não** existe base suficiente para afirmar, segundo a literatura, que ela é uma boa aproximação empírica do pátio agroindustrial real.

## O que a literatura pede

### 1. Integridade estrutural e relacional

Para dados sintéticos, especialmente relacionais, não basta olhar marginais. É preciso checar chaves, cardinalidades, coerência entre tabelas e plausibilidade de joins e dependências cruzadas.

Isso está alinhado com:

- Gretton et al. (2012): testes globais de distribuição ajudam, mas não substituem checks estruturais
- Hudovernik, Jurkovič & Štrumbelj (2024): dados relacionais exigem avaliação além de uma tabela achatada

### 2. Fidelidade estatística

A literatura recomenda combinar comparação visual com testes formais de duas amostras:

- `MMD` para diferenças globais de distribuição
- `C2ST` para diferenças multivariadas detectáveis por classificador
- métricas separadas para fidelidade, diversidade e autenticidade

Isso está alinhado com:

- Gretton et al. (2012)
- Lopez-Paz & Oquab (2016)
- Alaa et al. (2022)

### 3. Diversidade e cobertura

Um dataset pode ser coerente e ainda assim ruim como benchmark se cobrir mal o espaço de instâncias ou repetir casos quase iguais. A literatura de synthetic data e de instance space analysis recomenda verificar redundância local, cobertura e separação entre famílias de instâncias.

### 4. Utilidade downstream

Na literatura, a pergunta final não é só "parece real?", mas também "serve para a tarefa?". Para synthetic data, isso costuma ser medido por desempenho em tarefas auxiliares; para benchmark de otimização, isso inclui executabilidade, gradação de dificuldade e capacidade de diferenciar métodos.

## O que o repositório já demonstra bem

### A. Validade estrutural: forte

O release atual já mostra evidência forte de validade estrutural:

- `36/36` instâncias passam na validação estrutural
- todo job tem `4` operações e `3` precedências
- toda operação tem ao menos uma máquina elegível
- há reconciliação perfeita entre `jobs.csv` e `job_noise_audit.csv`
- há reconciliação perfeita entre `eligible_machines.csv` e `proc_noise_audit.csv`
- o baseline FIFO respeita precedência, elegibilidade, `release_time`, downtime e ausência de overlap

Evidência local:

- [README.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/README.md#L73)
- [notebook_summary.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/notebook_summary.md#L3)
- [relational_consistency_summary.csv](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/relational_consistency_summary.csv)

Leitura segundo a literatura:

- isso sustenta bem a afirmação de que a base é **válida como objeto relacional e operacional**
- isso ainda **não** prova fidelidade externa ao processo real

### B. Fidelidade estatística interna: boa, mas parcial

O release também mostra sinais de que a camada observacional ficou menos determinística sem destruir a semântica operacional:

- `R²(due slack ~ priority)` caiu de `1.0000` para `0.4848`
- `R²(proc UNLOAD ~ load + machine + moisture)` caiu de `0.7540` para `0.4995`
- a ordem esperada `balanced < peak < disrupted` foi preservada para `avg_fifo_mean_flow_min`
- a ordem esperada `balanced < peak < disrupted` foi preservada para `avg_fifo_p95_flow_min`

Evidência local:

- [README.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/README.md#L119)
- [notebook_summary.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/notebook_summary.md#L11)
- [formal_shift_experiments_summary.csv](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/formal_shift_experiments_summary.csv)

Leitura correta:

- isso mostra que a versão `observed` deixou de ser artificialmente limpa
- isso é um argumento forte para **plausibilidade interna**
- isso continua sendo uma comparação **interna** (`nominal` versus `observed`), não uma prova de aderência ao processo real

### C. Diversidade e cobertura do espaço de instâncias: forte

O repositório já mostra evidência boa de que o dataset não é redundante:

- `exact_core_duplicate_count = 0`
- `exact_feature_duplicate_count = 0`
- `duplicate_like_candidate_count = 0`
- distância mínima ao vizinho mais próximo no espaço padronizado: `2.3228`

Evidência local:

- [notebook_summary.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/notebook_summary.md#L16)
- [instance_space_summary.csv](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/instance_space_summary.csv)

Leitura correta:

- isso sustenta bem a afirmação de que o benchmark tem **cobertura inicial razoável** e não parece uma coleção de casos repetidos
- esse é um argumento de **diversidade**, não de fidelidade externa

### D. Utilidade como benchmark de otimização: forte

Há evidência de que a base serve para benchmarkar métodos:

- os casos pequenos fecham no smoke test exato
- todos os casos testados têm solução
- os casos maiores mantêm gap não trivial sob o mesmo orçamento
- a escada de dificuldade por tamanho permanece visível

Evidência local:

- [README.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/README.md#L145)
- [notebook_summary.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/notebook_summary.md#L24)

Leitura correta:

- isso sustenta a tese de que o dataset é **válido como benchmark algorítmico**
- a literatura de benchmarking recomenda agora complementar isso com `performance profiles`, `fixed-budget` e `instance space analysis` mais orientada a solver

## O que impede afirmar "dataset validado" em sentido forte

### 1. Falta validação contra dados reais

Segundo a literatura, sem `holdout` real vocês não conseguem sustentar **fidelidade externa**. Hoje a base está validada contra:

- regras do domínio
- consistência interna
- sanidade operacional
- transformação nominal -> observed

Mas não contra:

- uma amostra real fora do conjunto de calibração

Esse é o principal limite metodológico do release atual.

### 2. Alguns sinais de warning nas caudas

O próprio notebook registra:

- `Mean congestion regime checks pass = False`
- `Tail flow p99 regime checks pass = False`
- `Tail queue p99 regime checks pass = False`
- `Tail due-margin p05 regime checks pass = False`

Evidência local:

- [notebook_summary.md](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/notebook_summary.md#L15)
- [tail_regime_summary.csv](/home/marcusvinicius/Repositorios/PequiFlux/agro_yard_dfjsp_benchmark_go/output/jupyter-notebook/instance_validation_analysis_artifacts/tail_regime_summary.csv)

Leitura correta:

- isso **não invalida** o benchmark como um todo
- mas impede uma afirmação forte de que a camada observacional está bem calibrada nas regiões mais raras ou extremas

## Como defender com gráficos, segundo a literatura

Se a pergunta for "como mostrar com gráficos que esta base é válida?", a resposta correta é montar a narrativa em **cinco painéis**, cada um sustentando uma dimensão diferente.

### Painel 1. Integridade estrutural e auditabilidade

Use:

- `structural_validation_and_auditability.png`
- `relational_consistency_overview.png`

Mensagem do gráfico:

- todas as instâncias passam
- as relações entre arquivos estão preservadas
- a base é consistente como objeto relacional

### Painel 2. Comportamento da camada observacional

Use:

- `observational_layer_behavior.png`
- `formal_shift_experiments.png`

Mensagem do gráfico:

- a versão `observed` deixou de ser excessivamente determinística
- ainda preserva sinais operacionais esperados
- a transformação foi substantiva, mas não caótica

### Painel 3. Sanidade operacional por regime

Use:

- `operational_performance_and_regime_sanity.png`
- `tail_and_rare_segments.png`

Mensagem do gráfico:

- no centro da distribuição, `balanced < peak < disrupted` se mantém
- nas caudas ainda há pontos que pedem refinamento

### Painel 4. Cobertura e diversidade do espaço de instâncias

Use:

- `instance_space_coverage.png`

Mensagem do gráfico:

- não há duplicatas exatas
- não há candidatos `duplicate-like`
- as famílias ocupam regiões distintas, com vizinhança local razoável

### Painel 5. Utilidade como benchmark

Use:

- `solver_oriented_smoke_test.png`
- `go_xs_disrupted_01_fifo_schedule.png`

Mensagem do gráfico:

- a base é executável
- preserva dificuldade não trivial
- produz cenários auditáveis e interpretáveis por instância

## Frase metodologicamente segura para usar

Se vocês forem descrever o status atual em texto técnico, a formulação segura é:

> O release `v1.1.0-observed` está validado como benchmark sintético relacional auditável, estruturalmente consistente, operacionalmente executável e com diversidade inicial adequada no espaço de instâncias. No entanto, à luz da literatura, sua fidelidade externa ao processo real ainda não está estabelecida, pois falta validação holdout-based contra dados reais e permanecem warnings em métricas de cauda.

## Próximo passo para chegar a um "sim" mais forte

O próximo passo com maior ganho científico é:

1. comparar contra um `holdout` real
2. calcular `alpha-precision`, `beta-recall` e `authenticity`
3. rodar `MMD` e `C2ST` em `real vs synthetic`
4. validar segmentos raros e caudas
5. adicionar avaliação downstream do tipo `TSTR/TRTS` ou tarefa operacional equivalente

## Referências

- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). *A Kernel Two-Sample Test*. JMLR. https://jmlr.org/beta/papers/v13/gretton12a.html
- Lopez-Paz, D., & Oquab, M. (2016). *Revisiting Classifier Two-Sample Tests*. arXiv. https://arxiv.org/abs/1610.06545
- Alaa, A., Van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022). *How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models*. ICML / PMLR. https://proceedings.mlr.press/v162/alaa22a.html
- Hudovernik, V., Jurkovič, M., & Štrumbelj, E. (2024). *Benchmarking the Fidelity and Utility of Synthetic Relational Data*. arXiv. https://arxiv.org/abs/2410.03411
- El Emam, K., Mosquera, L., Fang, X., & El-Hussuna, A. (2022). *Utility Metrics for Evaluating Synthetic Health Data Generation Methods: Validation Study*. JMIR. https://pmc.ncbi.nlm.nih.gov/articles/PMC9030990/
- Liu, C., Smith, J. M., Wauters, T., & Cattrysse, D. (2024). *Instance space analysis for 2D bin packing mathematical models*. EJOR. https://www.sciencedirect.com/science/article/pii/S0377221723009335
