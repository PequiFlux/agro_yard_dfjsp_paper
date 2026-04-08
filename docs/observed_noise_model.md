# Modelo de ruído observacional estruturado

## Escopo

Este documento descreve a camada observacional usada na variante `v1.1.0-observed` do benchmark Agro Yard D-FJSP GO. Esta variante foi registrada como tendo sido gerada com **ChatGPT 5.4 PRO** a partir do release nominal `v1.0.0`, com integração e revalidação local no repositório em `2026-03-27`.

O objetivo desta camada não é simular perfeitamente uma unidade real, e sim aproximar o benchmark de um dado observacional plausível sem perder:

- viabilidade estrutural
- compatibilidade com Gurobi
- auditabilidade linha a linha
- comparabilidade com o benchmark nominal

## Princípios de projeto

1. O release nominal continua sendo a referência.
2. Toda alteração observada precisa ser rastreável até um valor nominal.
3. O ruído é estruturado, não branco.
4. A semântica operacional do benchmark deve sobreviver à transformação.
5. A release só é aceita se passar em validações estruturais e comportamentais.

## O que muda e o que não muda

### Campos centrais alterados

- `jobs.csv::completion_due_min`
- `eligible_machines.csv::proc_time_min`

### Artefatos recalculados

- `fifo_schedule.csv`
- `fifo_job_metrics.csv`
- `fifo_summary.json`
- `catalog/benchmark_catalog.csv`
- `catalog/instance_family_summary.csv`

### Estrutura preservada

- conjunto de instâncias
- operações, precedências e elegibilidade estrutural
- compatibilidade commodity-moega
- indisponibilidades de máquina
- eventos `JOB_VISIBLE`, `JOB_ARRIVAL`, `MACHINE_DOWN`, `MACHINE_UP`

## Formulação metodológica equivalente

O texto metodológico usado para descrever esta release pode ser condensado na forma abaixo.

### Prazo observado

```text
slack_obs_j =
    b(priority_j)
  + f(appointment_j, commodity_j, moisture_j, shift_j, regime)
  + u_inst
  + u_shift(j)
  + eps_j
```

Depois do cálculo da folga observada:

```text
due_obs_j =
    arrival_j
  + clip(
        value = slack_obs_j,
        lower = LB_j + 18,
        upper = b(priority_j) + 120
    )
```

Onde:

- `b(priority_j)` é a folga-base por classe de prioridade
- `f(.)` agrega efeitos fixos pequenos e interpretáveis
- `u_inst` é o efeito latente da instância
- `u_shift(j)` é o efeito latente do turno do job
- `eps_j` é ruído Student-t com escala dependente do regime
- `LB_j` é o lower bound nominal do job, calculado como a soma dos menores tempos elegíveis de suas quatro operações

### Tempo de processamento observado

```text
eta_raw_jom =
    u_m
  + u_shift
  + u_stage_inst
  + u_regime
  + beta_stage * g_j
  + u_commodity
  + u_moisture
  + eps_jom

eta_clip_jom = clip(value = eta_raw_jom, lower = -0.28, upper = 0.42)

p_obs_jom =
    max(
        p_stage_min,
        round(p_nom_jom * exp(eta_clip_jom) + pause_jom)
    )
```

Onde:

- `u_m` é o efeito persistente da máquina na instância
- `u_shift` é o efeito do turno
- `u_stage_inst` é o efeito latente do estágio na instância
- `u_regime` representa o ambiente `balanced`, `peak` ou `disrupted`
- `g_j` é o proxy contínuo de congestionamento do job
- `u_commodity` e `u_moisture` são ajustes semânticos pequenos por estágio
- `eps_jom` é ruído idiossincrático Student-t
- `pause_jom` representa microparadas ocasionais

Essas duas expressões são semanticamente equivalentes ao que o código implementa em `tools/create_observed_noise_layer.py`.

## Modelo de prazo observado

Para cada job `j`, a folga observada é:

```text
slack_obs(j) =
    base(priority_j)
  + fixed(appointment_j, commodity_j, moisture_j, shift_j, regime)
  + u_instance
  + u_shift(shift_j)
  + eps_j
```

O prazo absoluto observado é:

```text
due_obs(j) =
    arrival_j
  + clip(value = slack_obs(j), lower = lower_j, upper = upper_j)
```

Com:

- `lower_j = nominal_processing_lb(j) + 18`
- `upper_j = base(priority_j) + 120`
- `eps_j ~ Student-t(df=5, scale_regime)`

Observação metodológica:

- a prioridade continua dominante
- a prioridade deixa de ser o mecanismo determinístico único
- o truncamento inferior evita prazos fisicamente implausíveis
- o truncamento superior evita que a camada observacional exploda a folga muito além da semântica original da classe

### Componente base por prioridade

| priority_class | base_slack_min |
| --- | ---: |
| `REGULAR` | 300 |
| `CONTRACTED` | 270 |
| `URGENT` | 240 |

### Efeitos fixos do modelo de prazo

#### Appointment

| appointment_flag | efeito_min |
| --- | ---: |
| `0` | 6 |
| `1` | -8 |

#### Commodity

| commodity | efeito_min |
| --- | ---: |
| `SOY` | 10 |
| `CORN` | 0 |
| `SORGHUM` | 5 |

#### Moisture

| moisture_class | efeito_min |
| --- | ---: |
| `DRY` | -5 |
| `NORMAL` | 0 |
| `WET` | 9 |

#### Shift

| shift_bucket | efeito_min |
| --- | ---: |
| `EARLY` | 8 |
| `MID` | 0 |
| `LATE` | -12 |
| `OVERNIGHT` | -20 |

#### Regime

| regime_code | efeito_min |
| --- | ---: |
| `balanced` | 0 |
| `peak` | 8 |
| `disrupted` | 18 |

### Componentes aleatórios do modelo de prazo

- desvio-padrão do efeito latente de instância: `8.0 min`
- desvio-padrão do efeito latente de turno: `4.0 min`
- ruído Student-t com:
  - `balanced`: `8.0 min`
  - `peak`: `10.0 min`
  - `disrupted`: `12.0 min`

## Modelo de tempo de processamento observado

Para cada tripla elegível `(job, op, machine)`:

```text
eta_raw =
    u_machine
  + u_shift
  + u_instance
  + u_regime
  + beta_stage * congestion
  + u_commodity
  + u_moisture
  + eps_idio

eta_clip = clip(value = eta_raw, lower = -0.28, upper = 0.42)

p_obs =
    max(
        p_min(stage),
        round(p_nom * exp(eta_clip) + pause)
    )
```

Onde:

- `eta_raw` é o deslocamento logarítmico antes do truncamento
- `eta_clip` é o deslocamento efetivamente aplicado ao tempo nominal
- `clip(value, lower, upper)` significa truncar o valor ao intervalo fechado `[lower, upper]`

### Piso mínimo por estágio

| stage | min_proc_min |
| --- | ---: |
| `WEIGH_IN` | 4 |
| `SAMPLE_CLASSIFY` | 6 |
| `UNLOAD` | 10 |
| `WEIGH_OUT` | 4 |

### Escalas dos efeitos latentes

| stage | sigma_machine | sigma_shift | sigma_instance | sigma_idio |
| --- | ---: | ---: | ---: | ---: |
| `WEIGH_IN` | 0.030 | 0.012 | 0.010 | 0.020 |
| `SAMPLE_CLASSIFY` | 0.045 | 0.025 | 0.020 | 0.030 |
| `UNLOAD` | 0.060 | 0.030 | 0.025 | 0.040 |
| `WEIGH_OUT` | 0.025 | 0.010 | 0.008 | 0.018 |

O termo idiossincrático de processamento usa ruído Student-t com `df = 6`.

Observação metodológica:

- o valor nominal continua sendo o centro da transformação
- a variabilidade é multiplicativa no log do tempo, não aditiva pura
- isso preserva melhor a interpretação de persistência operacional
- uma pausa aditiva pequena foi mantida para representar microparadas ocasionais

### Efeito log por regime

| regime | WEIGH_IN | SAMPLE_CLASSIFY | UNLOAD | WEIGH_OUT |
| --- | ---: | ---: | ---: | ---: |
| `balanced` | 0.000 | 0.000 | 0.000 | 0.000 |
| `peak` | 0.010 | 0.018 | 0.022 | 0.008 |
| `disrupted` | 0.020 | 0.030 | 0.040 | 0.015 |

### Sensibilidade a congestionamento

| stage | beta_congestion |
| --- | ---: |
| `WEIGH_IN` | 0.020 |
| `SAMPLE_CLASSIFY` | 0.035 |
| `UNLOAD` | 0.050 |
| `WEIGH_OUT` | 0.015 |

## Proxy de congestionamento

O congestionamento do job é calculado a partir dos tempos de chegada por um kernel triangular com largura de banda `60 min`.

Resumo do procedimento:

1. para cada job, mede-se a proximidade das chegadas vizinhas
2. agrega-se a contribuição triangular das demais chegadas
3. normaliza-se pelo percentil 95
4. trunca-se o resultado em `[0.0, 1.5]`

O resultado é gravado em `job_congestion_proxy.csv`.

Em termos intuitivos, `arrival_congestion_score` mede quão "apertada" está a vizinhança temporal de chegadas em torno de um job. Isso faz os tempos observados responderem a pressão local do sistema sem exigir um log operacional real.

## Efeitos semânticos por commodity e umidade

### Commodity

| stage | SOY | CORN | SORGHUM |
| --- | ---: | ---: | ---: |
| `WEIGH_IN` | 0.000 | 0.000 | 0.000 |
| `SAMPLE_CLASSIFY` | 0.020 | 0.000 | 0.010 |
| `UNLOAD` | 0.010 | 0.000 | 0.015 |
| `WEIGH_OUT` | 0.000 | 0.000 | 0.000 |

### Moisture

| stage | DRY | NORMAL | WET |
| --- | ---: | ---: | ---: |
| `WEIGH_IN` | 0.000 | 0.000 | 0.000 |
| `SAMPLE_CLASSIFY` | -0.020 | 0.000 | 0.045 |
| `UNLOAD` | -0.010 | 0.000 | 0.028 |
| `WEIGH_OUT` | 0.000 | 0.000 | 0.000 |

## Microparadas

### Probabilidade de pausa por regime e estágio

| regime | WEIGH_IN | SAMPLE_CLASSIFY | UNLOAD | WEIGH_OUT |
| --- | ---: | ---: | ---: | ---: |
| `balanced` | 0.03 | 0.05 | 0.08 | 0.02 |
| `peak` | 0.05 | 0.08 | 0.13 | 0.04 |
| `disrupted` | 0.08 | 0.12 | 0.18 | 0.06 |

### Pausa máxima aditiva

| stage | pause_max_min |
| --- | ---: |
| `WEIGH_IN` | 2 |
| `SAMPLE_CLASSIFY` | 3 |
| `UNLOAD` | 5 |
| `WEIGH_OUT` | 2 |

## Arquivos de auditoria

### Por instância

- `job_noise_audit.csv`
- `proc_noise_audit.csv`
- `job_congestion_proxy.csv`
- `noise_manifest.json`

### No nível do release

- `catalog/observed_noise_manifest.json`
- `catalog/validation_report_observed.csv`
- `catalog/noise_diagnostics.json`
- `catalog/noise_diagnostics_before_after.json`
- `catalog/noise_validation_summary.md`

## Rastreabilidade em dois níveis

### Nível release

O arquivo `catalog/observed_noise_manifest.json` registra:

- `model_id`
- `global_seed`
- hiperparâmetros da camada observacional
- relação com o dataset pai
- papel desta release como base oficial para geração futura com G2MILP

### Nível instância

Em cada `params.json`, a instância registra:

- `dataset_version`
- `parent_dataset_version`
- `observational_noise_model_id`
- `observational_noise_seed`

Além disso, cada instância contém:

- `noise_manifest.json` com efeitos latentes sorteados
- `job_noise_audit.csv` com decomposição do prazo observado
- `proc_noise_audit.csv` com decomposição do tempo observado
- `job_congestion_proxy.csv` com o congestionamento usado na transformação

Em termos práticos, isso permite auditar linha a linha de onde saiu cada valor observado.

## O que exatamente pode ser auditado

### Em `job_noise_audit.csv`

Para cada `job_id`, o audit registra:

- prazo nominal
- prazo observado
- folga nominal
- folga observada
- efeito base da prioridade
- efeitos fixos por appointment, commodity, moisture, turno e regime
- componente aleatório de instância
- componente aleatório de turno
- ruído idiossincrático
- lower bound de processamento nominal

### Em `proc_noise_audit.csv`

Para cada `(job_id, op_seq, machine_id)`, o audit registra:

- `proc_time_nominal_min`
- `proc_time_observed_min`
- efeitos latentes de máquina, turno e instância
- efeito de regime
- efeito de congestionamento
- efeito de commodity
- efeito de moisture
- ruído idiossincrático
- pausa aditiva

## Critérios de aceitação da release

A release só deve ser tratada como válida quando todos os critérios abaixo forem satisfeitos.

### 1. Integridade estrutural

- cada job continua com 4 operações
- cada job continua com 3 precedências
- toda operação continua tendo ao menos uma máquina elegível
- nenhum prazo observado fica abaixo de `job_noise_audit.csv::nominal_processing_lb_min + 18`
- o baseline FIFO permanece sem overlap por máquina
- `fifo_schedule.csv` e `fifo_job_metrics.csv` permanecem consistentes com `eligible_machines.csv`

### 2. Reconciliação exata dos audits

- `jobs.csv::completion_due_min` deve coincidir com `job_noise_audit.csv::completion_due_observed_min`
- `eligible_machines.csv::proc_time_min` deve coincidir com `proc_noise_audit.csv::proc_time_observed_min`

### 3. Redução de sobre-determinismo

Os diagnósticos agregados observados foram:

| métrica | nominal v1.0.0 | observed v1.1.0 |
| --- | ---: | ---: |
| `R²(due slack ~ priority)` | 1.0000 | 0.4848 |
| `R²(UNLOAD proc ~ load + machine + moisture)` | 0.7540 | 0.4995 |

Esses valores indicam que o benchmark ficou menos determinístico sem perder interpretabilidade.

Em linguagem simples: a transformação de fato reduziu o excesso de determinismo, mas sem "implodir" a semântica do benchmark.

### 4. Preservação da semântica operacional

No sumário por famílias:

- `avg_fifo_mean_flow_min`: `balanced < peak < disrupted` em todas as escalas
- `avg_fifo_p95_flow_min`: `balanced < peak < disrupted` em todas as escalas

## Blocos de validação recomendados

Os testes mais úteis para qualquer revalidação desta release ou de futuros filhos derivados são estes quatro.

### 1. Integridade estrutural

Verificar:

- 4 operações por job
- precedências válidas
- nenhuma operação sem máquina elegível
- tempos positivos e plausíveis
- baseline sem overlap nem inconsistência com recurso e downtime

### 2. Reconciliação nominal-observado

Verificar:

- `jobs.csv::completion_due_min == job_noise_audit.csv::completion_due_observed_min`
- `eligible_machines.csv::proc_time_min == proc_noise_audit.csv::proc_time_observed_min`

### 3. Validação comportamental por regime

Verificar se `PEAK` e `DISRUPTED` continuam piores que `BALANCED` em:

- fluxo médio
- cauda de fluxo

### 4. Diagnóstico de sobre-determinismo

Recalcular os diagnósticos globais e confirmar:

- queda do `R²` da folga explicada só por prioridade
- queda do `R²` do tempo de `UNLOAD` explicado por `load_tons + machine_id + moisture_class`
- preservação da coerência operacional do benchmark

## Quando rejeitar a release

Ela deve ser rejeitada se:

- algum teste estrutural falhar
- a reconciliação audit vs arquivo central falhar
- os diagnósticos não mostrarem queda de determinismo
- a semântica por regime colapsar
- a integração quebrar os loaders ou o notebook comparativo

## Resultados observados nesta release

No release oficial integrado, os resultados relevantes ficaram assim:

- `36/36` instâncias aprovadas na validação estrutural
- `R²(due slack ~ priority) = 0.4848` contra `1.0000` no nominal
- `R²(UNLOAD proc ~ load + machine + moisture) = 0.4995` contra `0.7540` no nominal
- `avg_fifo_mean_flow_min` preservando `balanced < peak < disrupted`
- `avg_fifo_p95_flow_min` preservando `balanced < peak < disrupted`

Isso reforça a interpretação de que a release ficou menos artificial sem deixar de ser semanticamente coerente.

## Por que isso ajuda em problemas mais próximos da vida real

Esta camada observacional é útil porque insere:

- variabilidade intraclasse de prazo
- persistência por máquina e turno
- sensibilidade a congestionamento de chegada
- assimetria por commodity e umidade
- microparadas ocasionais

Esse conjunto é relevante para testar:

- robustez de heurísticas e metaheurísticas
- estabilidade de modelos exatos sob tempos menos ideais
- regras de despacho e rescheduling em cenários `peak` e `disrupted`
- sensibilidade de métricas operacionais a ruído estruturado

## Reprodução da derivação

Para reproduzir a derivação a partir de um release nominal em outro caminho:

```bash
python tools/create_observed_noise_layer.py \
  --src-root /caminho/para/agro_yard_dfjsp_benchmark_go_v1 \
  --out-root /caminho/para/agro_yard_dfjsp_benchmark_go_v1_1_observed
```

Para revalidar o release derivado:

```bash
python tools/validate_observed_release.py \
  /caminho/para/agro_yard_dfjsp_benchmark_go_v1_1_observed
```

## Checagem manual de rastreabilidade

Uma verificação manual direta pode ser feita em qualquer instância, por exemplo `GO_XS_BALANCED_01`.

1. Escolha um `job_id` em `job_noise_audit.csv`.
2. Verifique que `completion_due_observed_min` coincide com `jobs.csv::completion_due_min`.
3. Escolha uma tripla `(job_id, op_seq, machine_id)` em `proc_noise_audit.csv`.
4. Verifique que `proc_time_observed_min` coincide com `eligible_machines.csv::proc_time_min`.

Se essas reconciliações falharem, o ciclo de auditoria da release está quebrado.

## Limitação correta de interpretação

Esta variante melhora o realismo do benchmark, mas não deve ser vendida como substituta de dado real. O uso metodologicamente correto é:

- benchmark nominal para referência controlada
- benchmark observado para robustez e plausibilidade
- log real, quando existir, para validação externa final
