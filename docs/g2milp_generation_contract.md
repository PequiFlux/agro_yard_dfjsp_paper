# Contrato de derivação para G2MILP

## Status

Esta release `v1.1.0-observed` é o **dataset pai congelado** para geração de novas instâncias com modelos da família **G2MILP**.

Ela deve ser tratada como referência oficial para:

- expandir o número de instâncias
- gerar novas réplicas ou famílias de regime
- produzir datasets derivados para comparação metodológica
- testar variações de modelagem sem perder linhagem para a base oficial

## O que o G2MILP pode derivar

O uso esperado é gerar novas instâncias ou novos cenários a partir desta base preservando a semântica central do problema:

- jobs com quatro operações obrigatórias
- precedência linear entre operações
- elegibilidade de máquina coerente com o estágio
- compatibilidade por commodity no descarregamento
- coerência entre tempos, eventos, baseline e métricas

## O que deve permanecer rastreável

Toda instância gerada por G2MILP deve registrar, no mínimo:

- `parent_dataset_name`
- `parent_dataset_version`
- `parent_instance_id`
- `generator_model_family`
- `generator_model_name`
- `generator_model_version`
- `generation_seed`
- `generation_timestamp`
- `generation_notes`

Se houver transformação estrutural relevante, também deve registrar:

- `transformation_scope`
- `changed_core_fields`
- `preserved_constraints`
- `post_generation_validation_status`

## Campos-base que devem ser tratados como origem

Os arquivos centrais desta release oficial que devem servir como fonte primária para geração são:

- `instances/*/jobs.csv`
- `instances/*/operations.csv`
- `instances/*/precedences.csv`
- `instances/*/eligible_machines.csv`
- `instances/*/machine_downtimes.csv`
- `instances/*/events.csv`
- `instances/*/params.json`
- `catalog/benchmark_catalog.csv`
- `catalog/instance_family_summary.csv`

Os arquivos de auditoria desta release também devem ser preservados como contexto metodológico:

- `instances/*/job_noise_audit.csv`
- `instances/*/proc_noise_audit.csv`
- `instances/*/job_congestion_proxy.csv`
- `instances/*/noise_manifest.json`
- `catalog/observed_noise_manifest.json`
- `catalog/noise_diagnostics_before_after.json`
- `catalog/validation_report_observed.csv`

## Política de derivação

### Permitido

- gerar novas réplicas
- ampliar escalas
- gerar novos regimes
- recalibrar ruído ou parâmetros por meio de um pipeline G2MILP
- gerar cenários para experimentos de robustez

### Não permitido sem nova versão formal

- sobrescrever esta release pai
- apagar a linhagem com a `v1.1.0-observed`
- publicar filhos sem manifesto de geração
- alterar sem registro os campos estruturais centrais do problema

## Requisitos mínimos de validação para filhos G2MILP

Qualquer dataset filho gerado a partir desta base deve passar em pelo menos estes testes:

1. Cada job continua com quatro operações obrigatórias, salvo se a mudança de estrutura for declarada formalmente.
2. Toda operação continua com pelo menos uma máquina elegível.
3. As precedências permanecem acíclicas e coerentes.
4. Os tempos continuam positivos e fisicamente plausíveis.
5. O baseline ou um verificador equivalente continua livre de overlap por máquina.
6. Eventos continuam reconciliáveis com chegada, visibilidade e indisponibilidade.
7. O manifesto do filho preserva a referência explícita a esta release pai.

## Recomendação de metadado para datasets filhos

Um manifesto de dataset filho derivado por G2MILP deve conter uma estrutura equivalente a:

```json
{
  "dataset_name": "Agro Yard D-FJSP GO Benchmark - G2MILP Derived",
  "dataset_version": "x.y.z",
  "parent_dataset_name": "Agro Yard D-FJSP GO Benchmark",
  "parent_dataset_version": "1.1.0-observed",
  "parent_release_role": "base_dataset_for_g2milp_instance_generation",
  "generator_model_family": "G2MILP",
  "generator_model_name": "NOME_DO_MODELO",
  "generator_model_version": "VERSAO_DO_MODELO",
  "generation_seed": 12345,
  "generation_timestamp": "2026-03-27",
  "generation_notes": "Resumo curto da transformação aplicada."
}
```

## Interpretação correta

Esta base oficial não deve ser tratada como dado real. O uso metodologicamente correto é:

- usar esta release como pai congelado
- gerar filhos G2MILP com lineage explícito
- validar cada filho estrutural e comportamentalmente
- comparar filhos com esta base oficial antes de qualquer alegação de ganho metodológico
