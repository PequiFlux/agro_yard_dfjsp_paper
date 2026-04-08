# Agro Yard D-FJSP Paper Workspace

Este diretório é o workspace de desenvolvimento do paper derivado do benchmark
`agro_yard_dfjsp_benchmark_go`.

Ele existe para separar:

- manuscrito, revisão e narrativa do artigo
- experimentos computacionais e seleção de métodos
- resultados, figuras e tabelas reproduzíveis

da base oficial congelada do benchmark, que continua no repositório original.

## O que foi trazido do benchmark

Foram copiados para este workspace os insumos necessários para escrever,
executar e reproduzir o paper:

- `catalog/`
- `docs/`
- `gurobi/`
- `instances/`
- `tools/`
- `output/jupyter-notebook/`
- `manifest.json`

Esses arquivos funcionam aqui como snapshot operacional para pesquisa e
experimentação. A referência canônica do benchmark continua sendo o repositório
de origem.

## Estrutura de trabalho

- `paper/Artigo.md`: desenho experimental e metodológico do artigo
- `paper/assets/`: figuras, tabelas e anexos do manuscrito
- `experiments/`: novos códigos e pipelines do paper
- `results/`: saídas reproduzíveis dos experimentos do paper
- `catalog/`, `instances/`, `tools/`, `output/`: insumos copiados do benchmark

## Regra prática

Se uma mudança for:

- correção ou evolução da base oficial do benchmark, ela deve nascer no
  repositório original
- implementação específica do paper, análise ISA, seleção de métodos, figuras
  ou texto, ela deve nascer aqui

## Próximos passos sugeridos

1. Implementar um engine de replay em `experiments/`
2. Reproduzir o baseline FIFO oficial como primeiro check
3. Versionar as métricas comparativas em `results/`
4. Migrar o texto do paper de `Artigo.md` para a estrutura final do manuscrito

## Ambiente em container

O repositório agora inclui um ambiente reprodutível para o notebook do paper,
com JupyterLab, Gurobi e a pilha analítica necessária para ISA, seleção de
métodos e validação headless.

Arquivos principais:

- `Dockerfile`
- `compose.yaml`
- `requirements/paper.lock.txt`
- `docker/check_gurobi.py`
- `docker/run_headless_notebook.py`
- `.env.example`

Configuração esperada:

- a licença Gurobi fica fora da imagem e é montada em runtime
- o caminho padrão é `./gurobi.lic`
- o serviço de notebook expõe JupyterLab na porta `8888`

Fluxo de uso:

```bash
docker compose build
docker compose run --rm paper-check
docker compose up paper-notebook
```

Execução headless do notebook:

```bash
docker compose run --rm paper-exec
```

Se o notebook principal do paper ainda não existir, o runner executa um
notebook mínimo de sanity check do ambiente para validar kernel, imports e
stack analítico dentro do container.

Variáveis de ambiente suportadas:

- `GUROBI_LICENSE_FILE`
- `JUPYTER_TOKEN`
- `GUROBI_MAX_CONCURRENT_MODELS`
- `NOTEBOOK_PATH`
- `NOTEBOOK_EXECUTION_TIMEOUT`
- `UID`
- `GID`
