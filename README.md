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
- `output/jupyter-notebook/agro-yard-paper-benchmark-and-selection.ipynb`: notebook principal que implementa o desenho do paper em fluxo único reproduzível
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

## Ótimos Gurobi por instância

Tabela de acompanhamento dos resultados obtidos com Gurobi usando a instância
completa, `TimeLimit=60s`, `Threads=1`, `workers externos=1` e execução
`sequencial` sem paralelismo externo. Quando `Soluções = 0`, o solver não
encontrou incumbente dentro do orçamento, então `Ótimo Gurobi`, `Best bound` e
`Gap` ficam em branco.

| Instância | Escala | Regime | Réplica | TimeLimit (s) | Threads solver | Cores solver | Workers externos | Paralelismo externo | Modo batch | Status | Soluções | Jobs | Ops | Variáveis | Restrições | Ótimo Gurobi (makespan) | Best bound | Gap | Runtime solver (s) | Wall time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GO_XS_BALANCED_01 | XS | balanced | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 4 | 18 | 72 | 1804 | 3380 | 629.000000 | 629.000000 | 0.000000 | 0.271413 | 0.272162 |
| GO_XS_BALANCED_02 | XS | balanced | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 1 | 18 | 72 | 1850 | 3464 | 617.000000 | 617.000000 | 0.000000 | 0.018717 | 0.019423 |
| GO_XS_BALANCED_03 | XS | balanced | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 4 | 18 | 72 | 1834 | 3434 | 677.000000 | 677.000000 | 0.000000 | 0.076455 | 0.077148 |
| GO_XS_DISRUPTED_01 | XS | disrupted | 1 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 10 | 24 | 96 | 3223 | 6140 | 634.000000 | 625.000000 | 0.014196 | 60.000952 | 60.002234 |
| GO_XS_DISRUPTED_02 | XS | disrupted | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 10 | 24 | 96 | 3189 | 6074 | 627.000000 | 627.000000 | 0.000000 | 2.400370 | 2.401631 |
| GO_XS_DISRUPTED_03 | XS | disrupted | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 10 | 24 | 96 | 3168 | 6034 | 638.000000 | 638.000000 | 0.000000 | 2.159222 | 2.160500 |
| GO_XS_PEAK_01 | XS | peak | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 7 | 24 | 96 | 3181 | 6050 | 642.000000 | 642.000000 | 0.000000 | 0.858345 | 0.859730 |
| GO_XS_PEAK_02 | XS | peak | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 6 | 24 | 96 | 3175 | 6038 | 663.000000 | 663.000000 | 0.000000 | 0.438135 | 0.439672 |
| GO_XS_PEAK_03 | XS | peak | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 8 | 24 | 96 | 3100 | 5900 | 655.000000 | 655.000000 | 0.000000 | 0.578134 | 0.579413 |
| GO_S_BALANCED_01 | S | balanced | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 1 | 30 | 120 | 5794 | 11082 | 692.000000 | 692.000000 | 0.000000 | 0.138241 | 0.140778 |
| GO_S_BALANCED_02 | S | balanced | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 6 | 30 | 120 | 5752 | 11004 | 686.000000 | 686.000000 | 0.000000 | 3.025911 | 3.028365 |
| GO_S_BALANCED_03 | S | balanced | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 1 | 30 | 120 | 5806 | 11102 | 688.000000 | 688.000000 | 0.000000 | 0.152569 | 0.155001 |
| GO_S_DISRUPTED_01 | S | disrupted | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 4 | 40 | 160 | 10161 | 19650 | 560.000000 | 560.000000 | 0.000000 | 1.060468 | 1.064804 |
| GO_S_DISRUPTED_02 | S | disrupted | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 7 | 40 | 160 | 10151 | 19642 | 581.000000 | 581.000000 | 0.000000 | 10.589514 | 10.593856 |
| GO_S_DISRUPTED_03 | S | disrupted | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 9 | 40 | 160 | 10186 | 19708 | 589.000000 | 589.000000 | 0.000000 | 1.870183 | 1.874746 |
| GO_S_PEAK_01 | S | peak | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 6 | 40 | 160 | 10047 | 19430 | 612.000000 | 612.000000 | 0.000000 | 2.067710 | 2.072214 |
| GO_S_PEAK_02 | S | peak | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 1 | 40 | 160 | 10084 | 19502 | 677.000000 | 677.000000 | 0.000000 | 0.325154 | 0.329682 |
| GO_S_PEAK_03 | S | peak | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 10 | 40 | 160 | 10098 | 19526 | 596.000000 | 596.000000 | 0.000000 | 2.873097 | 2.877604 |
| GO_M_BALANCED_01 | M | balanced | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 10 | 48 | 192 | 19265 | 37490 | 667.000000 | 667.000000 | 0.000000 | 11.426142 | 11.434269 |
| GO_M_BALANCED_02 | M | balanced | 2 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 8 | 48 | 192 | 19193 | 37350 | 693.000000 | 693.000000 | 0.000000 | 8.457785 | 8.466278 |
| GO_M_BALANCED_03 | M | balanced | 3 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 9 | 48 | 192 | 19610 | 38164 | 665.000000 | 665.000000 | 0.000000 | 7.014080 | 7.022621 |
| GO_M_DISRUPTED_01 | M | disrupted | 1 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 3 | 64 | 256 | 34567 | 67732 | 767.000000 | 559.000000 | 0.271186 | 60.002299 | 60.017008 |
| GO_M_DISRUPTED_02 | M | disrupted | 2 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 10 | 64 | 256 | 34180 | 66972 | 615.000000 | 577.000000 | 0.061789 | 60.015432 | 60.029887 |
| GO_M_DISRUPTED_03 | M | disrupted | 3 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 10 | 64 | 256 | 34660 | 67918 | 589.000000 | 580.000000 | 0.015280 | 60.007864 | 60.023498 |
| GO_M_PEAK_01 | M | peak | 1 | 60 | 1 | 1 | 1 | não | sequencial | OPTIMAL | 10 | 64 | 256 | 34289 | 67180 | 641.000000 | 641.000000 | 0.000000 | 52.759937 | 52.774551 |
| GO_M_PEAK_02 | M | peak | 2 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 10 | 64 | 256 | 34372 | 67344 | 658.000000 | 643.000000 | 0.022796 | 60.007457 | 60.021857 |
| GO_M_PEAK_03 | M | peak | 3 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 10 | 64 | 256 | 34294 | 67192 | 685.000000 | 636.000000 | 0.071533 | 60.004482 | 60.019031 |
| GO_L_BALANCED_01 | L | balanced | 1 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 72 | 288 | 57321 | 112550 |  |  |  | 60.007759 | 60.031967 |
| GO_L_BALANCED_02 | L | balanced | 2 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 72 | 288 | 56628 | 111178 |  |  |  | 60.001579 | 60.024692 |
| GO_L_BALANCED_03 | L | balanced | 3 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 72 | 288 | 57213 | 112334 |  |  |  | 60.005726 | 60.029441 |
| GO_L_DISRUPTED_01 | L | disrupted | 1 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100784 | 198796 |  |  |  | 60.005790 | 60.047636 |
| GO_L_DISRUPTED_02 | L | disrupted | 2 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100640 | 198522 |  |  |  | 60.021942 | 60.064279 |
| GO_L_DISRUPTED_03 | L | disrupted | 3 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100860 | 198946 |  |  |  | 60.001619 | 60.043647 |
| GO_L_PEAK_01 | L | peak | 1 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100464 | 198184 |  |  |  | 60.002196 | 60.045977 |
| GO_L_PEAK_02 | L | peak | 2 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100711 | 198652 |  |  |  | 60.002052 | 60.044643 |
| GO_L_PEAK_03 | L | peak | 3 | 60 | 1 | 1 | 1 | não | sequencial | TIME_LIMIT | 0 | 96 | 384 | 100675 | 198588 |  |  |  | 60.009918 | 60.052473 |

### Ambiente computacional da campanha

- Modelo de CPU do host: `AMD Ryzen 7 7700 8-Core Processor`
- Número total de núcleos lógicos da máquina: `16`
- Núcleos físicos: `8`
- RAM do host: `30 GiB`
- Sistema operacional do host: `Fedora Linux 43 (Workstation Edition)`
- Kernel do host: `Linux 6.19.12-200.fc43.x86_64`
- Docker: `29.4.0`
- Docker Compose: `5.1.1`
- Python no container de execução: `3.12.13`
- Gurobi no container de execução: `12.0.1`
- Plataforma reportada no container: `Linux-6.19.12-200.fc43.x86_64-x86_64-with-glibc2.36`

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
- o caminho recomendado da licença é fora da árvore do repositório, por exemplo `~/.config/gurobi/gurobi.lic`
- `GUROBI_LICENSE_FILE` deve apontar explicitamente para esse arquivo antes de subir os serviços
- o serviço de notebook expõe JupyterLab na porta `8888`
- `JUPYTER_TOKEN` deve ser definido explicitamente com um valor forte antes de subir `paper-notebook`

Fluxo de uso:

```bash
cp .env.example .env
export GUROBI_LICENSE_FILE="$HOME/.config/gurobi/gurobi.lic"
export JUPYTER_TOKEN="defina-um-token-forte"
docker compose build
docker compose run --rm paper-check
docker compose up paper-notebook
```

Execução headless do notebook:

```bash
docker compose run --rm paper-exec
```

O alvo padrão do runner headless é:

- `output/jupyter-notebook/agro-yard-paper-benchmark-and-selection.ipynb`

Se o notebook principal do paper ainda não existir, o runner executa um
notebook mínimo de sanity check do ambiente para validar kernel, imports e
stack analítico dentro do container.

Variáveis de ambiente suportadas:

- `GUROBI_LICENSE_FILE`: caminho absoluto da licença Gurobi fora do repositório
- `JUPYTER_TOKEN`: token obrigatório para expor o JupyterLab
- `GUROBI_MAX_CONCURRENT_MODELS`
- `NOTEBOOK_PATH`
- `NOTEBOOK_EXECUTION_TIMEOUT`
- `UID`
- `GID`
