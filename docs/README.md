# Documentação Canônica da Release

Este diretório reúne a documentação metodológica versionada da release `v1.1.0-observed`.

Os documentos oficiais desta release são os listados em `manifest.json::documentation_files`. No estado atual do repositório, o conjunto canônico é:

- `README.md`: visão geral do benchmark, uso rápido, validação e artefatos principais
- `docs/observed_noise_model.md`: especificação da camada observacional aplicada sobre a base nominal
- `docs/g2milp_generation_contract.md`: contrato de derivação e linhagem para datasets filhos gerados com `G2MILP`
- `docs/synthetic_data_validation_next_steps.md`: backlog metodológico de validação e fortalecimento do benchmark

Regra de empacotamento:

- somente arquivos listados no manifesto devem ser tratados como documentação oficial da release
- arquivos locais fora dessa lista podem existir no workspace, mas não devem ser citados como parte do pacote publicado sem versionamento explícito
