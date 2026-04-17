#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${HOME:-/home/paper}" /workspace/tmp/jupyter-notebook

if [[ ! -f "${GRB_LICENSE_FILE:-/licenses/gurobi.lic}" ]]; then
  echo "Missing Gurobi license file at ${GRB_LICENSE_FILE:-/licenses/gurobi.lic}" >&2
  exit 1
fi

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  echo "Missing JUPYTER_TOKEN. Set a strong token before starting paper-notebook." >&2
  exit 1
fi

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.root_dir=/workspace \
  --LabApp.default_url=/lab/tree/output/jupyter-notebook \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.allow_origin='*'
