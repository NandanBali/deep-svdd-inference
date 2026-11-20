#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Reuse an existing compatible local torch wheel without changing the system environment.
python3 -m venv --system-site-packages .venv-software
if ! .venv-software/bin/python -c 'import torch; assert torch.__version__.split("+")[0] == "2.12.0"'; then
  torch_index="https://download.pytorch.org/whl/cpu"
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    torch_index="https://download.pytorch.org/whl/cu130"
  fi
  .venv-software/bin/python -m pip install 'torch==2.12.0' --index-url "${SVDD_TORCH_INDEX_URL:-$torch_index}"
fi
.venv-software/bin/python -m pip install -r requirements-software.lock -e '.[test]'
.venv-software/bin/python -m pytest -q
.venv-software/bin/python -m svdd_baseline.run "$@"
