#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m svdd_features.workflow --download "$@"
