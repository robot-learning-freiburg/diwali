#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.10 .venv
fi
if [ ! -d calvin_env ]; then
  git clone --recursive --branch lemon https://github.com/acl21/calvin_env calvin_env
fi
# RGB CALVIN evaluation needs no tactile sensor or optical-flow training stack.
uv sync --locked --inexact
uv pip install --python .venv/bin/python -e ./calvin_env
