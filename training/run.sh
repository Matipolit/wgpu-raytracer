#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv training/.venv
training/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r training/requirements.lock
exec training/.venv/bin/python -u training/overnight.py "$@"
