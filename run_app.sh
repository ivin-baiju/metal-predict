#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "Starting Metal Predict Full-Stack App at http://127.0.0.1:8000/"
exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
