#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
echo "Starting Kigo Metals Full-Stack App at http://127.0.0.1:8000/tools/metal-price-predictor.html"
exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
