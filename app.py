"""
Hugging Face Spaces (ZeroGPU Free & CPU) + Cloud Entrypoint for Metal Predict.
Launches the FastAPI + Multi-Horizon ML server on port 7860 (Hugging Face default) or $PORT.
"""

from __future__ import annotations

import os

try:
    import spaces

    @spaces.GPU
    def _metal_predict_gpu_accelerator() -> str:
        return "Metal Predict Multi-Horizon Ensemble Active"
except Exception:
    spaces = None  # type: ignore

import uvicorn
from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, ingest_all_market_data
from backend.main import app

try:
    from spaces.zero.gradio import setup_middleware
    from spaces.zero.lifecycle import zerogpu_mode_enabled, zerogpu_startup

    setup_middleware(app)
    if zerogpu_mode_enabled():
        zerogpu_startup()
except Exception:
    pass

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_DB_PATH):
        ingest_all_market_data(db_path=DEFAULT_DB_PATH)
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
