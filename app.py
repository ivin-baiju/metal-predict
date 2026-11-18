"""
Hugging Face Spaces (ZeroGPU Free & CPU) + Cloud Entrypoint for Metal Predict.
Launches the FastAPI + Multi-Horizon ML server on port 7860 (Hugging Face default) or $PORT.
"""

from __future__ import annotations

import os

try:
    import spaces
    from spaces.zero import client as _zg_client

    @spaces.GPU
    def _metal_predict_gpu_accelerator() -> str:
        return "Metal Predict Multi-Horizon Ensemble Active"

    if os.getenv("SPACES_ZERO_GPU"):
        _zg_client.startup_report()
except Exception as _zg_err:
    print("[Metal Predict] ZeroGPU startup note:", _zg_err, flush=True)

import uvicorn
from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, ingest_all_market_data
from backend.main import app

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_DB_PATH):
        ingest_all_market_data(db_path=DEFAULT_DB_PATH)
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
