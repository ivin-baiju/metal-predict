"""
Hugging Face Spaces & Cloud Entrypoint for Metal Predict.
Launches the FastAPI + Multi-Horizon ML server on port 7860 (Hugging Face default) or $PORT.
"""

from __future__ import annotations

import os
import uvicorn
from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, ingest_all_market_data
from backend.main import app

try:
    import spaces  # Provided by Hugging Face ZeroGPU runtime

    @spaces.GPU
    def _metal_predict_gpu_accelerator() -> str:
        return "Metal Predict Multi-Horizon Ensemble Active"
except Exception:
    pass

if __name__ == "__main__":
    if not os.path.exists(DEFAULT_DB_PATH):
        ingest_all_market_data(db_path=DEFAULT_DB_PATH)
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
