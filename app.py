"""
Hugging Face Spaces (ZeroGPU Free & CPU) + Cloud Entrypoint for Metal Predict.
Launches the FastAPI + Multi-Horizon ML server on port 7860 (Hugging Face default) or $PORT.
"""

from __future__ import annotations

import os
import threading
import time

try:
    import spaces

    @spaces.GPU
    def _metal_predict_gpu_accelerator() -> str:
        return "Metal Predict Multi-Horizon Ensemble Active"
except Exception:
    spaces = None  # type: ignore

import httpx
import uvicorn
from data_pipeline.ingest_market_data import DEFAULT_DB_PATH, ingest_all_market_data
from backend.main import app


def _zerogpu_handshake_loop() -> None:
    """Send POST /startup-report to the ZeroGPU supervisor once uvicorn is listening on port 7860."""
    api_url = os.getenv("SPACES_ZERO_DEVICE_API_URL")
    if not api_url:
        return
    url = api_url.rstrip("/") + "/startup-report"
    for _ in range(40):
        try:
            res = httpx.post(url, timeout=5.0, verify=False)
            if res.status_code == 200:
                print("[Metal Predict] ZeroGPU startup-report handshake succeeded (200 OK).", flush=True)
                return
        except Exception:
            pass
        time.sleep(0.75)


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_DB_PATH):
        ingest_all_market_data(db_path=DEFAULT_DB_PATH)
    if os.getenv("SPACES_ZERO_DEVICE_API_URL"):
        threading.Thread(target=_zerogpu_handshake_loop, daemon=True).start()
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
