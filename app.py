"""
Hugging Face Spaces & Cloud Entrypoint for Metal Predict.
Launches the FastAPI + Multi-Horizon ML server on port 7860 (Hugging Face default) or $PORT.
"""

from __future__ import annotations

import os
import uvicorn
from backend.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    uvicorn.run(app, host="0.0.0.0", port=port)
