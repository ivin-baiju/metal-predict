FROM python:3.12-slim

# OpenMP runtime required by XGBoost and LightGBM on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY data_pipeline ./data_pipeline
COPY ml_engine ./ml_engine
COPY scripts ./scripts
COPY kigo.studio ./kigo.studio

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
