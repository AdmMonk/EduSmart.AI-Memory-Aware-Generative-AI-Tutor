FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV USE_HF_INFERENCE_API=true

# Build vector index on startup if missing, then serve API
CMD ["sh", "-c", "python scripts/build_index.py && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
