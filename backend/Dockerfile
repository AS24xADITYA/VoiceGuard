FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./

# Install CPU-only torch first to prevent pulling multi-GB CUDA libraries
RUN pip install --no-cache-dir --prefix=/install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0,<2.5.0"
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ ./app/
COPY ai/ ./ai/
COPY alembic/ ./alembic/
COPY alembic.ini ./

RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/artifacts /app/data && \
    chown -R appuser:appuser /data /app

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
