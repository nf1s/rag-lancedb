# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .python-version ./

RUN useradd --uid 1000 --create-home app

# --------------------------------------------------------------------------- store
FROM base AS store

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra store

COPY src ./src

RUN chown -R app:app /app
USER app

EXPOSE 8081
CMD ["uvicorn", "store.main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]

# --------------------------------------------------------------------------- api
FROM base AS api

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api

ENV FASTEMBED_CACHE_PATH=/models

# Bake the embedding model into the image so pods do not download it on cold start.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /models

COPY src ./src

RUN chown -R app:app /app
USER app

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# --------------------------------------------------------------------------- ui
FROM base AS ui

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui

COPY ui ./ui

RUN chown -R app:app /app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.runOnSave=true"]
