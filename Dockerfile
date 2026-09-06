# syntax=docker/dockerfile:1.7

# ---- builder: resolve & install dependencies with uv ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (without the project) so this layer is cached
# and only re-runs when uv.lock / pyproject.toml change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then copy the source and install the project itself into the venv.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bake the embedding model into the image. Otherwise fastembed downloads it to
# a temp directory on the first request that needs an embedding — ~130MB, lost
# on every container restart, and a hard failure if HuggingFace is unreachable.
ARG EMBEDDING_MODEL_ID="BAAI/bge-small-en-v1.5"
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN uv run python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='${EMBEDDING_MODEL_ID}')"

# ---- runtime: slim image carrying just the venv + source ----
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

# CA certificates for TLS to MongoDB (e.g. Atlas).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user rather than root.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
WORKDIR /app

COPY --from=builder --chown=app:app /app /app
COPY --from=builder --chown=app:app /opt/fastembed_cache /opt/fastembed_cache

USER app

EXPOSE 8181

# Liveness probe hits the root endpoint; no curl needed in the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8181/', timeout=3)" || exit 1

# --no-server-header suppresses the `server:` response header, which otherwise
# names the ASGI server on every response.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8181", "--no-server-header"]
