FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY pyproject.toml ./
COPY scripts/ ./scripts/

RUN uv venv /app/.venv
RUN uv sync --no-dev --extra core --no-install-project

COPY src/ ./src/
COPY proto/ ./proto/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv run scripts/gen_proto.py

RUN uv sync --no-dev --extra core
RUN uv cache clean

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="${PYTHONPATH}:/app" \
    UV_CACHE_DIR=/tmp/uv-cache

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
  CMD python scripts/healthcheck.py || exit 1

CMD ["serve"]
