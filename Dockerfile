FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


WORKDIR /app

COPY pyproject.toml ./
COPY scripts/ ./scripts/

RUN uv venv /app/.venv
RUN uv sync --no-dev --extra core --no-install-project

COPY src/ ./src/

RUN uv sync --no-dev --extra core
RUN uv cache clean

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    UV_CACHE_DIR=/tmp/uv-cache

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
  CMD python scripts/healthcheck.py || exit 1

CMD ["serve"]
