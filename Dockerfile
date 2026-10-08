# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency specifications first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into virtual environment
RUN uv sync --frozen --no-dev --extra agents

# Runner stage
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

# Copy installed virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code and files
COPY src ./src
COPY pyproject.toml README.md ./

# Create runtime directories for data and outputs
RUN mkdir -p /app/data /app/outputs

EXPOSE 8000

# Run FastAPI app with uvicorn listening on $PORT
CMD ["sh", "-c", "uvicorn data_analyst.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
