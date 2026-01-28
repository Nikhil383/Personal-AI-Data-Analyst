# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for pandas/matplotlib)
# libgl1-mesa-glx is often needed for opencv or some plotting libs, but maybe not basic matplotlib
# simpler is better for now.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY static ./static
COPY templates ./templates

# Create virtual environment and install dependencies
# We install the project in editable mode or just dependencies
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies from pyproject.toml
RUN uv pip install .

# Expose Flask port
EXPOSE 5000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app.py"]
