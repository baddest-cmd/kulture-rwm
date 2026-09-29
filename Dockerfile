FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy configuration, application source code, and test suite
COPY configs/ ./configs/
COPY pyproject.toml .
COPY src/ ./src/
COPY tests/ ./tests/
COPY run_simulation_benchmark.py .
COPY train_world_model.py .

# Default entrypoint executes full PyTest suite inside container
CMD ["pytest", "-v"]
