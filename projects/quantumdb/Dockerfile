FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml first for better caching
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e .[dev,evaluation]

# Copy source code
COPY quantumdb/ ./quantumdb/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p models data notebooks

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "quantumdb.server:app", "--host", "0.0.0.0", "--port", "8000"]