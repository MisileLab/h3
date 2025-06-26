# Use official Python base image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY pyproject.toml ./
COPY uv.lock ./
COPY train.py ./

# Sync Python dependencies using uv
RUN uv sync

# Expose marimo default port
EXPOSE 8888

# Command to run marimo in edit mode on train.py as the main script
CMD ["marimo", "edit", "train.py", "--host", "0.0.0.0", "--port", "8888"]

