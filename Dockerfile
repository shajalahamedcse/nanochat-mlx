# tinychat-mlx inference server
#
# IMPORTANT: MLX requires Apple Silicon (Metal GPU). This image packages the
# server and its dependencies. Inference only works when run on Apple Silicon
# hardware. Running inside Docker on macOS is NOT supported by MLX (no Metal
# access from within the container).
#
# Recommended usage — run the server natively:
#   python -m scripts.serve --depth=4 --source=sft
#
# This Dockerfile is useful for:
#   - CI / testing (no GPU inference)
#   - Distributing the app as a container to Apple Silicon hosts
#   - Running on future hardware that exposes Metal to containers

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (mlx will install but won't use Metal inside Docker)
RUN uv sync --no-dev

# Copy source
COPY tinychat_mlx/ tinychat_mlx/
COPY scripts/      scripts/
COPY tasks/        tasks/

# Weights are mounted at runtime — do not bake them into the image
VOLUME ["/app/weights"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: serve depth=4 sft model, weights mounted at /app/weights
CMD ["uv", "run", "python", "-m", "scripts.serve", \
     "--depth=4", "--source=sft", \
     "--host=0.0.0.0", "--port=8000"]
