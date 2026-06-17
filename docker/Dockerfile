# DocCL grid runner — CUDA + Python 3.12 + uv.
#
# Builds a reproducible image that runs the heavy Core (54) + Prompt/LoRA (36) grid
# on a powerful GPU box, offloaded from the 6 GB laptop. Training protocol matches
# the local driver: val-F1 early stopping (patience=2, cap=100ep) + per-run FWT.
#
# torch in uv.lock uses bundled CUDA 13 wheels (nvidia-*-cu13), so we only need a
# CUDA-13-capable host driver; the CUDA runtime ships inside the wheels. We use a
# slim CUDA base for nvidia-smi + driver compatibility.
#
# Build:  docker build -t doccl-grid -f docker/Dockerfile .
# Run:    see docker/README.md (needs --gpus all + a WANDB_API_KEY + a results volume)

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.hf_cache

# System deps: python3.12 toolchain + git (HF datasets) + libs for image/OCR handling
# + rclone (durable resume sync to object storage on ephemeral/on-demand instances).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 rclone \
    && rm -rf /var/lib/apt/lists/*

# uv (fast, lockfile-reproducible installs).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Install deps FIRST (cached layer) from the lockfile — only invalidated when deps change.
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv --python 3.12 \
    && uv sync --frozen --no-install-project --extra dev
ENV PATH="/opt/venv/bin:$PATH"

# Then copy the project source (changes here don't reinstall deps).
COPY . .
RUN uv sync --frozen --extra dev

# Sanity: import the package + confirm torch sees CUDA at build time is NOT possible
# (no GPU in build), so we only verify the import graph here.
RUN python -c "import doccl, torch; print('doccl import OK; torch', torch.__version__)"

# Default: run the remote E2E grid (baselines -> core -> prompt -> aggregate).
# Override CMD to run a subset, e.g.  ... PHASES="core" bash scripts/run_grid_remote.sh
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["bash", "scripts/run_grid_remote.sh"]
