#!/usr/bin/env bash
# Vast.ai instance bootstrap script.
# Run inside a fresh pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime container.
#
# Usage on host (after SSH-ing into Vast.ai instance):
#   curl -O https://raw.githubusercontent.com/<your-fork>/doccl/main/scripts/setup_vastai.sh
#   bash setup_vastai.sh
#
# Or after `git clone`:
#   bash scripts/setup_vastai.sh

set -euo pipefail

echo "=== DocCL Vast.ai setup ==="
echo "Date: $(date)"
nvidia-smi || { echo "WARNING: nvidia-smi failed — GPU may not be available"; }

# 1. System tools
echo
echo "→ Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git wget curl unzip vim htop tmux \
    ca-certificates build-essential

# 2. Python deps
echo
echo "→ Installing Python dependencies..."
pip install --upgrade pip wheel

# Install in editable mode (assumes we are inside the cloned repo)
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev]"
else
    echo "WARNING: pyproject.toml not found in $(pwd) — install manually."
fi

# 3. W&B login (interactive — paste API key when prompted)
echo
echo "→ W&B login (paste your API key from https://wandb.ai/authorize)"
echo "  Skip with Ctrl-C if you'll set WANDB_API_KEY env var instead."
wandb login || echo "  (Skipped — set WANDB_API_KEY env var to log results)"

# 4. HuggingFace datasets cache
echo
echo "→ Pre-downloading datasets to HF cache..."
python -c "
from datasets import load_dataset
print('Downloading FUNSD...')
load_dataset('nielsr/funsd-layoutlmv3', trust_remote_code=True)
print('Downloading CORD-v2...')
load_dataset('naver-clova-ix/cord-v2', trust_remote_code=True)
print('Done. SROIE must be prepared via scripts/prepare_sroie.py.')
"

# 5. Smoke tests
echo
echo "→ Running smoke tests..."
pytest tests/test_smoke.py -v -m "not slow and not gpu" || {
    echo "Smoke tests failed — investigate before running full experiments."
    exit 1
}

echo
echo "✓ Setup complete. Quick checks:"
echo "  - GPU:           nvidia-smi"
echo "  - Python:        python --version"
echo "  - DocCL install: python -c 'import doccl; print(doccl.__version__)'"
echo "  - W&B project:   set WANDB_PROJECT=doccl-aaai2027"
echo
echo "Next: bash scripts/run_pilot.sh"
