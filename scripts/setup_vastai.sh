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

# 3. W&B login (OPTIONAL — the grid runs OFFLINE by default)
echo
echo "→ W&B is OPTIONAL: scripts/run_grid.sh defaults to WANDB_MODE=offline and"
echo "  scripts/analyze_results.py aggregates from local results/ (no W&B needed)."
echo "  Log in only if you want live dashboards (then run with WANDB_MODE=online):"
wandb login || echo "  (Skipped — running fully offline; results come from results/*/metrics.json)"

# 4. HuggingFace datasets cache
echo
echo "→ Pre-downloading datasets to HF cache..."
python -c "
from datasets import load_dataset
print('Downloading FUNSD...')
load_dataset('nielsr/funsd-layoutlmv3', trust_remote_code=True)
print('Downloading CORD-v2...')
load_dataset('naver-clova-ix/cord-v2', trust_remote_code=True)
print('Done.')
"
echo "  SROIE: either prepare locally —"
echo "      python scripts/prepare_sroie.py --raw_dir <SROIE_raw> --output_dir data/sroie"
echo "  — or use the HF mirror (source='hf' in doccl/data/sroie.py; confirm its BIO scheme)."

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
