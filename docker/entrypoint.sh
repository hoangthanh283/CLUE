#!/usr/bin/env bash
# Container entrypoint: verify GPU + W&B, then exec the requested command (default:
# scripts/run_grid_remote.sh). Keeps the image generic so CMD/args can override.
set -uo pipefail

echo "=== DocCL grid container ==="
echo "torch CUDA available:"
python -c "import torch; print('  cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())" || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  (nvidia-smi unavailable — did you pass --gpus all?)"

if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-online}" = "online" ]; then
  echo "WARNING: WANDB_API_KEY not set and WANDB_MODE=online — runs will fail to log."
  echo "         Pass -e WANDB_API_KEY=... or set -e WANDB_MODE=offline."
fi

# Materialise a minimal .env if not mounted, so source .env in the scripts is harmless.
[ -f .env ] || cat > .env <<EOF
WANDB_API_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=${WANDB_PROJECT:-CL4IE}
WANDB_ENTITY=${WANDB_ENTITY:-}
EOF

exec "$@"
