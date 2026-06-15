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

# SECURITY: we intentionally do NOT write any secret to disk. Creds (WANDB_API_KEY,
# R2_ACCESS_KEY_ID/SECRET/ENDPOINT) arrive as runtime `-e` env vars and are read straight
# from the process environment by the scripts — they never persist on the (rented,
# possibly-reused) instance disk. The training scripts `source .env 2>/dev/null || true`,
# so an absent .env is harmless; we create only a NON-SECRET .env with the W&B project name
# so that line is a clean no-op without leaking anything.
[ -f .env ] || printf 'WANDB_PROJECT=%s\n' "${WANDB_PROJECT:-CL4IE}" > .env

if [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
  echo "  durable-resume: R2 creds present in env (not written to disk) -> sync ON (bucket ${R2_BUCKET:-doccl-results})"
else
  echo "  durable-resume: no R2_* creds -> sync OFF (results on ephemeral disk only)"
fi

exec "$@"
