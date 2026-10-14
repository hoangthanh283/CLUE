#!/bin/bash
# R2 retry: the two CIL-CORD seed-42 cells that died on the wrapper `.classifier` AttributeError
# (fixed by the `classifier` property on the wrappers, AGENT FIX 2026-10-15). Later cpfd/is3
# cells in the R2 queue start fresh processes and pick the fix up on their own.
# Waits for the R2c queue (single GPU slot). Resume-safe. Log: results/logs_vision/r2_retry.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2_retry.log
mkdir -p results/logs_vision
until grep -q 'R2c QUEUE COMPLETE' results/logs_vision/r2c_queue.log 2>/dev/null; do sleep 300; done
echo "=== R2 RETRY $(date) ===" >>"$LOG"
COMMON="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
for method in cpfd is3; do
  name="cil_cord_${method}_seed42"
  if [ -f "results/$name/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py method=$method scenario=cil_cord model=layoutlmv3_base seed=42 \
    training.batch_size=1 $COMMON >"results/logs_vision/$name.log" 2>&1
  [ -f "results/$name/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
done
echo "=== R2 RETRY COMPLETE $(date) ===" >>"$LOG"
