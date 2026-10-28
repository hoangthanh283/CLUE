#!/bin/bash
# Tail of the fixed-12 matrix on the local box (plan rev. 9 + overnight authority 2026-10-22):
#   1. wait for TIER A COMPLETE (scripts/rca/final12_tierA.sh)
#   2. Mixed LCA x3 — ONLY if the disk-backed per-task store fix is in doccl/methods/lca.py
#      (marker "AGENT FIX: disk-backed"); otherwise they would OOM again at session 6.
#   3. the 18 image cells (CODA-Prompt, CL-LoRA, LCA x CIFAR-100 / ImageNet-R x 3 seeds) at the
#      house recipe (batch 16, fp32, grad-ckpt). run_vision_grid.sh skips any cell whose
#      metrics.json already exists, so cells the rented box delivers first are not redone.
# Serial, flock-guarded, resume-safe. Log: results/logs_vision/final12_tail.log
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/final12_tail.log
exec 9>/tmp/claude-1000/final12_tail.lock
flock -n 9 || { echo "tail queue already running" >&2; exit 1; }
mkdir -p results/logs_vision
until grep -q 'TIER A COMPLETE' results/logs_vision/final12_tierA.log 2>/dev/null; do sleep 120; done
echo "=== TAIL $(date) ===" >>"$LOG"

COMMON="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
if grep -q "AGENT FIX: disk-backed" doccl/methods/lca.py; then
  for sd in 42 7 123; do
    d="results/mixed_lca_seed${sd}"
    if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) SKIP $d" >>"$LOG"; continue; fi
    rm -rf "$d"   # stale tb/ from the OOM-killed attempt
    echo "$(date +%H:%M) START $d" >>"$LOG"
    uv run python scripts/train.py method=lca scenario=mixed model=layoutlmv3_base seed="$sd" $COMMON \
      >"results/logs_vision/mixed_lca_seed${sd}.log" 2>&1
    if [ -f "$d/metrics.json" ]; then
      echo "$(date +%H:%M) DONE  $d AA=$(python3 -c "import json;print(round(json.load(open('$d/metrics.json'))['AA'],2))")" >>"$LOG"
    else echo "$(date +%H:%M) FAIL  $d" >>"$LOG"; fi
  done
else
  echo "$(date +%H:%M) SKIP  mixed_lca x3 — lca.py memory fix not present" >>"$LOG"
fi

echo "$(date +%H:%M) IMAGE GRID start (house recipe bs16)" >>"$LOG"
SCENARIOS="cil_cifar100 cil_imagenet_r" METHODS="coda_prompt cl_lora lca" SEEDS="42 7 123" \
  BATCH_SIZE=16 GRAD_CKPT=true NUM_WORKERS=2 EPOCHS_CAP=100 bash scripts/run_vision_grid.sh
echo "$(date +%H:%M) IMAGE GRID end" >>"$LOG"
echo "=== TAIL COMPLETE $(date) ===" >>"$LOG"
