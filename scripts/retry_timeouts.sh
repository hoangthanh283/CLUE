#!/bin/bash
# Retry the 3 LiLT DER++ jobs that timed out at 4h.
# Using 8h (28800s) timeout — these are cil_cord (5 tasks) and mixed (3 tasks).
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

LOGDIR="results/logs_remaining"

run_one() {
  local sc="$1" sd="$2"
  local run="${sc}_der_pp_seed${sd}_lilt"
  local log="$LOGDIR/${run}.log"

  if [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) SKIP $run (done)"
    return 0
  fi

  echo "$(date +%H:%M) START $run (bs=1, timeout=28800s)"
  timeout 28800 uv run python scripts/train.py \
    method=der_pp scenario=$sc model=lilt_base seed=$sd \
    method.epochs=100 method.replay_batch_size=2 \
    training.batch_size=1 training.num_workers=0 \
    training.gradient_checkpointing=true \
    wandb.mode=offline \
    > "$log" 2>&1
  local rc=$?

  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) DONE  $run"
  else
    echo "$(date +%H:%M) FAIL  $run (rc=$rc)"
  fi
}

echo "=== RETRY TIMEOUTS — $(date) ==="
run_one cil_cord 42
run_one cil_cord 7
run_one mixed 7
echo "=== RETRY COMPLETE — $(date) ==="
