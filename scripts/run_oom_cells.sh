#!/bin/bash
# The 21 grid runs that OOM'd at fp32 on the 6 GB box, retried with fp16 autocast
# (training.amp=true). Resume-safe: skips runs with metrics.json.
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
LOGDIR="results/logs_remaining"
LOG="$LOGDIR/oom_queue.log"
declare -A MODEL=( [bros]=bros_base [lilt]=lilt_base )

run_one() {
  local bb="$1" sc="$2" me="$3" sd="$4"
  local run="${sc}_${me}_seed${sd}_${bb}"
  [ -f "results/${run}/metrics.json" ] && { echo "$(date +%H:%M) SKIP $run" >>"$LOG"; return; }
  # BROS has no gradient checkpointing; DER++ does 2 replay forwards, so replay batch 1
  # keeps it at ER's activation footprint (which fit at replay batch 2).
  local extra=""
  case "$me" in
    er) extra="method.replay_batch_size=2" ;;
    der_pp) [ "$bb" = bros ] && extra="method.replay_batch_size=1" || extra="method.replay_batch_size=2" ;;
  esac
  local tlimit=28800; case "$sc" in dil) tlimit=14400 ;; esac
  echo "$(date +%H:%M) START $run (amp)" >>"$LOG"
  timeout $tlimit uv run python scripts/train.py \
    method=$me scenario=$sc model=${MODEL[$bb]} seed=$sd \
    method.epochs=100 training.batch_size=1 training.num_workers=0 \
    training.gradient_checkpointing=true training.amp=true \
    wandb.mode=offline $extra > "$LOGDIR/${run}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run (rc=$rc)" >>"$LOG"; fi
}

echo "=== OOM CELLS (amp) — $(date) ===" >>"$LOG"
for sc in dil mixed cil_cord; do for sd in 42 7 123; do run_one bros $sc der_pp $sd; done; done
for sd in 42 7 123; do run_one lilt cil_cord ewc $sd; done
for sc in dil mixed cil_cord; do for sd in 42 7 123; do run_one lilt $sc lwf $sd; done; done
echo "=== OOM CELLS COMPLETE — $(date) ===" >>"$LOG"
