#!/bin/bash
# R2 retry 2: re-run the Mixed CPFD / IS3 seed-42 cells after the teacher-snapshot fix
# (AGENT FIX 2026-10-15: both methods snapshotted the teacher AFTER head expansion, so KD
# distilled the random-init new rows; mixed_cpfd_seed42 = 6.0 AA was invalid, mixed_is3_seed42
# was killed mid-run). Waits for the R2d queue (last job in the single-GPU chain). Resume-safe.
# Log: results/logs_vision/r2_retry2.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2_retry2.log
mkdir -p results/logs_vision
until grep -q 'R2d QUEUE COMPLETE' results/logs_vision/r2d_queue.log 2>/dev/null; do sleep 300; done
echo "=== R2 RETRY2 $(date) ===" >>"$LOG"
COMMON="training.batch_size=1 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
for m in cpfd is3; do
  name="mixed_${m}_seed42"
  if [ -f "results/$name/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py method=$m scenario=mixed model=layoutlmv3_base seed=42 $COMMON \
    >"results/logs_vision/$name.log" 2>&1
  [ -f "results/$name/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
done
echo "=== R2 RETRY2 COMPLETE $(date) ===" >>"$LOG"
