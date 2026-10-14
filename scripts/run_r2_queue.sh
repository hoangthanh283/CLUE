#!/bin/bash
# R2: continual-NER comparators + CoLaR (+ label refresh) on the two scenarios with headroom.
# Waits for the Independent grid to release the single GPU slot, then runs sequentially.
# Resume-safe: skips any cell whose metrics.json exists. Log: results/logs_vision/r2_queue.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2_queue.log
mkdir -p results/logs_vision
GRID_LOG=results/logs_vision/indep_queue.log
until grep -q 'INDEPENDENT GRID COMPLETE' "$GRID_LOG" 2>/dev/null; do sleep 120; done
echo "=== R2 QUEUE $(date) ===" >>"$LOG"

COLAR_AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
COMMON="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
SEEDS="${SEEDS:-42 7 123}"
SCENARIOS="${SCENARIOS:-cil_cord mixed}"

run() {  # run <dir> <method> <scenario> <seed> <bs> [extra overrides]
  local d="results/$1" method=$2 sc=$3 sd=$4 bs=$5; shift 5
  if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) SKIP $1" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $1" >>"$LOG"
  uv run python scripts/train.py method=$method scenario=$sc model=layoutlmv3_base seed=$sd \
    training.batch_size=$bs $COMMON "$@" >"results/logs_vision/$1.log" 2>&1
  [ -f "$d/metrics.json" ] && echo "$(date +%H:%M) DONE  $1" >>"$LOG" || echo "$(date +%H:%M) FAIL  $1" >>"$LOG"
}

for sd in $SEEDS; do
  for sc in $SCENARIOS; do
    run ${sc}_colar_seed${sd}_k4_d50_r128          colar          $sc $sd 2 $COLAR_AX
    run ${sc}_colar_r_oracle_seed${sd}_k4_d50_r128 colar_r_oracle $sc $sd 2 $COLAR_AX
    run ${sc}_colar_r_pseudo_seed${sd}_k4_d50_r128 colar_r_pseudo $sc $sd 2 $COLAR_AX
    run ${sc}_cpfd_seed${sd}                       cpfd           $sc $sd 1
    run ${sc}_is3_seed${sd}                        is3            $sc $sd 1
  done
done
echo "=== R2 QUEUE COMPLETE $(date) ===" >>"$LOG"
