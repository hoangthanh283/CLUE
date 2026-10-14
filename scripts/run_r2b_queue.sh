#!/bin/bash
# R2b: mechanism-isolating controls for CoLaR-R (tournament P1). Waits for the R2 queue to
# finish (single GPU slot), then runs the sliced-softmax control on both scenarios, three seeds.
# Resume-safe. Log: results/logs_vision/r2b_queue.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2b_queue.log
mkdir -p results/logs_vision
until grep -q 'R2 QUEUE COMPLETE' results/logs_vision/r2_queue.log 2>/dev/null; do sleep 300; done
echo "=== R2b QUEUE $(date) ===" >>"$LOG"

COLAR_AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
COMMON="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
SEEDS="${SEEDS:-42 7 123}"
SCENARIOS="${SCENARIOS:-cil_cord mixed}"
EPOCHS="${EPOCHS:-10}"

run() {  # run <dir> <method> <scenario> <seed> <bs> [extra overrides]
  local name=$1 method=$2 sc=$3 sd=$4 bs=$5; shift 5
  local d="results/$name"
  if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py method=$method scenario=$sc model=layoutlmv3_base seed=$sd \
    training.batch_size=$bs method.epochs=$EPOCHS $COMMON "$@" >"results/logs_vision/$name.log" 2>&1
  [ -f "$d/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
}

for sd in $SEEDS; do
  for sc in $SCENARIOS; do
    run ${sc}_colar_r_sliced_seed${sd}_k4_d50_r128 colar_r_sliced $sc $sd 2 $COLAR_AX
  done
done
echo "=== R2b QUEUE COMPLETE $(date) ===" >>"$LOG"
