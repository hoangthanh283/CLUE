#!/bin/bash
# R2c: restore disk-backed CoLaR DIL headline cells (10-epoch protocol, seeds 42/7/123).
# The July 10-epoch seed-42 run dir was overwritten by the 5-epoch Amendment-9 control and
# seeds 7/123 kept only train.log; this re-run writes metrics.json under the alias colar_e10.
# Waits for the R2b queue (single GPU slot). Resume-safe. Log: results/logs_vision/r2c_queue.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2c_queue.log
mkdir -p results/logs_vision
until grep -q 'R2b QUEUE COMPLETE' results/logs_vision/r2b_queue.log 2>/dev/null; do sleep 300; done
echo "=== R2c QUEUE $(date) ===" >>"$LOG"
COLAR_AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
COMMON="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
for sd in ${SEEDS:-42 7 123}; do
  name="dil_colar_e10_seed${sd}_k4_d50_r128"
  if [ -f "results/$name/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py method=colar +method.run_alias=colar_e10 scenario=dil model=layoutlmv3_base \
    seed=$sd method.epochs=10 training.batch_size=2 $COMMON $COLAR_AX >"results/logs_vision/$name.log" 2>&1
  [ -f "results/$name/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
done
echo "=== R2c QUEUE COMPLETE $(date) ===" >>"$LOG"
