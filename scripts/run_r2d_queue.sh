#!/bin/bash
# R2d: CoLaR-R sliced+pseudo refresh (tournament P1, deployable arm, second attempt).
# R2 pseudo seed 42 relabelled 1/0/0 tokens (oracle 101/760/1461): replaying stale O labels
# teaches the model "O" on the very memory tokens the refresh inspects, so no candidate
# survives. sliced_pseudo removes that signal during the task and refreshes after it.
# Waits for the R2 retry (which waits for R2c; single GPU slot). Resume-safe. Log: results/logs_vision/r2d_queue.log
# NOT auto-launched: start with `nohup bash scripts/run_r2d_queue.sh &` once R2c is confirmed.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2d_queue.log
mkdir -p results/logs_vision
until grep -q 'R2 RETRY COMPLETE' results/logs_vision/r2_retry.log 2>/dev/null; do sleep 300; done
echo "=== R2d QUEUE $(date) ===" >>"$LOG"
COLAR_AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
COMMON="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
for sd in ${SEEDS:-42 7 123}; do
  for sc in ${SCENARIOS:-cil_cord mixed}; do
    name="${sc}_colar_r_sliced_pseudo_seed${sd}_k4_d50_r128"
    if [ -f "results/$name/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
    echo "$(date +%H:%M) START $name" >>"$LOG"
    uv run python scripts/train.py method=colar_r_sliced_pseudo scenario=$sc model=layoutlmv3_base \
      seed=$sd method.epochs=10 training.batch_size=2 $COMMON $COLAR_AX >"results/logs_vision/$name.log" 2>&1
    [ -f "results/$name/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
  done
done
echo "=== R2d QUEUE COMPLETE $(date) ===" >>"$LOG"
