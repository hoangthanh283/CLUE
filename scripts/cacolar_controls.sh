#!/bin/bash
# CA-CoLaR equal-byte uniform control (colar d10/r64, k4, 5 ep) on seeds 7/123.
# Mirrors results/dil_colar_seed42_k4_d10 so the A1 treatment (seeds 7/123 done) has a
# seed-matched control. Waits for the DER++ retry queue to finish first (one GPU job at a time).
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
LOG=results/logs_remaining/cacolar_controls.log

while pgrep -f retry_timeouts.sh >/dev/null; do sleep 120; done

for sd in 7 123; do
  run="dil_colar_seed${sd}_k4_d10"
  if [ -f "results/${run}/metrics.json" ]; then echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $run" >>"$LOG"
  timeout 7200 uv run python scripts/train.py \
    method=colar scenario=dil model=layoutlmv3_base seed=$sd \
    method.epochs=5 method.split_layer_k=4 method.docs_per_task=10 \
    training.batch_size=1 training.num_workers=0 training.gradient_checkpointing=true \
    wandb.mode=offline > "results/logs_remaining/${run}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run (rc=$rc)" >>"$LOG"; fi
done
echo "$(date +%H:%M) CONTROLS COMPLETE" >>"$LOG"
