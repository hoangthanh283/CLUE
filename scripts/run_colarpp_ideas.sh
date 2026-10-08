#!/bin/bash
# Amendment 8 screen (seed 42, CIFAR-100). Sequential, resume-safe.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/ideas_queue.log; mkdir -p results/logs_vision
V="model=vit_b16 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true method=colar_pp scenario=cil_cifar100 seed=42"
S2="method.feature_anchor=1.0"
I1="method.drift_comp=true method.head_align_epochs=10"
I2="method.trunk_adapt=lora"
I4="method.rank_r=16 method.quant=int8 method.token_pool=2 method.docs_per_task=2000"
run() { local a="$1"; shift; local d="results/cil_cifar100_${a}_seed42_vit"
  if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) SKIP $a" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $a ($*)" >>"$LOG"
  uv run python scripts/train.py $V +method.run_alias=$a "$@" > "results/logs_vision/cil_cifar100_${a}_seed42_vit.log" 2>&1
  [ -f "$d/metrics.json" ] && echo "$(date +%H:%M) DONE  $a" >>"$LOG" || echo "$(date +%H:%M) FAIL  $a" >>"$LOG"; }
echo "=== IDEAS QUEUE $(date) ===" >>"$LOG"
run ppI1 $S2 $I1
run ppI2 $S2 $I2
run ppI4 $S2 $I4
run ppI1only $I1
run ppI124 $S2 $I1 $I2 $I4
echo "=== IDEAS QUEUE COMPLETE $(date) ===" >>"$LOG"
