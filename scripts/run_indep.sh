#!/bin/bash
# Independent per-task models (multi-model / task-ID-oracle upper bound): one naive run per
# task via scenario.kwargs.only_task=k. Resume-safe on results/<run>/metrics.json.
#   DIL (LayoutLMv3, local 6 GB recipe) x 3 tasks x 3 seeds, then Split CIFAR-100 and
#   Split ImageNet-R (ViT-B/16, fp16 AMP, no checkpointing) x 10 tasks x 3 seeds.
# Env: SEEDS="42 7 123"  DOC=1  IMG=1
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS="${SEEDS:-42 7 123}"; DOC="${DOC:-1}"; IMG="${IMG:-1}"
LOG=results/logs_indep/queue.log; mkdir -p results/logs_indep
run() {  # run <run-dir> <args...>
  local name="$1"; shift
  if [ -f "results/${name}/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py "$@" > "results/logs_indep/${name}.log" 2>&1
  if [ $? -eq 0 ] && [ -f "results/${name}/metrics.json" ]; then
    touch "results/${name}/.done"; echo "$(date +%H:%M) DONE  $name" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $name" >>"$LOG"; fi
}
echo "=== INDEPENDENT BOUND $(date) ===" >>"$LOG"
if [ "$DOC" = 1 ]; then
  for sd in $SEEDS; do for k in 0 1 2; do
    run "dil_indep${k}_naive_seed${sd}" method=naive scenario=dil seed=$sd +scenario.kwargs.only_task=$k \
      method.epochs=100 training.batch_size=2 training.num_workers=0 training.gradient_checkpointing=true
  done; done
fi
if [ "$IMG" = 1 ]; then
  VIT="model=vit_b16 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true method.epochs=100"
  for sc in cil_cifar100 cil_imagenet_r; do for sd in $SEEDS; do for k in 0 1 2 3 4 5 6 7 8 9; do
    run "${sc}_indep${k}_naive_seed${sd}_vit" method=naive scenario=$sc seed=$sd +scenario.kwargs.only_task=$k $VIT
  done; done; done
fi
echo "=== INDEPENDENT BOUND COMPLETE $(date) ===" >>"$LOG"
