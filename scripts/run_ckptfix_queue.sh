#!/bin/bash
# Re-runs after the latent-replay x gradient-checkpointing fix (2026-10-06).
# 1. Document latent replay k8/d5, seeds 42/7/123 — IDENTICAL config to the 66.47 rerun
#    (docs/gridfill/latent_rerun.sh) except the bug fix: isolates the bug's effect.
# 2. ViT Step-1 diagnostic arms with fp16 AMP and no checkpointing (3.3x faster).
# 3. AMP calibration: ER@2000 CIFAR-100 seed 42 under AMP vs the fp32 cell (76.2).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# W&B online by default; export WANDB_MODE=offline to run without network
LOG=results/logs_vision/ckptfix_queue.log; mkdir -p results/logs_vision
run() {  # run <run-dir-glob> <log-name> <args...>
  local glob="$1" name="$2"; shift 2
  if ls results/${glob}/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py "$@" > "results/logs_vision/${name}.log" 2>&1
  if [ $? -eq 0 ] && ls results/${glob}/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) DONE  $name" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $name" >>"$LOG"; fi
}
echo "=== CKPT-FIX QUEUE $(date) ===" >>"$LOG"

for sd in 42 7 123; do
  run "dil_latfix_seed${sd}" "dil_latfix_seed${sd}" method=latent_replay +method.run_alias=latfix \
    scenario=dil seed=$sd method.epochs=100 method.split_layer_k=8 method.docs_per_task=5 \
    method.replay_batch_size=4 training.batch_size=2 training.num_workers=0 \
    training.gradient_checkpointing=true
done

VIT="model=vit_b16 scenario=cil_cifar100 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true method.epochs=100"
for arm in "latA:8" "latB:11" "latD:12" "latC:4"; do
  a="${arm%%:*}"; k="${arm#*:}"
  run "cil_cifar100_${a}_seed42_vit*" "cil_cifar100_${a}_seed42_vit" method=latent_replay +method.run_alias=$a \
    seed=42 $VIT method.split_layer_k=$k method.docs_per_task=500 method.replay_batch_size=16 \
    +method.replay_task_balance=true
done

run "cil_cifar100_er_b2000amp_seed42_vit" "cil_cifar100_er_b2000amp_seed42_vit" method=er_b2000 \
  method.run_alias=er_b2000amp seed=42 $VIT
echo "=== CKPT-FIX QUEUE COMPLETE $(date) ===" >>"$LOG"
