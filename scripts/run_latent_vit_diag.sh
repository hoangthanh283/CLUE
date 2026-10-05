#!/bin/bash
# Step 1 (plan rev. 3): make latent replay work on ViT. Seed-42 diagnostic arms on CIFAR-100.
# Resume-safe on results/<run>/metrics.json. Arms are passed as "<alias>:<overrides>" lines.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
LOG=results/logs_vision/latent_diag.log; mkdir -p results/logs_vision
COMMON="model=vit_b16 scenario=cil_cifar100 training=vision training.batch_size=16 training.num_workers=2 training.gradient_checkpointing=true wandb.mode=offline method.epochs=100 method.docs_per_task=500 method.replay_batch_size=16 +method.replay_task_balance=true"
ARMS="${ARMS:-latA:method.split_layer_k=8 latB:method.split_layer_k=11 latC:method.split_layer_k=4}"
SEEDS="${SEEDS:-42}"
echo "=== LATENT-VIT DIAG $(date) ===" >>"$LOG"
for arm in $ARMS; do
  alias="${arm%%:*}"; ov="${arm#*:}"
  for sd in $SEEDS; do
    run="cil_cifar100_${alias}_seed${sd}_vit"
    # train.py appends _k<N>/_d<N> suffixes after the family tag; match any of them.
    if ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; fi
    echo "$(date +%H:%M) START $run ($ov)" >>"$LOG"
    uv run python scripts/train.py method=latent_replay +method.run_alias=$alias seed=$sd $COMMON $ov \
      > "results/logs_vision/${run}.log" 2>&1
    if [ $? -eq 0 ] && ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"
    else echo "$(date +%H:%M) FAIL  $run" >>"$LOG"; fi
  done
done
echo "=== LATENT-VIT DIAG COMPLETE $(date) ===" >>"$LOG"
