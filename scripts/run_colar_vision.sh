#!/bin/bash
# Step 2 (prereg Amendment 4): CoLaR on ViT image CIL. K = selected split layer from Step 1.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# W&B online by default; export WANDB_MODE=offline to run without network
K="${K:?set K=<split_layer_k from Step 1>}"
SCENARIOS="${SCENARIOS:-cil_cifar100 cil_imagenet_r}"; SEEDS="${SEEDS:-42 7 123}"
LOG=results/logs_vision/colar_queue.log; mkdir -p results/logs_vision
COMMON="model=vit_b16 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true method.split_layer_k=$K"
# alias:overrides — raw bank, CoLaR r128/64/16 @ d500, CoLaR r16 @ d2000, AGLR summary (H3)
CELLS="${CELLS:-bank500:method=latent_replay,method.docs_per_task=500,method.replay_batch_size=16,+method.replay_task_balance=true colar500r128:method=colar_vit,method.rank_r=128 colar500r64:method=colar_vit,method.rank_r=64 colar500r16:method=colar_vit,method.rank_r=16 colar2000r16:method=colar_vit,method.rank_r=16,method.docs_per_task=2000 aglr:method=aglr_replay,method.carriers_per_task=500,method.attn_keep=1.0,method.replay_batch_size=16}"
echo "=== COLAR VISION K=$K $(date) ===" >>"$LOG"
for sc in $SCENARIOS; do for cell in $CELLS; do for sd in $SEEDS; do
  alias="${cell%%:*}"; ov="${cell#*:}"; ov="${ov//,/ }"
  run="${sc}_${alias}_seed${sd}_vit"
  if ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $run" >>"$LOG"
  uv run python scripts/train.py scenario=$sc seed=$sd $ov +method.run_alias=$alias method.epochs=100 $COMMON \
    > "results/logs_vision/${run}.log" 2>&1
  if [ $? -eq 0 ] && ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run" >>"$LOG"; fi
done; done; done
echo "=== COLAR VISION COMPLETE $(date) ===" >>"$LOG"
