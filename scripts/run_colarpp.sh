#!/bin/bash
# CoLaR++ phases (docs/IMAGE_SCOPE_PREREG.md Amendment 5). Resume-safe, sequential.
#   PHASE=A  screening, seed 42, CIFAR-100, raw latent bank (k=4, 500 img/task)
#   PHASE=A2 compression on the winner: WIN="<overrides of the Phase-A winner>"
#   PHASE=B  confirmation: CELLS="alias:ov,ov ..." over SCENARIOS x SEEDS
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
PHASE="${PHASE:?PHASE=A|A2|B}"
LOG=results/logs_vision/colarpp_queue.log; mkdir -p results/logs_vision
COMMON="model=vit_b16 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true method=colar_pp wandb.mode=offline"
CA="method.head_align_epochs=10"; SL="method.trunk_lr_scale=0.1"; WA="method.weight_align=true"; BAL="method.replay_balance=class"
case "$PHASE" in
  A)  SCENARIOS="cil_cifar100"; SEEDS="42"
      CELLS="${CELLS:-ppCA:$CA ppSL:$SL ppCASL:$CA,$SL ppCASLWA:$CA,$SL,$WA ppFULL:$CA,$SL,$WA,$BAL}" ;;
  A2) SCENARIOS="cil_cifar100"; SEEDS="42"; W="${WIN:?WIN=<comma-separated winner overrides>}"
      CELLS="${CELLS:-ppR128:$W,method.rank_r=128 ppR64:$W,method.rank_r=64 ppR16:$W,method.rank_r=16 ppR16q:$W,method.rank_r=16,method.quant=int8 ppR16qp:$W,method.rank_r=16,method.quant=int8,method.token_pool=2 ppR16qp2k:$W,method.rank_r=16,method.quant=int8,method.token_pool=2,method.docs_per_task=2000}" ;;
  B)  SCENARIOS="${SCENARIOS:-cil_cifar100 cil_imagenet_r}"; SEEDS="${SEEDS:-42 7 123}"; CELLS="${CELLS:?CELLS required for phase B}" ;;
esac
echo "=== COLAR++ PHASE $PHASE $(date) ===" >>"$LOG"
for sc in $SCENARIOS; do for cell in $CELLS; do for sd in $SEEDS; do
  alias="${cell%%:*}"; ov="${cell#*:}"; ov="${ov//,/ }"
  run="${sc}_${alias}_seed${sd}_vit"
  if [ -f "results/${run}/metrics.json" ]; then echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $run ($ov)" >>"$LOG"
  uv run python scripts/train.py scenario=$sc seed=$sd $COMMON +method.run_alias=$alias $ov \
    > "results/logs_vision/${run}.log" 2>&1
  if [ $? -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run" >>"$LOG"; fi
done; done; done
echo "=== COLAR++ PHASE $PHASE COMPLETE $(date) ===" >>"$LOG"
