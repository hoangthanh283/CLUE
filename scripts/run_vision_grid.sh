#!/bin/bash
# Image-CL scope test (docs/IMAGE_SCOPE_PREREG.md): ViT-B/16 on Split CIFAR-100 / ImageNet-R.
# Resume-safe (skips runs with metrics.json). Env knobs:
#   SCENARIOS="cil_cifar100 cil_imagenet_r"  METHODS="naive joint ewc lwf er der_pp slca"
#   SEEDS="42 7 123"  BATCH_SIZE=128  GRAD_CKPT=false  NUM_WORKERS=4  EPOCHS_CAP=100
# Launch one copy per GPU with disjoint METHODS (CUDA_VISIBLE_DEVICES=<id>) to fan out.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
SCENARIOS="${SCENARIOS:-cil_cifar100 cil_imagenet_r}"
METHODS="${METHODS:-naive joint ewc lwf er der_pp slca}"
SEEDS="${SEEDS:-42 7 123}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_CKPT="${GRAD_CKPT:-false}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
LOGDIR=results/logs_vision; mkdir -p "$LOGDIR"; LOG="$LOGDIR/queue.log"

echo "=== VISION GRID — $(date) ===" >>"$LOG"
for sc in $SCENARIOS; do for me in $METHODS; do for sd in $SEEDS; do
  run="${sc}_${me}_seed${sd}_vit"
  [ -f "results/${run}/metrics.json" ] && { echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; }
  # SLCA carries its own epoch budget (20) and LR schedule; the cap applies to the rest.
  ep="method.epochs=${EPOCHS_CAP}"; [ "$me" = slca ] && ep=""
  echo "$(date +%H:%M) START $run" >>"$LOG"
  uv run python scripts/train.py method=$me scenario=$sc model=vit_b16 seed=$sd \
    training=vision training.batch_size=$BATCH_SIZE training.num_workers=$NUM_WORKERS \
    training.gradient_checkpointing=$GRAD_CKPT $ep wandb.mode=offline \
    > "$LOGDIR/${run}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then
    touch "results/${run}/.done"; echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run (rc=$rc)" >>"$LOG"; fi
done; done; done
echo "=== VISION GRID COMPLETE — $(date) ===" >>"$LOG"
