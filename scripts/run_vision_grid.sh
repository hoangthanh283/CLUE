#!/bin/bash
# Image-CL scope test (docs/IMAGE_SCOPE_PREREG.md): ViT-B/16 on Split CIFAR-100 / ImageNet-R.
# Resume-safe (skips runs with metrics.json). Env knobs:
#   SCENARIOS="cil_cifar100 cil_imagenet_r"  METHODS="naive joint ewc lwf er der_pp slca"
#   SEEDS="42 7 123"  BATCH_SIZE=128  GRAD_CKPT=false  NUM_WORKERS=4  EPOCHS_CAP=100
# Launch one copy per GPU with disjoint METHODS (CUDA_VISIBLE_DEVICES=<id>) to fan out.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# W&B online by default; export WANDB_MODE=offline to run without network
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
  # House recipe for every image cell in the paper tables = batch 16, fp32, gradient
  # checkpointing (all SLCA/RanPAC/ER/DER++/naive/joint cells on disk were produced that
  # way; the 128 default below was never used for a reported cell). Any OTHER batch size
  # gets its own run_alias so it lands in <sc>_<me>_bs<N>_seed<sd>_vit and can never
  # overwrite or masquerade as a house-recipe cell. See plan rev. 9, Tier B.
  REF_BATCH="${REF_BATCH:-16}"
  alias=""; mk="$me"
  if [ "$BATCH_SIZE" != "$REF_BATCH" ]; then
    mk="${me}_bs${BATCH_SIZE}"
    if grep -q "^run_alias:" "configs/method/${me}.yaml"; then alias="method.run_alias=${mk}"; else alias="+method.run_alias=${mk}"; fi
  fi
  run="${sc}_${mk}_seed${sd}_vit"
  [ -f "results/${run}/metrics.json" ] && { echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; }
  # SLCA carries its own epoch budget (20) and LR schedule; the cap applies to the rest.
  ep="method.epochs=${EPOCHS_CAP}"; [ "$me" = slca ] && ep=""
  echo "$(date +%H:%M) START $run" >>"$LOG"
  uv run python scripts/train.py method=$me scenario=$sc model=vit_b16 seed=$sd \
    training=vision training.batch_size=$BATCH_SIZE training.num_workers=$NUM_WORKERS \
    training.gradient_checkpointing=$GRAD_CKPT $ep $alias \
    > "$LOGDIR/${run}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then
    touch "results/${run}/.done"; echo "$(date +%H:%M) DONE  $run" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $run (rc=$rc)" >>"$LOG"; fi
done; done; done
echo "=== VISION GRID COMPLETE — $(date) ===" >>"$LOG"
