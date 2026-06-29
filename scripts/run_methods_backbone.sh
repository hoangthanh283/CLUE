#!/usr/bin/env bash
# Generic METHODS x SCENARIOS x SEEDS launcher for ONE backbone family — for running a
# chosen method subset (e.g. the LexSlot comparison baselines) on a single backbone
# (e.g. BERT) on the local box. Resume-safe (skips metrics.json/.done, backfills .done).
# Self-parallel via JOBS. Each job is self-contained (COMMON baked in) so the xargs
# subshell needs no inherited env. Run-names mirror train.py exactly so resume + analysis dedup.
#
#   Knobs (env):
#     METHODS    space list (default "doccl er der_pp joint" = LexSlot's direct comparators)
#     FAMILY     backbone family (default "bert"); "" or "layoutlmv3" = primary (no suffix)
#     SCENARIOS  default "cil_cord dil mixed dil_xlingual cil_wildreceipt"
#     SEEDS      default "42 123 7"
#     JOBS       concurrent procs (default 1 on the 6GB box; 8 on an A6000)
#     BATCH_SIZE default 2 (use 16 on A6000); GRAD_CKPT default true; NUM_WORKERS default 0
#     EPOCHS_CAP default 100 ; WANDB_MODE default offline ; DRY_RUN default 0 ; GPUS default 0
#
#   Example (BERT baselines, local):  bash scripts/run_methods_backbone.sh
#   Dry-run:                          DRY_RUN=1 bash scripts/run_methods_backbone.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DOCCL_AMP="${AMP:-1}"

METHODS="${METHODS:-doccl er der_pp joint}"
FAMILY="${FAMILY:-bert}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
SEEDS="${SEEDS:-42 123 7}"
JOBS="${JOBS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_CKPT="${GRAD_CKPT:-true}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DRY_RUN="${DRY_RUN:-0}"
GPUS="${GPUS:-0}"

PYBIN="${PYTHON:-python}"; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN="python3"
COMMON="training.batch_size=${BATCH_SIZE} training.gradient_checkpointing=${GRAD_CKPT} \
training.num_workers=${NUM_WORKERS} method.epochs=${EPOCHS_CAP} wandb.mode=${WANDB_MODE} \
wandb.project=CL4IE"

# family suffix + model override (mirror train.py: primary layoutlmv3 has no suffix).
fam_suffix=""; model_ov=""
if [ -n "$FAMILY" ] && [ "$FAMILY" != "layoutlmv3" ]; then
  fam_suffix="_${FAMILY}"; model_ov="model=${FAMILY}_base"
fi

JOBS_LIST=()
for m in $METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
  rn="${sc}_${m}_seed${s}${fam_suffix}"
  JOBS_LIST+=("${rn}|method=${m} scenario=${sc} seed=${s} ${model_ov} ${COMMON}")
done; done; done

echo "### methods=[$METHODS] family=${FAMILY:-primary}: ${#JOBS_LIST[@]} jobs (JOBS=$JOBS bs=$BATCH_SIZE ckpt=$GRAD_CKPT) ###"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS_LIST[@]}"; do i=$((i+1)); printf '  %3d  %s -> %s\n' "$i" "${j%%|*}" "${j#*|}"; done
  exit 0
fi

run_one() {
  local rn="${1%%|*}" ovr="${1#*|}"
  if [ -f "results/${rn}/.done" ] || [ -f "results/${rn}/metrics.json" ]; then
    [ -f "results/${rn}/.done" ] || touch "results/${rn}/.done"
    echo "SKIP ${rn} (done)"; return 0
  fi
  echo "=== RUN ${rn} ==="
  # shellcheck disable=SC2086
  if CUDA_VISIBLE_DEVICES="${GPUS}" "$PYBIN" scripts/train.py $ovr \
       >"results/logs/${rn//\//_}.log" 2>&1; then
    touch "results/${rn}/.done"; echo "=== DONE ${rn} ==="
  else
    echo "!!! FAILED ${rn} (see results/logs/${rn//\//_}.log) !!!"
  fi
}
export -f run_one; export PYBIN GPUS

mkdir -p results/logs
printf '%s\n' "${JOBS_LIST[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "### ALL JOBS COMPLETE (methods=[$METHODS] family=${FAMILY:-primary}) ###"
