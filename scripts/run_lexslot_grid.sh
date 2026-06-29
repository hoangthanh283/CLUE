#!/usr/bin/env bash
# LexSlot-ONLY grid launcher — runs just the lexslot contribution across all scenarios,
# seeds, and (optionally) backbones, WITHOUT re-running the baseline tiers that the full
# run_grid_multigpu.sh pulls in. Resume-safe (skips results/<run>/.done). Parallel via a
# simple slot pool. Intended for a rented A6000 (48 GB): pack many jobs concurrently.
#
#   Knobs (env):
#     JOBS         concurrent train.py procs (default 8 on A6000; 1 on the 6GB box)
#     BATCH_SIZE   per-step batch (default 16; use 2 on the 6GB box)
#     GRAD_CKPT    gradient checkpointing (default false; true on the 6GB box)
#     EPOCHS_CAP   early-stop epoch ceiling (default 100)
#     SEEDS        default "42 123 7"
#     SCENARIOS    default "cil_cord dil mixed dil_xlingual cil_wildreceipt"
#     BACKBONES    secondary families to ALSO run beyond the primary LayoutLMv3.
#                  default "lilt bros bert" = ALL backbones. Set BACKBONES="" for primary only.
#     RUN_ABLATION default 1 (slot_depth x slot_sharing ablation on ABLATION_SCENARIOS)
#     ABLATION_SCENARIOS default "cil_cord"
#     WANDB_MODE   default offline
#     DRY_RUN      default 0 (print the plan + count, then exit)
#
#   Example (A6000):  JOBS=8 bash scripts/run_lexslot_grid.sh
#   Example (2060):   JOBS=1 BATCH_SIZE=2 GRAD_CKPT=true bash scripts/run_lexslot_grid.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DOCCL_AMP="${AMP:-1}"

JOBS="${JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_CKPT="${GRAD_CKPT:-false}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
SEEDS="${SEEDS:-42 123 7}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
# All backbones by default (primary LayoutLMv3 + LiLT + BROS + BERT). BACKBONES="" = primary only.
BACKBONES="${BACKBONES-lilt bros bert}"
RUN_ABLATION="${RUN_ABLATION:-1}"
ABLATION_SCENARIOS="${ABLATION_SCENARIOS:-cil_cord}"
WANDB_MODE="${WANDB_MODE:-offline}"
DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

PYBIN="${PYTHON:-python}"; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN="python3"
COMMON="training.batch_size=${BATCH_SIZE} training.gradient_checkpointing=${GRAD_CKPT} \
training.num_workers=${NUM_WORKERS} method.epochs=${EPOCHS_CAP} wandb.mode=${WANDB_MODE} \
wandb.project=CL4IE"

JOBS_LIST=()
add() { JOBS_LIST+=("$1|$2"); }   # run-name (== train.py run.name) | overrides

# model= override + run-name family suffix for secondary backbones (mirrors train.py).
emit() {  # $1 scenario, $2 seed, $3 family ("" = primary), rest = extra overrides + name suffix
  local sc="$1" s="$2" fam="$3"; shift 3
  local name_suffix="$1" ovr="$2"
  local rn="${sc}_lexslot_seed${s}"
  local model_ov=""
  [ -n "$fam" ] && { rn="${rn}_${fam}"; model_ov="model=${fam}_base"; }
  rn="${rn}${name_suffix}"
  add "$rn" "method=lexslot scenario=${sc} seed=${s} ${model_ov} ${ovr}"
}

build() {
  local fams="primary"; [ -n "$BACKBONES" ] && fams="primary $BACKBONES"
  for fam in $fams; do
    local f=""; [ "$fam" != "primary" ] && f="$fam"
    for sc in $SCENARIOS; do for s in $SEEDS; do
      # canonical full method (head_late + soft -> no suffix)
      emit "$sc" "$s" "$f" "" ""
    done; done
    # ablation only on the primary backbone (keep secondary sweeps to the main config)
    if [ "$RUN_ABLATION" = "1" ] && [ -z "$f" ]; then
      for sc in $ABLATION_SCENARIOS; do for s in $SEEDS; do
        emit "$sc" "$s" "" "_off"       "method.slot_sharing=off"
        emit "$sc" "$s" "" "_uniform"   "method.slot_depth=uniform"
        emit "$sc" "$s" "" "_head_only" "method.slot_depth=head_only"
      done; done
    fi
  done
}

build
echo "### LexSlot grid: ${#JOBS_LIST[@]} jobs (JOBS=$JOBS bs=$BATCH_SIZE ckpt=$GRAD_CKPT) ###"
if [ "$DRY_RUN" = "1" ]; then
  i=0; for j in "${JOBS_LIST[@]}"; do i=$((i+1)); printf '  %3d  %s -> %s\n' "$i" "${j%%|*}" "${j#*|}"; done
  exit 0
fi

run_one() {
  local rn="${1%%|*}" ovr="${1#*|}"
  # Resume: skip if already finished. train.py writes metrics.json on success; the grid
  # scheduler writes .done. Treat EITHER as complete (and backfill .done for grid-compat).
  if [ -f "results/${rn}/.done" ] || [ -f "results/${rn}/metrics.json" ]; then
    [ -f "results/${rn}/.done" ] || touch "results/${rn}/.done"
    echo "SKIP ${rn} (done)"; return 0
  fi
  echo "=== RUN ${rn} ==="
  # shellcheck disable=SC2086
  if CUDA_VISIBLE_DEVICES="${GPUS:-0}" "$PYBIN" scripts/train.py $ovr \
       >"results/logs/${rn//\//_}.log" 2>&1; then
    touch "results/${rn}/.done"   # mark complete so re-runs skip it
    echo "=== DONE ${rn} ==="
  else
    echo "!!! FAILED ${rn} (see results/logs/${rn//\//_}.log) !!!"
  fi
}
export -f run_one; export PYBIN GPUS

mkdir -p results/logs
# Simple parallel slot pool: keep up to $JOBS train.py procs running.
printf '%s\n' "${JOBS_LIST[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "### ALL LEXSLOT GRID JOBS COMPLETE ###"
