#!/usr/bin/env bash
# Rented-GPU launcher for the IMAGE cells of the frozen 12-row baseline set (plan rev. 8,
# Tier B): CODA-Prompt, CL-LoRA, LCA x {CIFAR-100, ImageNet-R} x seeds {42,7,123} at the
# HOUSE recipe every image cell in the table was produced with: ViT-B/16, batch 16, fp32,
# gradient checkpointing, 100-epoch cap (peak VRAM < 2 GB). Renting buys wall-clock only.
#
# THIN wrapper (like setup_vastai.sh): fetches ImageNet-R if missing, pins the cell list,
# then exec's setup_remote.sh (deps, CUDA check, W&B/R2 prompts) with GRID_SCRIPT pointed at
# the vision grid instead of the document grid.
#
# ── vast.ai / any rented box ───────────────────────────────────────────────────────────
#   IMAGE:   pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime (match cuda12.x to the host driver)
#   CARD:    anything with >= 8 GB (an RTX 4070 is plenty); the recipe peaks < 2 GB VRAM.
#   RUN:     git clone <repo> && cd CLUE && git checkout doccl && bash scripts/setup_remote_vision.sh
#   SECRETS: W&B / R2 creds as instance ENV (-e ...), never a file in the repo. Both optional:
#            no W&B key -> offline; no R2 -> no durable resume (fine for ~1 day of work).
#   RESULTS: run_vision_grid.sh writes results/<cell>_vit/{metrics.json,matrix.npy,.done}.
#            Pull them back from the local box when done:
#              rsync -avz -e "ssh -p <port>" --include='*/' --include='cil_*_vit/**' --exclude='*' \
#                    <user>@<host>:~/CLUE/results/ ./results/
#   TIME:    measured on the local RTX 2060: full-fine-tune image cells take 2.1-3.1 h (CIFAR-100)
#            and 1.5-2.2 h (ImageNet-R) -> the 18 cells ≈ 35-40 GPU-h on the 2060, ≈ 10-15 h on a
#            3090/4070-class card. SEED_MAJOR=1 (default) runs all 6 cells at seed 42 first, then
#            7, then 123. Do NOT change batch size / precision: that breaks comparability.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- data: CIFAR-100 auto-downloads (torchvision); ImageNet-R does not --------------------
export DOCCL_VISION_ROOT="${DOCCL_VISION_ROOT:-data/vision}"   # repo-relative, same default as doccl/data/vision.py
IMR="$DOCCL_VISION_ROOT/imagenet-r"
if [ ! -d "$IMR" ]; then
  echo ">> ImageNet-R not found at $IMR — fetching (≈2 GB). Source: https://github.com/hendrycks/imagenet-r"
  mkdir -p "$DOCCL_VISION_ROOT" && cd "$DOCCL_VISION_ROOT"
  # Direct tarball linked from that repo's README; if the URL moves, download by hand and
  # extract so that $IMR/<wnid>/*.jpg exists (doccl/data/vision.py:imagenet_r_bases).
  curl -fL --retry 3 -o imagenet-r.tar "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" \
    || { echo "!! download failed — fetch imagenet-r.tar manually into $DOCCL_VISION_ROOT and re-run"; exit 1; }
  tar -xf imagenet-r.tar && rm -f imagenet-r.tar
  cd - >/dev/null
  [ -d "$IMR" ] || { echo "!! extraction did not produce $IMR"; exit 1; }
fi
echo ">> ImageNet-R ready: $(find "$IMR" -mindepth 1 -maxdepth 1 -type d | wc -l) classes"

# ---- the Tier B cell list, house recipe (do not change batch/precision) -----------------
export GRID_SCRIPT="scripts/run_vision_grid.sh"
export SCENARIOS="${SCENARIOS:-cil_cifar100 cil_imagenet_r}"
export METHODS="${METHODS:-coda_prompt cl_lora lca}"
export SEEDS="${SEEDS:-42 7 123}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_CKPT="${GRAD_CKPT:-true}"
export AMP=0; export DOCCL_AMP=0   # fp32, like every image cell on disk
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EPOCHS_CAP="${EPOCHS_CAP:-100}"
export WANDB_MODE="${WANDB_MODE:-offline}"
echo ">> cells: ${SCENARIOS} x ${METHODS} x seeds ${SEEDS} @ bs${BATCH_SIZE} grad_ckpt=${GRAD_CKPT}"

# Seed-major order: run the whole method x dataset block at seed 42 first, then 7, then 123,
# so a budget cut-off still leaves every method/dataset with at least one complete seed
# (and the grid's metrics.json skip makes each pass resume-safe).
if [ "${SEED_MAJOR:-1}" = 1 ]; then
  ALL_SEEDS="$SEEDS"
  for sd in $ALL_SEEDS; do
    echo ">> pass: seed $sd"
    SEEDS="$sd" GRID_SCRIPT="$GRID_SCRIPT" bash scripts/setup_remote.sh || exit $?
    # setup_remote.sh exec's the grid; it returns here only after that pass finishes.
  done
  exit 0
fi
exec bash scripts/setup_remote.sh
