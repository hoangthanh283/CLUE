#!/usr/bin/env bash
# Remote E2E grid driver for a POWERFUL machine (more VRAM, no 6 GB / cgroup limits).
#
# Runs the heavy part of the benchmark that we offload from the laptop:
#   prepare data -> single-task FWT baselines (9) -> PHASE 3 core (54)
#   -> PHASE 4 prompt/LoRA (36) -> aggregate.
# DocCL (PHASE 5) is run on the local machine; it is NOT run here by default.
#
# Same training protocol as the local driver: val-F1 early stopping (patience=2),
# epochs cap=100, per-run FWT (records zero-shot term + seeds tracker with b_i).
# Resume-safe: results/<run>/.done is written only on success, so reruns skip.
#
# Designed to run inside the Docker image (docker/Dockerfile). Outside Docker it also
# works as long as the env is set up (uv sync) and a GPU is visible.
#
# Tunables (env):
#   BATCH_SIZE     per-step batch (default 8 — a big GPU can go higher than the laptop's 2)
#   GRAD_CKPT      gradient checkpointing true/false (default false — not needed with VRAM)
#   EPOCHS_CAP     early-stop epoch ceiling (default 100)
#   SEEDS          space-separated seeds (default "42 123 7")
#   SCENARIOS      space-separated scenarios (default "cil_cord dil mixed")
#   PHASES         which phases to run (default "baselines core prompt aggregate")
#   WANDB_MODE     online/offline (default online; needs WANDB_API_KEY)
#   MEM_CAP        optional systemd cgroup cap (default empty = no cap on a big box)

set -uo pipefail
cd "$(dirname "$0")/.."

# Prefer an active venv / uv; fall back to system python.
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; source .env 2>/dev/null || true; set +a

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_CKPT="${GRAD_CKPT:-false}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
export SEEDS="${SEEDS:-42 123 7}"
export SCENARIOS="${SCENARIOS:-cil_cord dil mixed}"
PHASES="${PHASES:-baselines core prompt aggregate}"
export WANDB_MODE="${WANDB_MODE:-online}"
export MEM_CAP="${MEM_CAP:-}"          # empty => run_grid.sh runs train.py without a cgroup cap

LOG=results/logs/remote_grid.log
mkdir -p results/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

EXTRA="training.batch_size=${BATCH_SIZE} training.gradient_checkpointing=${GRAD_CKPT} \
training.num_workers=4 method.epochs=${EPOCHS_CAP} wandb.project=${WANDB_PROJECT:-CL4IE}"
export EXTRA

run_capped() {
  if [ -n "$MEM_CAP" ] && command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope -q -p "MemoryMax=$MEM_CAP" -p "MemorySwapMax=0" \
      python scripts/train.py "$@"
  else
    python scripts/train.py "$@"
  fi
}

say "=== REMOTE GRID START (bs=${BATCH_SIZE} ckpt=${GRAD_CKPT} cap=${EPOCHS_CAP}ep, phases: ${PHASES}) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOG" || true

# ── Prepare SROIE (needed by dil + mixed). Regenerates data/sroie/ from HF mirror. ──
if [ ! -f data/sroie/train.json ] || [ ! -f data/sroie/test.json ]; then
  say "Preparing SROIE from HuggingFace mirror (data/sroie/ missing) ..."
  python scripts/prepare_sroie.py --source hf >> "$LOG" 2>&1 || say "prepare_sroie returned nonzero (continuing)"
fi

# ── Single-task FWT baselines (9) — needed for per-run FWT (b_i). RUN FIRST. ────────
if echo "$PHASES" | grep -qw baselines; then
  say "Single-task FWT baselines (naive x {single_funsd,single_cord,single_sroie} x seeds)"
  for sc in single_funsd single_cord single_sroie; do
    for s in $SEEDS; do
      run="${sc}_naive_seed${s}"
      if [ -f "results/${run}/.done" ]; then say "  [skip] $run"; continue; fi
      say "  [run] $run"
      # shellcheck disable=SC2086
      if run_capped method=naive scenario=$sc seed=$s wandb.mode=$WANDB_MODE $EXTRA >> "$LOG" 2>&1; then
        mkdir -p "results/${run}"; touch "results/${run}/.done"
      else
        say "  [FAIL] $run (continuing)"
      fi
    done
  done
  say "Building single-task baseline CSV (b_i) for per-run FWT ..."
  python scripts/analyze_results.py >> "$LOG" 2>&1 || say "baseline-CSV build returned nonzero (continuing)"
fi

# ── PHASE 3: core baselines (54) ───────────────────────────────────────────────────
if echo "$PHASES" | grep -qw core; then
  say "PHASE 3 core (naive joint ewc lwf er der_pp x ${SCENARIOS} x ${SEEDS})"
  PHASE=3 bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase3 returned nonzero (continuing)"
  say "core .done: $(ls results/*_{naive,joint,ewc,lwf,er,der_pp}_seed*/.done 2>/dev/null | grep -v single | wc -l)"
fi

# ── PHASE 4: prompt/LoRA baselines (36) ────────────────────────────────────────────
if echo "$PHASES" | grep -qw prompt; then
  say "PHASE 4 prompt/LoRA (l2p dualprompt coda_prompt o_lora x ${SCENARIOS} x ${SEEDS})"
  PHASE=4 bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase4 returned nonzero (continuing)"
  say "prompt .done: $(ls results/*_{l2p,dualprompt,coda_prompt,o_lora}_seed*/.done 2>/dev/null | wc -l)"
fi

# ── Aggregate ──────────────────────────────────────────────────────────────────────
if echo "$PHASES" | grep -qw aggregate; then
  say "Aggregate (analyze_results.py)"
  python scripts/analyze_results.py >> "$LOG" 2>&1 || say "analyze_results returned nonzero"
fi

say "=== REMOTE GRID COMPLETE. total .done=$(ls results/*/.done 2>/dev/null | wc -l) ==="
