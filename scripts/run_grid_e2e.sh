#!/usr/bin/env bash
# =============================================================================
# End-to-end driver for the 54-run CONVERGED baseline grid (train-to-convergence
# via val-F1 early stopping). Designed to be dropped onto a fresh, more powerful
# machine and run with a single command:
#
#     bash scripts/run_grid_e2e.sh
#
# It does the whole pipeline from a clean checkout:
#   0. prereq checks (python>=3.10, uv, optional CUDA)            -> fail fast
#   1. dependency install via `uv sync` (deterministic, uv.lock)  -> idempotent
#   2. W&B credentials (.env or env vars; offline if absent)
#   3. GPU auto-tuning (batch size / checkpointing from VRAM)     -> no 6GB throttle
#   4. PHASE 3: 54 core runs  (naive joint ewc lwf er der_pp x cil_cord dil mixed x 3 seeds)
#   5. single-task FWT baselines (9) so FWT is real, not zero
#   6. aggregate (analyze_results.py) + ingest (ingest_to_thesis.py)
#
# Everything is RESUME-SAFE: run_grid.sh writes results/<run>/.done only on a
# successful run, so re-invoking this script skips completed work. Safe to Ctrl-C
# and restart, or to run under nohup/tmux on a remote box.
#
# Knobs (env overrides, all optional):
#   SEEDS="42 123 7"          seeds to sweep
#   SCENARIOS="cil_cord dil mixed"
#   METHODS="naive joint ewc lwf er der_pp"
#   EPOCHS_CAP=100            early-stopping ceiling (rarely binds)
#   BATCH_SIZE=<auto>         override the VRAM-based auto choice
#   GRAD_CKPT=<auto>          true|false; override the VRAM-based auto choice
#   NUM_WORKERS=4             DataLoader workers (0 on tiny/limited-RAM boxes)
#   MEM_CAP=""                e.g. "9G" to cgroup-cap each run (Linux+systemd only;
#                             leave empty on a roomy machine — it is a 6GB-box safety net)
#   WANDB_MODE=online         online|offline|disabled
#   WANDB_PROJECT=CL4IE
#   SKIP_INSTALL=0            set 1 to skip `uv sync` (deps already present)
#   SKIP_AGGREGATE=0          set 1 to stop after training (no analyze/ingest)
# =============================================================================
set -uo pipefail

# ── Locate repo root (script may be invoked from anywhere) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/grid_e2e.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { say "FATAL: $*"; exit 1; }

# ── Config (env-overridable) ─────────────────────────────────────────────────
SEEDS="${SEEDS:-42 123 7}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed}"
METHODS="${METHODS:-naive joint ewc lwf er der_pp}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MEM_CAP="${MEM_CAP:-}"            # empty = no cgroup cap (default for powerful box)
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-CL4IE}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 0 so a fresh box can download datasets/models

say "=== GRID E2E START (train-to-convergence, val-F1 early stopping, cap=${EPOCHS_CAP}ep) ==="
say "repo: $REPO_ROOT"

# ── 0. Prereqs ───────────────────────────────────────────────────────────────
command -v uv >/dev/null 2>&1 || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
PYV="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
say "system python: $PYV  (uv will provision the project interpreter from pyproject.toml)"

# ── 1. Dependency install (idempotent, deterministic via uv.lock) ────────────
if [ "$SKIP_INSTALL" = "1" ]; then
    say "SKIP_INSTALL=1 -> skipping uv sync"
else
    say "uv sync (installing locked dependencies)…"
    uv sync >> "$LOG" 2>&1 || die "uv sync failed (see $LOG)"
    say "uv sync OK"
fi
PYBIN="$REPO_ROOT/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"   # fall back if venv layout differs

# ── 2. W&B credentials ───────────────────────────────────────────────────────
set -a; source .env 2>/dev/null || true; set +a
export WANDB_PROJECT
if [ "$WANDB_MODE" != "disabled" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    say "WARN: WANDB_API_KEY not set and WANDB_MODE=$WANDB_MODE -> falling back to offline"
    WANDB_MODE="offline"
fi
export WANDB_MODE
say "W&B: mode=$WANDB_MODE project=$WANDB_PROJECT"

# ── 3. GPU detection + auto-tuning ───────────────────────────────────────────
# On a powerful GPU we lift batch size and drop gradient checkpointing so the
# stronger hardware is actually used (the 6GB box forced bs=2 + checkpointing).
VRAM_MB=0
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    VRAM_MB="${VRAM_MB:-0}"
    say "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | ${VRAM_MB} MiB VRAM"
else
    say "WARN: nvidia-smi not found — assuming CPU/unknown GPU; using conservative defaults"
fi

# Auto choose batch size + checkpointing from VRAM (override with BATCH_SIZE/GRAD_CKPT).
if [ -z "${BATCH_SIZE:-}" ] || [ -z "${GRAD_CKPT:-}" ]; then
    if   [ "$VRAM_MB" -ge 40000 ]; then AUTO_BS=16; AUTO_CKPT=false   # A100/H100-class
    elif [ "$VRAM_MB" -ge 24000 ]; then AUTO_BS=12; AUTO_CKPT=false   # 3090/4090/A6000
    elif [ "$VRAM_MB" -ge 16000 ]; then AUTO_BS=8;  AUTO_CKPT=false   # V100/4080
    elif [ "$VRAM_MB" -ge 10000 ]; then AUTO_BS=4;  AUTO_CKPT=true    # 3080/2080Ti
    else                                AUTO_BS=2;  AUTO_CKPT=true    # 6GB box / unknown
    fi
fi
BATCH_SIZE="${BATCH_SIZE:-$AUTO_BS}"
GRAD_CKPT="${GRAD_CKPT:-$AUTO_CKPT}"
say "training: batch_size=$BATCH_SIZE gradient_checkpointing=$GRAD_CKPT num_workers=$NUM_WORKERS"

# Hydra overrides appended to every run.
EXTRA="training.batch_size=$BATCH_SIZE training.gradient_checkpointing=$GRAD_CKPT training.num_workers=$NUM_WORKERS method.epochs=$EPOCHS_CAP wandb.project=$WANDB_PROJECT"

# Optional per-run cgroup memory cap (Linux + systemd only). Empty by default.
run_train() {
    if [ -n "$MEM_CAP" ] && command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q -p "MemoryMax=$MEM_CAP" -p "MemorySwapMax=0" \
            "$PYBIN" scripts/train.py "$@"
    else
        "$PYBIN" scripts/train.py "$@"
    fi
}

# ── 4. PHASE 3: 54 core converged runs (resume-safe via run_grid.sh) ──────────
say "PHASE 3 core: METHODS=[$METHODS] SCENARIOS=[$SCENARIOS] SEEDS=[$SEEDS]"
PHASE=3 METHODS="$METHODS" SCENARIOS="$SCENARIOS" SEEDS="$SEEDS" \
    WANDB_MODE="$WANDB_MODE" EXTRA="$EXTRA" MEM_CAP="$MEM_CAP" \
    bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "PHASE 3 returned nonzero (continuing)"
CORE_DONE="$(ls results/*_{naive,joint,ewc,lwf,er,der_pp}_seed*/.done 2>/dev/null | wc -l | tr -d ' ')"
say "PHASE 3 done markers: $CORE_DONE"

# ── 5. Single-task FWT baselines (9): single_{funsd,cord,sroie} x seeds ───────
say "FWT single-task baselines (single_funsd single_cord single_sroie x [$SEEDS])"
for sc in single_funsd single_cord single_sroie; do
    for s in $SEEDS; do
        run="${sc}_naive_seed${s}"
        if [ -f "results/${run}/.done" ]; then say "  [skip] $run"; continue; fi
        say "  [run]  $run"
        if run_train method=naive scenario="$sc" seed="$s" wandb.mode="$WANDB_MODE" $EXTRA >> "$LOG" 2>&1; then
            mkdir -p "results/${run}"; touch "results/${run}/.done"
        else
            say "  [FAIL] $run"
        fi
    done
done
FWT_DONE="$(ls results/single_*_naive_seed*/.done 2>/dev/null | wc -l | tr -d ' ')"
say "FWT done markers: $FWT_DONE"

# ── 6. Aggregate + ingest ────────────────────────────────────────────────────
if [ "$SKIP_AGGREGATE" = "1" ]; then
    say "SKIP_AGGREGATE=1 -> skipping analyze/ingest"
else
    say "aggregate: analyze_results.py"
    "$PYBIN" scripts/analyze_results.py >> "$LOG" 2>&1 || say "analyze_results.py returned nonzero"
    say "ingest: ingest_to_thesis.py"
    "$PYBIN" scripts/ingest_to_thesis.py >> "$LOG" 2>&1 || say "ingest_to_thesis.py returned nonzero (ok if thesis/ absent on this box)"
fi

TOTAL_DONE="$(ls results/*/.done 2>/dev/null | wc -l | tr -d ' ')"
say "=== GRID E2E COMPLETE. total .done=$TOTAL_DONE (core=$CORE_DONE, fwt=$FWT_DONE) ==="
say "results/analysis/all_runs.csv + pivot_*.csv (if aggregated); full log: $LOG"
