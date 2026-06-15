#!/usr/bin/env bash
# Autonomous baseline-grid driver (train-to-convergence: val-F1 early stopping).
#
# Runs the pipeline sequentially, resume-safe, within hard resource limits:
#   single-task FWT baselines (9) -> PHASE 5 (9 doccl) -> [PHASE 3 core (54) +
#   PHASE 4 prompt/LoRA (36) unless SKIP_CORE=1] -> aggregate -> ingest.
# DocCL runs right after the baselines so the proposed method lands early; Core and
# Prompt/LoRA are a local fallback meant for a more powerful machine (docker/).
#
# Training protocol: each task trains until its held-out val-F1 stops improving for
# `early_stop_patience` consecutive epochs (then best-val weights are restored), with
# a high epoch ceiling (100) that should rarely bind. This replaces the earlier fixed
# 3-epoch budget, which under-trained EWC and used no convergence criterion.
#
# Safety: every training run uses num_workers=0 (no DataLoader fork OOM), bs=2 +
# gradient checkpointing (VRAM < 5 GB). A separate watchdog
# (run_grid_watchdog.sh) enforces RAM<14GB / VRAM<5GB and kills this driver if breached.
#
# "skip + continue": a failed run is logged and skipped, never halting the pipeline
# (run_grid.sh marks results/<run>/.done only on success, so reruns resume).

set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
set -a; source .env 2>/dev/null || true; set +a

LOG=results/logs/autonomous.log
mkdir -p results/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Train-to-convergence budget + memory-safe DataLoader + limited-VRAM recipe.
# Gradient checkpointing ON: the regularization/replay methods (EWC stores Fisher+theta_star
# for all params; LwF holds a frozen teacher; ER/DER hold replay batches) need more VRAM than
# naive/joint, and without checkpointing EWC OOM'd the 6 GB GPU at ~5.6 GB. Checkpointing keeps
# every method under ~2.6 GB VRAM (proven across the naive/joint runs). bs=2 retained.
# epochs=100 is a safety ceiling; val-F1 early stopping (patience=2) ends each task at
# convergence and restores its best-val weights. Most tasks stop in 4-8 epochs.
# Early stopping is ON by default in code (make_early_stopper: enabled when a val_loader
# is present, patience defaults to 2) so no method.early_stop* override is needed here —
# the method configs are immutable and Hydra struct-mode rejects unknown keys.
EXTRA="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 method.epochs=100 wandb.project=CL4IE"
export WANDB_MODE=online

# HARD per-run memory cap: each train.py runs in a cgroup capped at MEM_CAP with swap
# disabled, so the kernel OOM-kills only that run (never the machine) if it spikes. The
# real peak for the heaviest scenario (mixed, 6 sessions) + model is ~3.2 GB, so 9G is
# generous headroom while staying far under the 15 GB box.
export MEM_CAP="${MEM_CAP:-9G}"

# Helper: run a single capped train.py (used by the FWT single-task loop below).
run_capped() {
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q -p "MemoryMax=$MEM_CAP" -p "MemorySwapMax=0" \
            python scripts/train.py "$@"
    else
        python scripts/train.py "$@"
    fi
}

say "=== AUTONOMOUS GRID START (train-to-convergence, val-F1 early stopping patience=2, cap=100ep) ==="

# ── Single-task FWT baselines (9) — RUN FIRST ──────────────────────────────────
# These produce the b_i term (from-scratch single-task F1 per dataset) that true FWT
# subtracts. Running them BEFORE the core methods means results/table_single_task_
# baselines.csv exists when each CL run starts, so train.py can seed the metrics
# tracker and write a real per-run FWT into metrics.json alongside AA/BWT/AF.
say "Single-task FWT baselines (naive x {single_funsd,single_cord,single_sroie} x 3 seeds) — FIRST"
for sc in single_funsd single_cord single_sroie; do
  for s in 42 123 7; do
    run="${sc}_naive_seed${s}"
    if [ -f "results/${run}/.done" ]; then say "  [skip] $run"; continue; fi
    say "  [run] $run"
    # shellcheck disable=SC2086
    if run_capped method=naive scenario=$sc seed=$s wandb.mode=online $EXTRA >> "$LOG" 2>&1; then
      mkdir -p "results/${run}"; touch "results/${run}/.done"
    else
      say "  [FAIL] $run (continuing)"
    fi
  done
done

# Materialise the b_i baseline CSV from the single-task runs so the CL runs below can
# read it for per-run FWT. (analyze_results.py also (re)writes this at the end.)
say "Building single-task baseline CSV (b_i) for per-run FWT ..."
python scripts/analyze_results.py >> "$LOG" 2>&1 || say "baseline-CSV build returned nonzero (continuing)"

# ── PHASE 5: doccl (9, DocCL_A placeholder) — RUN RIGHT AFTER BASELINES ─────────
# The proposed method lands first so its results are available early. PHASE 3/4 (Core
# + Prompt/LoRA) follow as a local fallback but are intended to run on a more powerful
# machine (see docker/ + scripts/run_grid_remote.sh). Set SKIP_CORE=1 to skip them
# locally once the remote machine has them.
say "PHASE 5 doccl (DocCL_A placeholder) x cil_cord dil mixed x 3 seeds — AFTER baselines"
PHASE=5 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase5 returned nonzero (continuing)"
say "PHASE 5 done markers: $(ls results/*_doccl_seed*/.done 2>/dev/null | wc -l)"

if [ "${SKIP_CORE:-0}" = "1" ]; then
  say "SKIP_CORE=1 set — skipping PHASE 3 (core) and PHASE 4 (prompt/LoRA) on this machine "\
"(run them on the powerful machine via scripts/run_grid_remote.sh / docker)."
else
  # ── PHASE 3: core baselines (54) — local fallback ────────────────────────────
  say "PHASE 3 core baselines (naive joint ewc lwf er der_pp x cil_cord dil mixed x 3 seeds)"
  PHASE=3 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase3 returned nonzero (continuing)"
  say "PHASE 3 done markers: $(ls results/*_{naive,joint,ewc,lwf,er,der_pp}_seed*/.done 2>/dev/null | wc -l)"

  # ── PHASE 4: prompt/LoRA baselines (36) — local fallback ─────────────────────
  say "PHASE 4 prompt/LoRA (l2p dualprompt coda_prompt o_lora x cil_cord dil mixed x 3 seeds)"
  PHASE=4 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase4 returned nonzero (continuing)"
  say "PHASE 4 done markers: $(ls results/*_{l2p,dualprompt,coda_prompt,o_lora}_seed*/.done 2>/dev/null | wc -l)"
fi

# ── Aggregate + ingest ─────────────────────────────────────────────────────────
say "Aggregate (analyze_results.py)"
python scripts/analyze_results.py >> "$LOG" 2>&1 || say "analyze_results returned nonzero"
say "Ingest into thesis (ingest_to_thesis.py)"
python scripts/ingest_to_thesis.py >> "$LOG" 2>&1 || say "ingest returned nonzero"

say "=== AUTONOMOUS GRID COMPLETE. total .done=$(ls results/*/.done 2>/dev/null | wc -l) ==="
