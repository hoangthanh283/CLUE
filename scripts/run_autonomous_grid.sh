#!/usr/bin/env bash
# Autonomous baseline-grid driver (deadline run, 3 epochs).
#
# Runs the full pipeline sequentially, resume-safe, within hard resource limits:
#   PHASE 3 (54 core) -> single-task FWT baselines (9) -> PHASE 4 (36 prompt/LoRA)
#   -> PHASE 5 (9 doccl) -> aggregate -> ingest.
#
# Safety: every training run uses num_workers=0 (no DataLoader fork OOM), bs=1 +
# gradient checkpointing (VRAM < 5 GB), epochs=3. A separate watchdog
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

# 3-epoch deadline budget + memory-safe DataLoader + limited-VRAM recipe.
EXTRA="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 method.epochs=3 wandb.project=CL4IE"
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

say "=== AUTONOMOUS GRID START (3 epochs) ==="

# ── PHASE 3: core baselines (54) ───────────────────────────────────────────────
say "PHASE 3 core baselines (naive joint ewc lwf er der_pp x cil_cord dil mixed x 3 seeds)"
PHASE=3 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase3 returned nonzero (continuing)"
say "PHASE 3 done markers: $(ls results/*_{naive,joint,ewc,lwf,er,der_pp}_seed*/.done 2>/dev/null | wc -l)"

# ── Single-task FWT baselines (9) ──────────────────────────────────────────────
say "Single-task FWT baselines (naive x {single_funsd,single_cord,single_sroie} x 3 seeds)"
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

# ── PHASE 4: prompt/LoRA baselines (36) ────────────────────────────────────────
say "PHASE 4 prompt/LoRA (l2p dualprompt coda_prompt o_lora x cil_cord dil mixed x 3 seeds)"
PHASE=4 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase4 returned nonzero (continuing)"
say "PHASE 4 done markers: $(ls results/*_{l2p,dualprompt,coda_prompt,o_lora}_seed*/.done 2>/dev/null | wc -l)"

# ── PHASE 5: doccl (9, DocCL_A placeholder) ────────────────────────────────────
say "PHASE 5 doccl (DocCL_A placeholder) x cil_cord dil mixed x 3 seeds"
PHASE=5 EXTRA="$EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 || say "phase5 returned nonzero (continuing)"
say "PHASE 5 done markers: $(ls results/*_doccl_seed*/.done 2>/dev/null | wc -l)"

# ── Aggregate + ingest ─────────────────────────────────────────────────────────
say "Aggregate (analyze_results.py)"
python scripts/analyze_results.py >> "$LOG" 2>&1 || say "analyze_results returned nonzero"
say "Ingest into thesis (ingest_to_thesis.py)"
python scripts/ingest_to_thesis.py >> "$LOG" 2>&1 || say "ingest returned nonzero"

say "=== AUTONOMOUS GRID COMPLETE. total .done=$(ls results/*/.done 2>/dev/null | wc -l) ==="
