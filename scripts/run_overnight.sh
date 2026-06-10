#!/usr/bin/env bash
# Autonomous end-to-end driver: pilot -> analyze -> GATE A -> grid -> aggregate -> ingest.
# Designed to run unattended overnight. All steps resume-safe; failures are skipped
# and logged, never halting the whole pipeline (per user's "skip + continue").
#
# Memory: every full-FT run uses bs=1 + gradient checkpointing + expandable_segments
# (verified to peak ~2.6/6.1 GB on the RTX 2060 — no OOM / machine-restart risk).
#
# W&B: grid logs ONLINE to thanh-workspace/CL4IE (creds from .env).
#
# Output: a running narrative to results/logs/overnight.log; phase markers under
# results/logs/overnight.state.

set -uo pipefail   # NOT -e: we want to continue past failures
cd "$(dirname "$0")/.."

export PATH="$PWD/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1   # datasets are cached; avoid network stalls during pilot

LOG=results/logs/overnight.log
STATE=results/logs/overnight.state
mkdir -p results/logs
: > "$STATE"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
mark() { echo "$1" >> "$STATE"; say "=== $1 ==="; }

G6='training.batch_size=1 training.gradient_checkpointing=true'

# ─────────────────────────────────────────────────────────────────────────────
mark "PHASE_PILOT_START"
# A standalone pilot job may already be running (it owns the GPU). If no pilot
# process is alive, launch the runs here. Either way, resume-safe: run_pilot.sh
# skips existing JSONs. We expect 12 main + 3 alt-order = 15 JSONs.
EXPECTED_PILOT=15
launch_pilot() {
    GRAD_CKPT=1 BATCH_SIZE=1 CKA_N=100 FISHER_N=50 \
        bash scripts/run_pilot.sh >> "$LOG" 2>&1 || say "pilot main returned nonzero (continuing)"
    GRAD_CKPT=1 BATCH_SIZE=1 CKA_N=100 FISHER_N=50 ORDER="2 1 0" CONDITIONS="c4_full" \
        bash scripts/run_pilot.sh >> "$LOG" 2>&1 || say "pilot alt-order returned nonzero (continuing)"
}

if pgrep -f "doccl.pilot.run_pilot" >/dev/null 2>&1; then
    say "Existing pilot process detected — waiting for it to finish, then filling gaps."
    # Poll until the running job stops, then launch to fill any remaining runs.
    while pgrep -f "doccl.pilot.run_pilot" >/dev/null 2>&1; do
        n=$(ls results/pilot/*.json 2>/dev/null | wc -l)
        say "  pilot in progress: $n/$EXPECTED_PILOT JSONs"
        sleep 300
    done
fi

# Launch (or resume) until all expected JSONs exist. Guard against infinite loop.
attempts=0
while [ "$(ls results/pilot/*.json 2>/dev/null | wc -l)" -lt "$EXPECTED_PILOT" ] && [ "$attempts" -lt 3 ]; do
    attempts=$((attempts + 1))
    say "Launching pilot to fill remaining runs (attempt $attempts)."
    launch_pilot
done

n_json=$(ls results/pilot/*.json 2>/dev/null | wc -l)
say "pilot produced $n_json/$EXPECTED_PILOT JSON files"
mark "PHASE_PILOT_DONE"

# ─────────────────────────────────────────────────────────────────────────────
mark "PHASE_ANALYZE_START"
python -m doccl.pilot.analyze --pilot_dir results/pilot >> "$LOG" 2>&1 \
    || say "analyze returned nonzero (continuing)"
mark "PHASE_ANALYZE_DONE"

# ─────────────────────────────────────────────────────────────────────────────
mark "PHASE_GATE_A_START"
DECISION=$(python scripts/gate_a_decision.py results/pilot/findings_summary.md 2>>"$LOG" | tail -1)
say "GATE A decision: $DECISION"

RUN_PROPOSED=1
case "$DECISION" in
    A) WINNER=doccl_a; WINNER_CLASS=DocCL_A ;;
    B) WINNER=doccl_b; WINNER_CLASS=DocCL_B ;;
    C) WINNER=doccl_c; WINNER_CLASS=DocCL_C ;;
    *) RUN_PROPOSED=0; say "GATE A inconclusive -> baselines only (Phases 3+4), pausing proposed method per user." ;;
esac

if [ "$RUN_PROPOSED" = "1" ]; then
    say "Wiring doccl alias -> $WINNER_CLASS ($WINNER.yaml)"
    # 1) Repoint METHOD_REGISTRY["doccl"] in scripts/train.py
    sed -i -E "s/^(\s*)\"doccl\":\s*DocCL_[ABC],/\1\"doccl\": ${WINNER_CLASS},/" scripts/train.py
    # 2) Copy winner config body into doccl.yaml (keep name: doccl)
    python - "$WINNER" <<'PY'
import sys, pathlib
winner = sys.argv[1]
body = pathlib.Path(f"configs/method/{winner}.yaml").read_text().splitlines()
out = ["# AUTO-WIRED at GATE A from {}.yaml (overnight run).".format(winner)]
for line in body:
    if line.strip().startswith("name:"):
        out.append("name: doccl")
    elif line.startswith("#"):
        continue
    else:
        out.append(line)
pathlib.Path("configs/method/doccl.yaml").write_text("\n".join(out) + "\n")
print("wrote configs/method/doccl.yaml from", winner)
PY
    git add scripts/train.py configs/method/doccl.yaml
    git commit -m "AGENT IMPL: GATE A -> wire doccl alias to ${WINNER_CLASS} (overnight auto-decision)" >> "$LOG" 2>&1 \
        || say "gate-a commit skipped"
fi
mark "PHASE_GATE_A_DONE"

# ─────────────────────────────────────────────────────────────────────────────
# Grid: ONLINE W&B to thanh-workspace/CL4IE.
set -a; source .env 2>/dev/null || true; set +a
GRID_EXTRA="$G6 wandb.project=CL4IE"

mark "PHASE_GRID3_START"
PHASE=3 WANDB_MODE=online EXTRA="$GRID_EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 \
    || say "grid phase 3 returned nonzero (continuing)"
mark "PHASE_GRID3_DONE"

mark "PHASE_GRID4_START"
PHASE=4 WANDB_MODE=online EXTRA="$GRID_EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 \
    || say "grid phase 4 returned nonzero (continuing)"
mark "PHASE_GRID4_DONE"

if [ "$RUN_PROPOSED" = "1" ]; then
    mark "PHASE_GRID5_START"
    PHASE=5 WANDB_MODE=online EXTRA="$GRID_EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 \
        || say "grid phase 5 returned nonzero (continuing)"
    mark "PHASE_GRID5_DONE"

    mark "PHASE_ABLATION_START"
    PHASE=ablation WANDB_MODE=online EXTRA="$GRID_EXTRA" bash scripts/run_grid.sh >> "$LOG" 2>&1 \
        || say "ablation returned nonzero (continuing)"
    mark "PHASE_ABLATION_DONE"
else
    say "Skipping Phase 5 + ablation (GATE A inconclusive)."
fi

# ─────────────────────────────────────────────────────────────────────────────
mark "PHASE_AGGREGATE_START"
python scripts/analyze_results.py >> "$LOG" 2>&1 || say "analyze_results returned nonzero (continuing)"
python scripts/ingest_to_thesis.py >> "$LOG" 2>&1 || say "ingest returned nonzero (continuing)"
mark "PHASE_AGGREGATE_DONE"

mark "ALL_DONE"
say "Overnight pipeline finished. Decision=$DECISION  pilot_json=$n_json"
