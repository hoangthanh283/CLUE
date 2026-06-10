#!/usr/bin/env bash
# Watch results/pilot/ and backfill new pilot JSONs to W&B as they complete.
# Decoupled from the pilot runner and the overnight orchestrator. Idempotent:
# pilot_to_wandb.py skips runs already present in W&B. Exits once the pilot is
# done (orchestrator past PHASE_PILOT_DONE) AND all current JSONs are logged.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
set -a; source .env 2>/dev/null || true; set +a

LOG=results/logs/pilot_wandb.log
mkdir -p results/logs
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

say "pilot->W&B watcher started (project=$WANDB_PROJECT entity=$WANDB_ENTITY)"
while true; do
    n=$(ls results/pilot/*.json 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        python scripts/pilot_to_wandb.py \
            --pilot_dir results/pilot \
            --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" >> "$LOG" 2>&1 \
            || say "backfill attempt errored (will retry)"
    fi
    # Stop once pilot phase is complete and everything is logged.
    if grep -q "PHASE_PILOT_DONE" results/logs/overnight.state 2>/dev/null; then
        # one final sweep, then exit
        python scripts/pilot_to_wandb.py --pilot_dir results/pilot \
            --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" >> "$LOG" 2>&1 || true
        say "pilot phase done; final backfill complete; watcher exiting ($n JSONs)."
        break
    fi
    sleep 180
done
