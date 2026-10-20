#!/bin/bash
# Probe LexSlot cells at their actual landed paths (dil_lexslot_seed42,
# cil_cord_lexslot_seed42, mixed_lexslot_seed42). Waits for pass2b_single
# to finish all LexSlot training first.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_lexslot_probe.log
LOCK=/tmp/claude-1000/pass2b_lexslot_probe.lock
exec 9>"$LOCK"
if ! flock -n 9; then echo already; exit 0; fi
echo "=== LEXSLOT PROBE $(date) ===" >>"$LOG"
probe() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then
    echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ ! -f "results/${name}/final_model.pt" ]; then
    echo "$(date +%H:%M) MISS ${name}" >>"$LOG"; return; fi
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do sleep 60; done
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
    >"results/logs_vision/${name}_probe.log" 2>&1
  [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
    || echo "$(date +%H:%M) FAIL ${name}" >>"$LOG"
}
# Wait for the single queue to finish its three LexSlot training cells
until grep -q 'PASS 2B SINGLE COMPLETE' results/logs_vision/pass2b_single.log 2>/dev/null; do sleep 120; done
probe dil_lexslot_seed42       dil
probe cil_cord_lexslot_seed42  cil_cord
probe mixed_lexslot_seed42     mixed
echo "=== LEXSLOT PROBE COMPLETE $(date) ===" >>"$LOG"
