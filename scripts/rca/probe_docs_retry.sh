#!/bin/bash
# Retry the 3 doc CoLaR probes after probe_cell.py fix. Waits for LexSlot
# training slots to free (checks GPU idle).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_docs_retry.log
LOCK=/tmp/claude-1000/pass2b_docs_retry.lock
mkdir -p results/logs_vision /tmp/claude-1000
exec 9>"$LOCK"
if ! flock -n 9; then echo "already running"; exit 0; fi
echo "=== DOCS RETRY $(date) ===" >>"$LOG"

wait_for_gpu() {
  # Only run when nothing else is heavily using the GPU
  while true; do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${used:-0}" -lt 500 ] && return
    sleep 60
  done
}

probe() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then
    echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  wait_for_gpu
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
    >"results/logs_vision/${name}_probe.log" 2>&1
  [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
    || echo "$(date +%H:%M) FAIL ${name}" >>"$LOG"
}
# Wait for the main single queue to be fully done first
until grep -q 'PASS 2B SINGLE COMPLETE' results/logs_vision/pass2b_single.log 2>/dev/null; do sleep 120; done
probe dil_colar_pp_seed42       dil
probe cil_cord_colar_pp_seed42  cil_cord
probe mixed_colar_pp_seed42     mixed
echo "=== DOCS RETRY COMPLETE $(date) ===" >>"$LOG"
