#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_all_docs.log
LOCK=/tmp/claude-1000/pass2b_all_docs.lock
exec 9>"$LOCK"
if ! flock -n 9; then echo "already running"; exit 0; fi
echo "=== ALL DOCS PROBE $(date) ===" >>"$LOG"
probe() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ ! -f "results/${name}/final_model.pt" ]; then echo "$(date +%H:%M) MISS ${name}" >>"$LOG"; return; fi
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
    >"results/logs_vision/${name}_probe.log" 2>&1
  [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
    || echo "$(date +%H:%M) FAIL ${name}" >>"$LOG"
}
probe dil_lexslot_seed42         dil
probe cil_cord_lexslot_seed42    cil_cord
probe mixed_lexslot_seed42       mixed
probe dil_colar_pp_seed42        dil
probe cil_cord_colar_pp_seed42   cil_cord
probe mixed_colar_pp_seed42      mixed
echo "=== ALL DOCS PROBE COMPLETE $(date) ===" >>"$LOG"
