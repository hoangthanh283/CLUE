#!/bin/bash
# Pass 2b retry: probe already-dumped cells directly (training already done).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_queue2.log
mkdir -p results/logs_vision
echo "=== PASS 2B QUEUE2 $(date) ===" >>"$LOG"
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
probe dil_colar_pp_seed42         dil
probe cil_cord_colar_pp_seed42    cil_cord
probe mixed_colar_pp_seed42       mixed
probe cil_imagenet_r_colar_pp_seed42_vit cil_imagenet_r
echo "=== PASS 2B QUEUE2 COMPLETE $(date) ===" >>"$LOG"
