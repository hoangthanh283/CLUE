#!/bin/bash
# Tier A of plan rev. 8 (2026-10-21): fill the document-side cells of the frozen 12-row
# baseline set on the local box. Serial, flock-guarded, resume-safe (skips cells whose
# metrics.json exists). Waits for the Phase-1 queue (scripts/rca/phase1_queue.sh) to finish,
# runs the 19 ungated cells, then reads the two Phase-1 results and applies the decision
# table for CPFD / CODA-Prompt automatically.
# Log: results/logs_vision/final12_tierA.log
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/final12_tierA.log
exec 9>/tmp/claude-1000/final12_tierA.lock
flock -n 9 || { echo "tierA already running" >&2; exit 1; }
mkdir -p results/logs_vision
COMMON="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"

# Expected run dir mirrors train.py: <scenario>_<run_alias|method>_seed<N> (LayoutLMv3 => no family suffix).
run_dir() {  # method scenario seed
  local alias; alias=$(sed -nE 's/^run_alias:[[:space:]]*([A-Za-z0-9_]+).*/\1/p' "configs/method/$1.yaml" | head -1)
  echo "results/${2}_${alias:-$1}_seed${3}"
}

run_cell() {  # method scenario seed [extra overrides...]
  local me="$1" sc="$2" sd="$3"; shift 3
  local dir; dir=$(run_dir "$me" "$sc" "$sd")
  if [ -f "$dir/metrics.json" ]; then echo "$(date +%H:%M) SKIP $dir" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $dir $*" >>"$LOG"
  uv run python scripts/train.py method="$me" scenario="$sc" model=layoutlmv3_base seed="$sd" \
    $COMMON "$@" >"results/logs_vision/$(basename "$dir").log" 2>&1
  if [ -f "$dir/metrics.json" ]; then
    local aa; aa=$(python3 -c "import json;print(round(json.load(open('$dir/metrics.json'))['AA'],2))")
    echo "$(date +%H:%M) DONE  $dir AA=$aa" >>"$LOG"
  else
    echo "$(date +%H:%M) FAIL  $dir" >>"$LOG"
  fi
}

# AA>2 on every diagonal cell AND overall -> "converged" for the Phase-1 gate.
gate_ok() {  # glob of a phase1 metrics.json
  python3 - "$@" <<'PY'
import json, sys, glob
fs = [f for g in sys.argv[1:] for f in glob.glob(g)]
if not fs: print("NOFILE"); sys.exit(0)
m = json.load(open(fs[0])); M = m["matrix"]
ok = m["AA"] > 2 and all(M[i][i] > 2 for i in range(len(M)))
print("OK" if ok else "COLLAPSE", round(m["AA"], 2), [round(M[i][i], 1) for i in range(len(M))])
PY
}

until grep -q 'PHASE1B COMPLETE' results/logs_vision/phase1_queue.log 2>/dev/null; do sleep 120; done
echo "=== TIER A $(date) ===" >>"$LOG"

# ---- ungated cells (cheapest / most certain first) ----
for sd in 42 7 123; do run_cell lca dil      "$sd"; done
for sd in 42 7 123; do run_cell lca cil_cord "$sd"; done
for sd in 42 7 123; do run_cell lca mixed    "$sd"; done
run_cell cl_lora cil_cord 123
for sd in 42 7 123; do run_cell is3 dil "$sd"; done

# ---- Phase-1 gated cells (decision table, plan rev. 8) ----
CPFD_GATE=$(gate_ok "results/phase1/cil_cord_cpfd*seed42*/metrics.json")
CODA_GATE=$(gate_ok "results/phase1/cil_cord_coda_prompt*seed42*/metrics.json")
echo "$(date +%H:%M) GATE cpfd_ceMin05: $CPFD_GATE" >>"$LOG"
echo "$(date +%H:%M) GATE coda_n50:    $CODA_GATE" >>"$LOG"

CPFD_EXTRA=""; case "$CPFD_GATE" in OK*) CPFD_EXTRA="+method.adaptive_ce_min=0.5";; esac
for sd in 42 7 123; do run_cell cpfd dil "$sd" $CPFD_EXTRA; done   # DIL runs regardless
case "$CPFD_GATE" in
  OK*) for sc in cil_cord mixed; do for sd in 42 7 123; do run_cell cpfd "$sc" "$sd" $CPFD_EXTRA; done; done;;
  *)   echo "$(date +%H:%M) NULL  cpfd cil_cord+mixed (disclosure; see plan rev. 8)" >>"$LOG";;
esac
case "$CODA_GATE" in
  OK*) for sd in 42 7 123; do run_cell coda_prompt cil_cord "$sd" method.n_components=50; done;;
  *)   echo "$(date +%H:%M) NULL  coda_prompt cil_cord (disclosure; see plan rev. 8)" >>"$LOG";;
esac
echo "=== TIER A COMPLETE $(date) ===" >>"$LOG"
