#!/usr/bin/env bash
# SBR follow-up (2026-11-03): waits for the seed-42 gate; if AA >= 70, preempts the gate runner
# (controls can wait), runs seeds 7/123 on CIL-CORD + Mixed seed 42, then resumes the gate
# script (resume-safe: controls, E3, G0c, G0d, image gates).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=results/gates/cil_protocol_2026-11; LOG=$P/queue.log
until grep -q "DONE  cil_cord_colar_sbr_seed42\|FAIL  cil_cord_colar_sbr_seed42" $LOG; do sleep 60; done
AA=$(python3 -c "import json,glob;d=[x for x in glob.glob('$P/cil_cord_colar_sbr_seed42*') if not x.endswith('.log')];print(json.load(open(d[0]+'/metrics.json'))['AA'] if d else 0)")
echo "$(date '+%m-%d %H:%M') SBR gate AA=$AA" >>$LOG
if ! python3 -c "import sys;sys.exit(0 if float('$AA')>=70 else 1)"; then echo "$(date '+%m-%d %H:%M') SBR below 70: follow-up skipped" >>$LOG; exit 0; fi
# preempt the gate runner and whatever control it just started
for pid in $(pgrep -f "audit_gates_2026-11.sh"); do kill $pid 2>/dev/null; done; sleep 2
for pid in $(pgrep -f "run_alias=colar_sbr_noreplay\|run_alias=colar_sbr_frozen"); do kill $pid 2>/dev/null; done; sleep 5
rm -rf $P/cil_cord_colar_sbr_noreplay_seed42* $P/cil_cord_colar_sbr_frozen_seed42* 2>/dev/null
echo "$(date '+%m-%d %H:%M') PREEMPT: gate runner paused for SBR seeds/Mixed" >>$LOG
AX="method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
run() { local name=$1 sc=$2 sd=$3
  local COMMON="scenario=$sc model=layoutlmv3_base seed=$sd training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true"
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  uv run python scripts/train.py method=colar_sbr $AX $COMMON >"$P/$name.log" 2>&1
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
run cil_cord_colar_sbr_seed7   cil_cord 7
run cil_cord_colar_sbr_seed123 cil_cord 123
run mixed_colar_sbr_seed42     mixed    42
echo "=== SBR FOLLOW-UP COMPLETE $(date) ===" >>$LOG
bash scripts/rca/audit_gates_2026-11.sh
