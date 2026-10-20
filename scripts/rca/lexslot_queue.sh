#!/bin/bash
# Pass 2b LexSlot: re-run DIL / CIL-CORD / Mixed LexSlot seed 42 with
# dump_final=true (LayoutLMv3), then probe. Resume-safe. Expects CoLaR
# probe queue to be free.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_lexslot.log
mkdir -p results/logs_vision
echo "=== PASS 2B LEXSLOT $(date) ===" >>"$LOG"
run_cell() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then
    echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ ! -f "results/${name}/final_model.pt" ]; then
    echo "$(date +%H:%M) TRAIN ${name}" >>"$LOG"
    uv run python scripts/train.py method=lexslot scenario="${scen}" model=layoutlmv3_base \
      seed=42 training.batch_size=2 training.gradient_checkpointing=true \
      training.num_workers=0 wandb.mode=offline +method.dump_final=true \
      hydra.run.dir=results/${name} >"results/logs_vision/${name}_train.log" 2>&1
    [ ! -f "results/${name}/final_model.pt" ] && \
      echo "$(date +%H:%M) TRAIN_FAIL ${name}" >>"$LOG" && return
  fi
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
    >"results/logs_vision/${name}_probe.log" 2>&1
  [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
    || echo "$(date +%H:%M) FAIL ${name}" >>"$LOG"
}
# wait for CoLaR probe queue to finish
until grep -q 'PASS 2B QUEUE2 COMPLETE' results/logs_vision/pass2b_queue2.log 2>/dev/null; do sleep 60; done
run_cell dil_lexslot_pp_seed42        dil
run_cell cil_cord_lexslot_pp_seed42   cil_cord
run_cell mixed_lexslot_pp_seed42      mixed
echo "=== PASS 2B LEXSLOT COMPLETE $(date) ===" >>"$LOG"
