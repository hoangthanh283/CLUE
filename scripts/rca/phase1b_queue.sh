#!/bin/bash
# Phase 1b: second (and last) no-source-edit CPFD discriminating run. The CE-floor run
# (phase1) still collapsed at task 1 (new-task F1 0.00), so the vanishing CE factor is not
# the cause. This run switches the distillation term off (distill_weight=0): if the new
# task is learned, the KL+attention distillation magnitude is what suppresses new classes;
# if it still collapses, CPFD is not salvageable by any knob -> null with disclosure.
# Informational only: Tier A's gate keys on the phase1 (ceMin05) result, not this one.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/phase1_queue.log
exec 9>/tmp/claude-1000/phase1b.lock
flock -n 9 || { echo "phase1b already running" >&2; exit 1; }
until grep -q 'PHASE1 COMPLETE' "$LOG" 2>/dev/null; do sleep 60; done
COMMON="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=results/phase1"
echo "$(date +%H:%M) START cpfd_dw0_e2" >>"$LOG"
uv run python scripts/train.py method=cpfd scenario=cil_cord model=layoutlmv3_base seed=42 \
  method.epochs=2 method.distill_weight=0 +method.run_alias=cpfd_dw0 $COMMON \
  >results/logs_vision/phase1_cpfd_dw0.log 2>&1
ls results/phase1/cil_cord_cpfd_dw0_seed42/metrics.json >/dev/null 2>&1 \
  && echo "$(date +%H:%M) DONE  cpfd_dw0_e2" >>"$LOG" || echo "$(date +%H:%M) FAIL  cpfd_dw0_e2" >>"$LOG"
echo "=== PHASE1B COMPLETE $(date) ===" >>"$LOG"
