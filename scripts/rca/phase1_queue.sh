#!/bin/bash
# Phase 1 (plan rev. 7): two no-source-edit discriminating runs on CIL-CORD seed 42.
#  1. CPFD with the CE adaptive-factor floor raised (adaptive_ce_min 0.0 -> 0.5), 2 ep/task:
#     tests whether the "CE prints 0.000 from task 1" collapse is the vanishing CE factor.
#  2. CODA-Prompt with 50 prompt components instead of the static 10 (5 sessions x 10):
#     tests the prompt-pool-starvation hypothesis.
# Outputs go under results/phase1/ so no existing cell is touched. Serial, flock-guarded.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/phase1_queue.log
exec 9>/tmp/claude-1000/phase1.lock
flock -n 9 || { echo "phase1 already running" >&2; exit 1; }
mkdir -p results/logs_vision results/phase1
COMMON="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=results/phase1"
echo "=== PHASE1 $(date) ===" >>"$LOG"

echo "$(date +%H:%M) START cpfd_ceMin05_e2" >>"$LOG"
uv run python scripts/train.py method=cpfd scenario=cil_cord model=layoutlmv3_base seed=42 \
  method.epochs=2 +method.adaptive_ce_min=0.5 $COMMON >results/logs_vision/phase1_cpfd_ceMin05.log 2>&1
ls results/phase1/cil_cord_cpfd_seed42/metrics.json >/dev/null 2>&1 \
  && echo "$(date +%H:%M) DONE  cpfd_ceMin05_e2" >>"$LOG" || echo "$(date +%H:%M) FAIL  cpfd_ceMin05_e2" >>"$LOG"

echo "$(date +%H:%M) START coda_n50" >>"$LOG"
uv run python scripts/train.py method=coda_prompt scenario=cil_cord model=layoutlmv3_base seed=42 \
  method.n_components=50 $COMMON >results/logs_vision/phase1_coda_n50.log 2>&1
ls results/phase1/cil_cord_coda_prompt_seed42/metrics.json >/dev/null 2>&1 \
  && echo "$(date +%H:%M) DONE  coda_n50" >>"$LOG" || echo "$(date +%H:%M) FAIL  coda_n50" >>"$LOG"

echo "=== PHASE1 COMPLETE $(date) ===" >>"$LOG"
