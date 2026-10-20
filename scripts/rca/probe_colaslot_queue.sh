#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_colaslot.log
LOCK=/tmp/claude-1000/pass2b_colaslot.lock
exec 9>"$LOCK"
if ! flock -n 9; then echo "already running"; exit 0; fi
mkdir -p results/logs_vision
echo "=== PASS 2B COLASLOT $(date) ===" >>"$LOG"

# Shadow-replay at d50 r128 is the one surviving CoLaSlot variant (86.6) — probe its ridge
# ceiling to see whether the slot machinery adds anything over plain CoLaR (ridge 91.3).
NAME=dil_colaslot_shadow_replay_pass2b_seed42
echo "$(date +%H:%M) TRAIN ${NAME}" >>"$LOG"
uv run python scripts/train.py method=colaslot_shadow scenario=dil model=layoutlmv3_base \
  seed=42 method.epochs=5 training.batch_size=2 training.gradient_checkpointing=true \
  training.num_workers=0 wandb.mode=offline method.split_layer_k=4 method.rank_r=128 \
  method.docs_per_task=50 method.replay_batch_size=4 method.store_input_ids=true \
  +method.dump_final=true hydra.run.dir=results/${NAME} \
  >"results/logs_vision/${NAME}_train.log" 2>&1

# The dump lands at the actual output dir chosen by the loop (same hydra-vs-method-name issue).
# Figure out where final_model.pt landed.
for cand in "results/${NAME}" results/dil_colaslot_shadow_seed42 results/dil_colaslot_shadow_replay_seed42; do
  if [ -f "$cand/final_model.pt" ]; then
    DUMP="$cand"; break
  fi
done
if [ -z "${DUMP:-}" ]; then
  echo "$(date +%H:%M) TRAIN_FAIL ${NAME} (no final_model.pt found)" >>"$LOG"
  exit 1
fi
echo "$(date +%H:%M) PROBE ${DUMP}" >>"$LOG"
uv run python scripts/rca/probe_cell.py --run "${DUMP}" --scenario dil --k 4 \
  >"results/logs_vision/${NAME}_probe.log" 2>&1
if [ -f "${DUMP}/probes.json" ]; then
  echo "$(date +%H:%M) DONE ${DUMP}" >>"$LOG"
else
  echo "$(date +%H:%M) FAIL ${DUMP}" >>"$LOG"
fi
echo "=== PASS 2B COLASLOT COMPLETE $(date) ===" >>"$LOG"
