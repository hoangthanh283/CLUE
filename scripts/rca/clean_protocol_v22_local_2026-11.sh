#!/usr/bin/env bash
# Clean-protocol queue v22 (2026-11-15, Dev, LOCAL box): only the three BROS oracle-pre runs of v19's b1
# recipe (batch 1, replay batch 2, fp32, expandable segments), resume-safe, then it ends. Everything else of
# v19 (BROS const_o / plain CoLaR, rank-64, Mixed joint, the v13 list) moves to the rented box (v21).
# Takes the lock first, then waits for the orphaned job of the driver it replaces and logs its DONE/FAIL.
#   setsid nohup bash scripts/rca/clean_protocol_v22_local_2026-11.sh [pid-to-wait-for] [its-run-name] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; ORPHAN=${2:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60 9>&-; done; fi
if [ -n "$ORPHAN" ]; then
  if ls $P/${ORPHAN}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $ORPHAN" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $ORPHAN" >>$Q; fi
fi
# v19's T / LR (BROS b1 recipe), unchanged
T="training.batch_size=1 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="scenario=cil_cord model=bros_base"
echo "$(date '+%m-%d %H:%M') v22 local driver running (pid $$): BROS oracle-pre b1 x3 only" >>$Q
for s in 42 7 123; do run cil_cord_colar_r_oracle_pre_bros_b1_seed$s method=colar_r_oracle_pre method.run_alias=colar_r_oracle_pre_bros_b1 method.docs_per_task=50 $LR $B seed=$s; done
echo "$(date '+%m-%d %H:%M') v22 done (local queue ends here; the rest runs on the rented box)" >>$Q
