#!/usr/bin/env bash
# Insert the SBR-Recall gate between two rescore jobs (same pattern as pause_insert_2026-11.sh):
# caller SIGSTOPs the rescore's xargs, this waits for the running train PID, runs the recall
# smoke, then the CIL-CORD seed-42 gate (d50) and its no-bank ablation (d0), then SIGCONTs xargs.
#   kill -STOP <xargs>; setsid nohup bash scripts/rca/pause_insert_recall_2026-11.sh <xargs> <train pid> &
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
echo "$(date '+%m-%d %H:%M') START recall_smoke" >>$Q
if uv run python scripts/rca/recall_smoke_2026-11.py >$P/recall_smoke.log 2>&1; then
  echo "$(date '+%m-%d %H:%M') DONE  recall_smoke (PASS)" >>$Q
else
  echo "$(date '+%m-%d %H:%M') FAIL  recall_smoke -> see $P/recall_smoke.log; gate skipped" >>$Q; exit 1
fi
COMMON="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true"
for SPEC in "cil_cord_colar_sbr_recall_seed42 method.docs_per_task=50" "cil_cord_colar_sbr_recall_d0_seed42 method.docs_per_task=0 +method.run_alias=colar_sbr_recall_d0"; do
  NAME=${SPEC%% *}; ARGS=${SPEC#* }
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && continue
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py method=colar_sbr_recall $COMMON $ARGS >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
done
