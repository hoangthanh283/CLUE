#!/usr/bin/env bash
# Insert jobs between two rescore jobs without a second GPU job: the caller SIGSTOPs the
# rescore's xargs (it cannot launch the next job), this waits for the running train PID,
# runs the AMP smoke + the SBR v2 DIL seeds, then SIGCONTs xargs (always, via trap).
#   kill -STOP <xargs>; setsid nohup bash scripts/rca/pause_insert_2026-11.sh <xargs> <train pid> &
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
echo "$(date '+%m-%d %H:%M') START amp_smoke" >>$Q
if uv run python scripts/rca/amp_smoke_2026-11.py >$P/amp_smoke.log 2>&1; then
  echo "$(date '+%m-%d %H:%M') DONE  amp_smoke (PASS)" >>$Q
else
  echo "$(date '+%m-%d %H:%M') FAIL  amp_smoke -> see $P/amp_smoke.log; DIL seeds skipped" >>$Q; exit 1
fi
for SD in 7 123; do
  NAME=dil_colar_sbr2_seed$SD
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && continue
  echo "$(date '+%m-%d %H:%M') START $NAME (V2 seeds)" >>$Q
  uv run python scripts/train.py method=colar_sbr +method.run_alias=colar_sbr2 method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=dil model=layoutlmv3_base seed=$SD training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
done
