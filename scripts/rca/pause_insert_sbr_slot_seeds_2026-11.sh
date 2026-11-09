#!/usr/bin/env bash
# CoLaR-SBR + LexSlot seeds 7/123 on CIL-CORD (s42 = 90.2 > SBR 87.6 by 2.6 -> rule met). Same recipe.
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
for SEED in 7 123; do
NAME=cil_cord_colar_sbr_slot_seed$SEED
ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && continue
echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
uv run python scripts/train.py method=colar_sbr_slot method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=cil_cord model=layoutlmv3_base seed=$SEED training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/$NAME.log" 2>&1
if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
done
