#!/usr/bin/env bash
# CoLaR-SBR + LexSlot (user request 2026-11-09): one seed-42 CIL-CORD run, SBR gate recipe
# (k4/d50/r128/e10, fp32, restricted scoring). Smoke first. Same pause-insert pattern.
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
echo "$(date '+%m-%d %H:%M') START sbr_slot_smoke" >>$Q
if ! uv run python scripts/rca/sbr_slot_smoke_2026-11.py >$P/sbr_slot_smoke.log 2>&1; then
  echo "$(date '+%m-%d %H:%M') FAIL  sbr_slot_smoke -> $P/sbr_slot_smoke.log" >>$Q; exit 1
fi
echo "$(date '+%m-%d %H:%M') DONE  sbr_slot_smoke" >>$Q
NAME=cil_cord_colar_sbr_slot_seed42
ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && exit 0
echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
uv run python scripts/train.py method=colar_sbr_slot method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/$NAME.log" 2>&1
if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
