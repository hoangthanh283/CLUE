#!/usr/bin/env bash
# CoLaR-SBR + LexSlot on Mixed and DIL, seeds 42/7/123 (user: "run fully on other datasets").
# Chains after pause_insert_sbr_slot_seeds (arg 2 = its PID): when it exits the rescore xargs
# resumes and starts one job; we wait for that job, STOP xargs again, run, CONT on exit.
set -u
cd "$(dirname "$0")/../.."
XPID=$1; PREV=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
while kill -0 "$PREV" 2>/dev/null; do sleep 60; done
sleep 120  # let xargs launch the next rescore job
TPID=$(pgrep -f "^uv run python scripts/train.py" | head -1)
kill -STOP "$XPID"
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
echo "$(date '+%m-%d %H:%M') sbr_slot_all: waiting on train PID ${TPID:-none}" >>$Q
while [ -n "${TPID:-}" ] && kill -0 "$TPID" 2>/dev/null; do sleep 30; done
for SC in mixed dil; do for SEED in 42 7 123; do
  NAME=${SC}_colar_sbr_slot_seed$SEED
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && continue
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py method=colar_sbr_slot method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=$SC model=layoutlmv3_base seed=$SEED training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
done; done
