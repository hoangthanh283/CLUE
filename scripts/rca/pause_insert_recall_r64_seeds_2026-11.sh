#!/usr/bin/env bash
# SBR-Recall r=64 passed (CIL-CORD s42 AA 92.7 > SBR 85.2, > joint). Plan rule: seeds 7/123 + Mixed s42.
# Same pause-insert pattern: kill -STOP <xargs>; this waits for the running train PID, runs, SIGCONTs.
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
AX="method=colar_sbr_recall +method.run_alias=colar_sbr_recall_r64 method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense] method.docs_per_task=0 method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 model=layoutlmv3_base training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true"
for SPEC in "cil_cord 7" "cil_cord 123" "mixed 42"; do
  SC=${SPEC% *}; SD=${SPEC#* }; NAME=${SC}_colar_sbr_recall_r64_seed$SD
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && continue
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py $AX scenario=$SC seed=$SD >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
done
