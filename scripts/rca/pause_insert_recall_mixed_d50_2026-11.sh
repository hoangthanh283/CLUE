#!/usr/bin/env bash
# Mixed s42 SBR-Recall r=64 WITH the bank (d50): d0 under the per-delta fix gave 44.7 — cross-schema
# all-O terms have no positive counterweight without replay, blocks learn "always O". Same pattern.
set -u
cd "$(dirname "$0")/../.."
XPID=$1; TPID=$2; P=results/gates/cil_protocol_2026-11; Q=$P/queue.log
trap 'kill -CONT $XPID; echo "$(date "+%m-%d %H:%M") RESUME rescore (xargs $XPID)" >>$Q' EXIT
while kill -0 "$TPID" 2>/dev/null; do sleep 30; done
NAME=mixed_colar_sbr_recall_r64c_seed42
ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && exit 0
echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
uv run python scripts/train.py method=colar_sbr_recall +method.run_alias=colar_sbr_recall_r64c method.lora_rank=64 method.lora_alpha=128 'method.target_modules=[query,key,value,dense]' method.docs_per_task=50 method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 scenario=mixed model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/$NAME.log" 2>&1
if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
