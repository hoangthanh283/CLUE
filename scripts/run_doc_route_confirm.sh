#!/bin/bash
# Amendment 10 confirmation: the only routed arm that cleared +1 pp on seed 42 (far, d=50) on seeds 7/123.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/doc_route_queue.log; mkdir -p results/logs_vision
SEEDS="${SEEDS:-7 123}"; ARMS="${ARMS:-docRfar_d50:far:50}"
V="method=colar_pp scenario=dil model=layoutlmv3_base method.epochs=5 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 training.batch_size=2 training.num_workers=0 training.gradient_checkpointing=true"
echo "=== DOC ROUTE CONFIRM $(date) ===" >>"$LOG"
for arm in $ARMS; do a="${arm%%:*}"; rest="${arm#*:}"; r="${rest%%:*}"; dpt="${rest#*:}"
  for sd in $SEEDS; do d="results/dil_${a}_seed${sd}"
    if ls ${d}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $a seed$sd" >>"$LOG"; continue; fi
    echo "$(date +%H:%M) START $a seed$sd" >>"$LOG"
    uv run python scripts/train.py $V seed=$sd +method.run_alias=$a method.docs_per_task=$dpt method.replay_route=$r > "results/logs_vision/dil_${a}_seed${sd}.log" 2>&1
    ls ${d}*/metrics.json >/dev/null 2>&1 && echo "$(date +%H:%M) DONE  $a seed$sd" >>"$LOG" || echo "$(date +%H:%M) FAIL  $a seed$sd" >>"$LOG"
  done; done
echo "=== DOC ROUTE CONFIRM COMPLETE $(date) ===" >>"$LOG"
