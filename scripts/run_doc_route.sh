#!/bin/bash
# Amendment 10: lexically-routed latent replay on DIL / LayoutLMv3, seed 42.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/doc_route_queue.log; mkdir -p results/logs_vision
V="method=colar_pp scenario=dil model=layoutlmv3_base seed=42 method.epochs=5 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 training.batch_size=2 training.num_workers=0 training.gradient_checkpointing=true"
run() { local a="$1"; shift; local d="results/dil_${a}_seed42"
  if ls ${d}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $a" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $a ($*)" >>"$LOG"
  uv run python scripts/train.py $V +method.run_alias=$a "$@" > "results/logs_vision/dil_${a}_seed42.log" 2>&1
  ls ${d}*/metrics.json >/dev/null 2>&1 && echo "$(date +%H:%M) DONE  $a" >>"$LOG" || echo "$(date +%H:%M) FAIL  $a" >>"$LOG"; }
echo "=== DOC ROUTE QUEUE $(date) ===" >>"$LOG"
# Memory-only control at d=5 (uniform sampling, same budget); the d=50 control is
# results/dil_colar_seed42_k4_d50_r128 (AA 86.4).
run "docU_d5" method.docs_per_task=5 method.replay_route=none
for dpt in 50 5; do for r in near far task; do
  run "docR${r}_d${dpt}" method.docs_per_task=$dpt method.replay_route=$r
done; done
echo "=== DOC ROUTE QUEUE COMPLETE $(date) ===" >>"$LOG"
