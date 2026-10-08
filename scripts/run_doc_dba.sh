#!/bin/bash
# Amendment 9: D and B cells on DIL / LayoutLMv3, seed 42, CoLaR r128 d50 5-epoch recipe.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/doc_dba_queue.log; mkdir -p results/logs_vision
V="method=colar_pp scenario=dil model=layoutlmv3_base seed=42 method.epochs=5 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4 training.batch_size=2 training.num_workers=0 training.gradient_checkpointing=true"
run() { local a="$1"; shift; local d="results/dil_${a}_seed42"
  if ls ${d}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $a" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $a ($*)" >>"$LOG"
  uv run python scripts/train.py $V +method.run_alias=$a "$@" > "results/logs_vision/dil_${a}_seed42.log" 2>&1
  ls ${d}*/metrics.json >/dev/null 2>&1 && echo "$(date +%H:%M) DONE  $a" >>"$LOG" || echo "$(date +%H:%M) FAIL  $a" >>"$LOG"; }
echo "=== DOC D/B QUEUE $(date) ===" >>"$LOG"
run docD1 method.docs_per_task=50 method.keep_tokens=labeled method.dump_final=true
run docD2 method.docs_per_task=50 method.keep_tokens=labeled method.visual_tokens=drop
run docB1 method.docs_per_task=5  method.drift_comp=true method.head_align_epochs=10
run docB2 method.docs_per_task=50 method.drift_comp=true method.head_align_epochs=10
echo "=== DOC D/B QUEUE COMPLETE $(date) ===" >>"$LOG"
