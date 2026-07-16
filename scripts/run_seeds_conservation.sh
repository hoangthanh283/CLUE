#!/usr/bin/env bash
# Multi-seed the conservation finding (EXPLORE.md §7): 2 methods x 2 selections x seeds 7,123.
# Sequential — the local box fits exactly one dataset builder. Resume-safe: skips completed runs.
# Read results: python3 scripts/summarize_conservation.py
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # $1=method $2=seed $3=extra $4=suffix
  local out="results/dil_$1_seed$2_k4_d50_r128$4"
  [ -f "$out/metrics.json" ] && { echo "SKIP $out"; return; }
  rm -rf "$out"
  uv run python scripts/train.py method=$1 scenario=dil model=layoutlmv3_base seed=$2 \
    method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 $3 \
    training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 \
    wandb.mode=offline hydra.run.dir="$out" > "$out.log" 2>&1
  echo "DONE $out exit=$?"
}
for s in 7 123; do
  run colar     "$s" ""                                                        ""
  run colar     "$s" "+method.doc_selection=kcenter +method.selection_pool=100" "_kc"
  run colar_bal "$s" "method.soft_labels=true method.soft_T=2.0"                ""
  run colar_bal "$s" "method.soft_labels=true method.soft_T=2.0 +method.doc_selection=kcenter +method.selection_pool=100" "_kc"
done
echo "SEED GRID COMPLETE"
