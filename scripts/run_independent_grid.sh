#!/bin/bash
# Independent-model baselines across backbones for the DIL datasets.
# Supplies the "Independent" column in Table 1 (one model per dataset, no CL).
# LayoutLMv3 cells already cached in results/table_single_task_baselines.csv.
# Default run names: single_${ds}_naive_seed${sd}[_${family}]
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/indep_queue.log
mkdir -p results/logs_vision
MODELS="${MODELS:-lilt_base bros_base bert_base}"
DATASETS="${DATASETS:-funsd sroie cord}"
SEEDS="${SEEDS:-42 7 123}"
EPOCHS="${EPOCHS:-15}"
BS="${BS:-1}"
echo "=== INDEPENDENT GRID $(date) ===" >>"$LOG"
for model in $MODELS; do
  fam="${model%%_*}"  # lilt_base -> lilt
  for ds in $DATASETS; do
    for sd in $SEEDS; do
      d="results/single_${ds}_naive_seed${sd}_${fam}"
      tag="${ds}/${fam}/seed${sd}"
      if [ -f "${d}/metrics.json" ]; then
        echo "$(date +%H:%M) SKIP $tag" >>"$LOG"; continue
      fi
      echo "$(date +%H:%M) START $tag" >>"$LOG"
      uv run python scripts/train.py method=naive scenario=single_${ds} model=${model} \
        seed=${sd} method.epochs=${EPOCHS} training.batch_size=${BS} \
        training.gradient_checkpointing=true training.num_workers=0 \
        wandb.mode=offline \
        > "results/logs_vision/single_${ds}_naive_seed${sd}_${fam}.log" 2>&1
      [ -f "${d}/metrics.json" ] && echo "$(date +%H:%M) DONE  $tag" >>"$LOG" \
                                 || echo "$(date +%H:%M) FAIL  $tag" >>"$LOG"
    done
  done
done
echo "=== INDEPENDENT GRID COMPLETE $(date) ===" >>"$LOG"
