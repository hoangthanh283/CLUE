#!/bin/bash
# CoLaR OOM retry for BROS and LiLT cells that the backbone queue left without metrics.json.
# BROS OOMs at batch 2 (5.52 GB, 2026-10-26); LiLT OOMs on CIL-CORD at batch 2 (5.38 GB,
# 2026-10-28 — 283M params + optimizer state + replay buffer). Recipe: batch 1, grad-ckpt, AMP,
# replay_batch_size 2. Chained after the LexSlot doc queue. Resume-safe; disk-guarded.
set -u; cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/colar_backbones.log
until grep -q 'OURS DOC QUEUE COMPLETE' results/logs_vision/ours_doc_queue.log 2>/dev/null; do sleep 300; done
echo "=== COLAR OOM RETRY (bs1 + amp) $(date) ===" >>"$LOG"
for fam in bros lilt; do
  model="${fam}_base"
  for sc in dil cil_cord mixed; do for sd in 42 7 123; do
    name="${sc}_colar_e10_seed${sd}_${fam}"
    if ls results/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
    until [ "$(df -m /mnt/DataDrive | awk 'NR==2{print $4}')" -gt 10000 ]; do sleep 600; done
    rm -rf results/${name}_k4_d50_r128
    echo "$(date +%H:%M) START $name (bs1+amp)" >>"$LOG"
    uv run python scripts/train.py method=colar +method.run_alias=colar_e10 scenario="$sc" model="$model" \
      seed="$sd" method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 \
      method.replay_batch_size=2 training.batch_size=1 training.gradient_checkpointing=true \
      training.amp=true training.num_workers=0 wandb.mode=offline >"results/logs_vision/$name.log" 2>&1
    if ls results/${name}*/metrics.json >/dev/null 2>&1; then
      echo "$(date +%H:%M) DONE  $name AA=$(python3 -c "import json,glob;print(round(json.load(open(glob.glob('results/${name}*/metrics.json')[0]))['AA'],2))")" >>"$LOG"
    else echo "$(date +%H:%M) FAIL  $name" >>"$LOG"; fi
  done; done
done
echo "=== COLAR OOM RETRY COMPLETE $(date) ===" >>"$LOG"
