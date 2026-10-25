#!/bin/bash
# Proposed-method gap (2026-10-25): LexSlot (slot_sharing=off = the thesis/CVPR recipe that gave
# DIL 87.3) on CIL-CORD and Mixed x 3 seeds; only seed-42 plain-routing runs existed. Peaks ~5 GB
# VRAM, so it must own the GPU: waits for the peer's CoLaR backbone queue to finish. Resume-safe.
set -u; cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
LOG=results/logs_vision/ours_doc_queue.log; mkdir -p results/logs_vision
until grep -q 'COLAR BACKBONES COMPLETE' results/logs_vision/colar_backbones.log 2>/dev/null; do sleep 300; done
echo "=== OURS DOC QUEUE (lexslot off) $(date) ===" >>"$LOG"
for sc in cil_cord mixed; do for sd in 42 7 123; do
  d="results/${sc}_lexslot_seed${sd}_off"
  if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) SKIP $d" >>"$LOG"; continue; fi
  until [ "$(df -m /mnt/DataDrive | awk 'NR==2{print $4}')" -gt 10000 ]; do echo "$(date +%H:%M) waiting for disk" >>"$LOG"; sleep 600; done
  echo "$(date +%H:%M) START $d" >>"$LOG"
  uv run python scripts/train.py method=lexslot scenario=$sc seed=$sd method.slot_sharing=off \
    training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 \
    >"results/logs_vision/${sc}_lexslot_seed${sd}_off.log" 2>&1
  if [ -f "$d/metrics.json" ]; then echo "$(date +%H:%M) DONE  $d AA=$(python3 -c "import json;print(round(json.load(open('$d/metrics.json'))['AA'],2))")" >>"$LOG"; else echo "$(date +%H:%M) FAIL  $d" >>"$LOG"; fi
done; done
echo "=== OURS DOC QUEUE COMPLETE $(date) ===" >>"$LOG"
