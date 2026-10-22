#!/bin/bash
# Re-run the image grid once the first pass ends, so the six CIFAR-100 CODA-Prompt /
# CL-LoRA cells that crashed before the 2026-10-22 ViT fix (commit 2f08440) are filled.
# Resume-safe: run_vision_grid.sh skips any cell with metrics.json. Log: queue.log (same).
cd "$(dirname "$0")/../.."
until grep -q 'IMAGE GRID end' results/logs_vision/final12_tail.log 2>/dev/null; do sleep 300; done
echo "$(date +%H:%M) IMAGE GRID retry pass (post ViT fix)" >>results/logs_vision/final12_tail.log
SCENARIOS="cil_cifar100 cil_imagenet_r" METHODS="coda_prompt cl_lora lca" SEEDS="42 7 123" \
  BATCH_SIZE=16 GRAD_CKPT=true NUM_WORKERS=2 EPOCHS_CAP=100 bash scripts/run_vision_grid.sh
echo "$(date +%H:%M) IMAGE GRID retry end" >>results/logs_vision/final12_tail.log
