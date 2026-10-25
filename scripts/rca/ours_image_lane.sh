#!/bin/bash
# Proposed-method gap (2026-10-25): CoLaR r128 (500/class, K=4) on CIFAR-100 x 3 seeds — the
# ImageNet-R cells exist (results/cil_imagenet_r_colar500r128_seed*_vit_k4_d500_r128), CIFAR
# never ran. Same recipe as those cells (run_colar_vision.sh COMMON) except num_workers=1 so it
# can share the box with the running CL-LoRA image cell (2.5 GB + 0.8 GB VRAM). Resume-safe.
set -u; cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
LOG=results/logs_vision/colar_queue.log; mkdir -p results/logs_vision
echo "=== OURS IMAGE LANE (colar500r128 cifar100) $(date) ===" >>"$LOG"
for sd in 42 7 123; do
  run="cil_cifar100_colar500r128_seed${sd}_vit"
  if ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $run" >>"$LOG"; continue; fi
  until [ "$(df -m /mnt/DataDrive | awk 'NR==2{print $4}')" -gt 10000 ]; do echo "$(date +%H:%M) waiting for disk" >>"$LOG"; sleep 600; done
  echo "$(date +%H:%M) START $run" >>"$LOG"
  uv run python scripts/train.py scenario=cil_cifar100 seed=$sd method=colar_vit method.rank_r=128 \
    +method.run_alias=colar500r128 method.epochs=100 model=vit_b16 training=vision \
    training.batch_size=16 training.num_workers=1 training.gradient_checkpointing=false \
    training.amp=true method.split_layer_k=4 >"results/logs_vision/${run}.log" 2>&1
  if ls results/${run}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) DONE  $run" >>"$LOG"; else echo "$(date +%H:%M) FAIL  $run" >>"$LOG"; fi
done
echo "=== OURS IMAGE LANE COMPLETE $(date) ===" >>"$LOG"
