#!/bin/bash
# CoLaR on the secondary backbones (2026-10-22, user: "utilize the GPU maximally"):
# closes the gap between tab:grid (four backbones) and tab:main (LayoutLMv3 only).
# Recipe = the DIL headline protocol of scripts/run_r2c_queue.sh (alias colar_e10, 10 epochs
# early-stopped, k=4, 50 docs/task, r=128), applied to BROS / BERT / LiLT.
# Order: DIL x3 backbones x3 seeds, then CIL-CORD, then Mixed (27 runs). Serial, own flock,
# resume-safe; runs BESIDE the image grid (CODA/CL-LoRA/LCA on ViT, ~0.85 GB VRAM, ~5.5 GB RAM),
# which is why num_workers=0 and batch 2 + grad-ckpt are kept. Log: results/logs_vision/colar_backbones.log
# ponytail: LiLT's separate layout stream is not banked (paper Limitations) — run last, disclosed.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/colar_backbones.log
exec 9>/tmp/claude-1000/colar_backbones.lock
flock -n 9 || { echo "colar_backbones already running" >&2; exit 1; }
mkdir -p results/logs_vision
# 2026-10-22 22:51: tried beside the image grid — CoLaR/BROS alone took 4.8 GB VRAM (total 5.6 GB,
# over the 5 GB cap) and was killed. A 6 GB card holds one of these at a time, so wait for the
# image grid AND its retry pass to finish (scripts/rca/final12_image_retry.sh starts on
# "IMAGE GRID end" and writes "IMAGE GRID retry end" when done), unless WAIT=0.
if [ "${WAIT:-1}" = 1 ]; then
  echo "$(date +%H:%M) waiting for IMAGE GRID retry end in results/logs_vision/final12_tail.log" >>"$LOG"
  until grep -q 'IMAGE GRID retry end' results/logs_vision/final12_tail.log 2>/dev/null; do sleep 300; done
fi
echo "=== COLAR BACKBONES $(date) ===" >>"$LOG"
COLAR_AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
COMMON="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"

for sc in ${SCENARIOS:-dil cil_cord mixed}; do
  for fam in ${FAMILIES:-bros bert lilt}; do
    for sd in ${SEEDS:-42 7 123}; do
      name="${sc}_colar_e10_seed${sd}_${fam}"
      if ls results/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
      # 2026-10-24 05:03: /mnt/DataDrive filled (LCA stashes) and killed three image cells. Do not start a
      # run with < 10 GB free; wait and re-check. (colar_e10 runs write no model dumps — the 7.8 GB of
      # final_model.pt/final_store.pt on disk are the Pass 2b probe runs.)
      while [ "$(df --output=avail -BG /mnt/DataDrive | tail -1 | tr -dc 0-9)" -lt 10 ]; do
        echo "$(date +%H:%M) DISK LOW ($(df -h /mnt/DataDrive | awk 'NR==2{print $4}') free) — waiting before $name" >>"$LOG"; sleep 600
      done
      echo "$(date +%H:%M) START $name" >>"$LOG"
      uv run python scripts/train.py method=colar +method.run_alias=colar_e10 scenario="$sc" model="${fam}_base" \
        seed="$sd" method.epochs=10 $COMMON $COLAR_AX >"results/logs_vision/$name.log" 2>&1
      if ls results/${name}*/metrics.json >/dev/null 2>&1; then
        echo "$(date +%H:%M) DONE  $name AA=$(python3 -c "import json,glob;print(round(json.load(open(glob.glob('results/${name}*/metrics.json')[0]))['AA'],2))")" >>"$LOG"
      else echo "$(date +%H:%M) FAIL  $name" >>"$LOG"; fi
    done
  done
done
echo "=== COLAR BACKBONES COMPLETE $(date) ===" >>"$LOG"
