#!/bin/bash
# Regenerate the cited latent-replay 87.3 artifact: canonical k=8 / d=5, EARLY-STOPPED
# (STATE:928 "that was a prior-session early-stop run"), 3 seeds. Waits for the grid first.
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "grid_fill.sh" > /dev/null; do sleep 300; done
echo "LATENT: grid finished, starting canonical latent_replay re-runs"
for SD in 42 7 123; do
  RUN="dil_latent_replay_seed${SD}"
  if [ -f "results/${RUN}/metrics.json" ]; then echo "LATENT: SKIP $RUN"; continue; fi
  echo "LATENT: start $RUN (k8 d5 early-stop)"
  uv run python scripts/train.py method=latent_replay scenario=dil seed=$SD \
    method.epochs=100 method.split_layer_k=8 method.docs_per_task=5 \
    training.batch_size=2 training.num_workers=0 training.gradient_checkpointing=true \
    wandb.mode=offline > "results/logs_latent_rerun_${RUN}.log" 2>&1
  echo "LATENT: done $RUN rc=$?"
done
echo "LATENT: all done"
