#!/bin/bash
# Runs the H1 / H3 / calibration cells after the main vision grid exits (prereg Amendment 2).
# Waits on the grid's cmdline token; this script's own cmdline never contains it (no self-match).
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
LOG=results/logs_vision/queue_after.log; mkdir -p results/logs_vision results/pilot
echo "=== AFTER-GRID QUEUE armed $(date) ===" >>"$LOG"
while pgrep -f "run_vision_grid.sh" >/dev/null; do sleep 300; done
echo "=== main grid finished; starting $(date) ===" >>"$LOG"

run() {  # run <name> <cmd...>   resume-safe on results/<name>/metrics.json
  local name="$1"; shift
  [ -f "results/${name}/metrics.json" ] && { echo "$(date +%H:%M) SKIP $name" >>"$LOG"; return; }
  echo "$(date +%H:%M) START $name" >>"$LOG"
  "$@" > "results/logs_vision/${name}.log" 2>&1
  if [ $? -eq 0 ] && [ -f "results/${name}/metrics.json" ]; then touch "results/${name}/.done"; echo "$(date +%H:%M) DONE  $name" >>"$LOG"
  else echo "$(date +%H:%M) FAIL  $name" >>"$LOG"; fi
}
COMMON="model=vit_b16 scenario=cil_cifar100 training=vision training.batch_size=16 training.num_workers=2 training.gradient_checkpointing=true wandb.mode=offline"

# H2b calibration: published SLCA schedule (fixed 20 ep, cosine, no early stop).
for sd in 42 7 123; do
  run "cil_cifar100_slca_noca_pub_seed${sd}_vit" uv run python scripts/train.py method=slca_noca_pub seed=$sd $COMMON
done
# H3: real CLS bank (500 images/task at layer 8) vs class-Gaussian summary (500 label carriers).
for sd in 42 7 123; do
  run "cil_cifar100_latent_replay_seed${sd}_vit_d500" uv run python scripts/train.py method=latent_replay seed=$sd \
      method.split_layer_k=8 method.docs_per_task=500 method.epochs=100 $COMMON
  run "cil_cifar100_aglr_replay_seed${sd}_vit" uv run python scripts/train.py method=aglr_replay seed=$sd \
      method.split_layer_k=8 method.carriers_per_task=500 method.attn_keep=1.0 method.epochs=100 $COMMON
done
# H1: pilot diagnostics (Fisher-weighted displacement by depth + CKA per layer), naive, 5 ep/task.
for cond in cv_vit_fast cv_vit_slow; do
  for sd in 42 7 123; do
    name="pilot_${cond}_seed${sd}"
    if [ -f "results/pilot/${cond}_seed${sd}.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
    echo "$(date +%H:%M) START $name" >>"$LOG"
    uv run doccl-pilot --conditions $cond --seeds $sd --epochs 5 --batch_size 8 --output_dir results/pilot \
      > "results/logs_vision/${name}.log" 2>&1 && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
  done
done
echo "=== AFTER-GRID QUEUE COMPLETE $(date) ===" >>"$LOG"
