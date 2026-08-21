#!/bin/bash
# Fill BROS/LiLT grid gaps. Resume-safe: skips any run whose metrics.json exists.
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS_CAP="${EPOCHS_CAP:-100}"  # MUST match run_grid_multigpu.sh; configs default to 10
JOBS=/tmp/claude-1000/-mnt-DataDrive-Workspace-Master-HUST-Thesis-CLUE/3706e3e1-eb49-4559-a434-4972c37e1c26/scratchpad/grid_jobs_local.txt
declare -A MODEL=( [bros]=bros_base [lilt]=lilt_base )
n=0; total=$(wc -l < "$JOBS")
while read -r BB SC ME SD; do
  n=$((n+1))
  RUN="${SC}_${ME}_seed${SD}_${BB}"
  if [ -f "results/${RUN}/metrics.json" ]; then echo "SKIP [$n/$total] $RUN"; continue; fi
  echo "GRID [$n/$total] start $RUN"
  # lwf/der_pp hold a frozen teacher (or logit buffer) alongside the student: bs=2 OOMs the
  # 6 GB box on the larger backbones, so they start at bs=1. Any other failure retries at bs=1.
  case "$ME" in lwf|der_pp) BS=1 ;; *) BS=2 ;; esac
  uv run python scripts/train.py method=$ME scenario=$SC model=${MODEL[$BB]} seed=$SD \
    method.epochs=${EPOCHS_CAP} training.batch_size=$BS training.num_workers=0 training.gradient_checkpointing=true \
    wandb.mode=offline > "results/logs_gridfill_${RUN}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && [ "$BS" != "1" ]; then
    echo "GRID [$n/$total] retry $RUN at bs=1"
    uv run python scripts/train.py method=$ME scenario=$SC model=${MODEL[$BB]} seed=$SD \
      method.epochs=${EPOCHS_CAP} training.batch_size=1 training.num_workers=0 training.gradient_checkpointing=true \
      wandb.mode=offline > "results/logs_gridfill_${RUN}.log" 2>&1
    rc=$?
  fi
  if [ $rc -ne 0 ]; then echo "GRID [$n/$total] FAIL $RUN rc=$rc (continuing)"; else echo "GRID [$n/$total] done $RUN"; fi
done < "$JOBS"
echo "GRID: all done"
