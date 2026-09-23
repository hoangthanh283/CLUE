#!/bin/bash
# Fill remaining grid cells on the local RTX 2060 (6 GB).
# Resume-safe: skips runs with metrics.json.
# Priority: ER > DER++ > EWC > LWF (LWF will likely OOM but try anyway).
# CA-CoLaR T5 seeds queued after the grid.
set -u
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

EPOCHS_CAP=100
LOGDIR="results/logs_remaining"
mkdir -p "$LOGDIR"

declare -A MODEL=( [bros]=bros_base [lilt]=lilt_base )

run_one() {
  local bb="$1" sc="$2" me="$3" sd="$4"
  local run="${sc}_${me}_seed${sd}_${bb}"
  local log="$LOGDIR/${run}.log"

  if [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) SKIP $run (done)"
    return 0
  fi

  local bs=1 extra=""
  # ponytail: replay_batch_size=2 for ER/DER++ to fit 6GB; upgrade if renting GPU
  case "$me" in
    er)     extra="method.replay_batch_size=2" ;;
    der_pp) extra="method.replay_batch_size=2" ;;
  esac

  # ponytail: 4h timeout for cil_cord (5 tasks); 2.5h for others
  local tlimit=9000
  case "$sc" in cil_cord) tlimit=14400 ;; esac

  echo "$(date +%H:%M) START $run (bs=$bs)"
  timeout $tlimit uv run python scripts/train.py \
    method=$me scenario=$sc model=${MODEL[$bb]} seed=$sd \
    method.epochs=${EPOCHS_CAP} training.batch_size=$bs \
    training.num_workers=0 training.gradient_checkpointing=true \
    wandb.mode=offline $extra \
    > "$log" 2>&1
  local rc=$?

  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) DONE  $run"
  elif [ $rc -eq 137 ] || [ $rc -eq 134 ]; then
    echo "$(date +%H:%M) OOM   $run (rc=$rc, skipping method on this backbone)"
    return 1
  else
    echo "$(date +%H:%M) FAIL  $run (rc=$rc, see $log)"
    return 1
  fi
}

TOTAL=0
DONE=0
FAIL=0
OOM=0

echo "=== REMAINING GRID FILL — $(date) ==="

# Phase 1: ER (most likely to fit — tested at 5664 MiB)
echo "--- Phase 1: ER ---"
for bb in bros lilt; do
  for sc in dil cil_cord mixed; do
    for sd in 42 7 123; do
      TOTAL=$((TOTAL+1))
      if run_one "$bb" "$sc" er "$sd"; then
        DONE=$((DONE+1))
      else
        FAIL=$((FAIL+1))
      fi
    done
  done
done

# Phase 2: DER++ (2x replay samples, tighter)
echo "--- Phase 2: DER++ ---"
for bb in bros lilt; do
  for sc in dil cil_cord mixed; do
    for sd in 42 7 123; do
      TOTAL=$((TOTAL+1))
      if run_one "$bb" "$sc" der_pp "$sd"; then
        DONE=$((DONE+1))
      else
        FAIL=$((FAIL+1))
      fi
    done
  done
done

# Phase 3: LiLT cil_cord EWC (Fisher on growing head)
echo "--- Phase 3: EWC ---"
for sd in 42 7 123; do
  TOTAL=$((TOTAL+1))
  if run_one lilt cil_cord ewc "$sd"; then
    DONE=$((DONE+1))
  else
    FAIL=$((FAIL+1))
  fi
done

# Phase 4: LWF (teacher deepcopy — expect OOM but try)
echo "--- Phase 4: LWF (expect OOM) ---"
for bb in bros lilt; do
  for sc in dil cil_cord mixed; do
    for sd in 42 7 123; do
      TOTAL=$((TOTAL+1))
      if run_one "$bb" "$sc" lwf "$sd"; then
        DONE=$((DONE+1))
      else
        FAIL=$((FAIL+1))
      fi
    done
  done
done

# Phase 5: CA-CoLaR T5 — multi-seed robustness for A1 schedule [18,5,2]/[64,128,64]
echo "--- Phase 5: CA-CoLaR T5 seeds ---"
for sd in 7 123; do
  run="dil_colar_adaptive_seed${sd}"
  if [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) SKIP $run (done)"
    continue
  fi
  TOTAL=$((TOTAL+1))
  echo "$(date +%H:%M) START $run"
  timeout 7200 uv run python scripts/train.py \
    method=colar_adaptive scenario=dil model=layoutlmv3_base seed=$sd \
    method.epochs=5 method.split_layer_k=4 \
    'method.docs_schedule=[18,5,2]' 'method.rank_schedule=[64,128,64]' \
    training.batch_size=1 training.num_workers=0 \
    training.gradient_checkpointing=true \
    wandb.mode=offline \
    > "$LOGDIR/${run}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "results/${run}/metrics.json" ]; then
    echo "$(date +%H:%M) DONE  $run"
    DONE=$((DONE+1))
  else
    echo "$(date +%H:%M) FAIL  $run (rc=$rc)"
    FAIL=$((FAIL+1))
  fi
done

echo ""
echo "=== GRID FILL COMPLETE — $(date) ==="
echo "Total: $TOTAL | Done: $DONE | Failed: $FAIL"
echo "Check $LOGDIR/ for per-run logs."
