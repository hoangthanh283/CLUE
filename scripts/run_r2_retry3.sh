#!/bin/bash
# R2 retry 3: re-run cil_cord IS3 seed 7 after the token_features no_grad fix (AGENT FIX
# 2026-10-16: LayoutLMv3Wrapper.token_features was @torch.no_grad(), so IS3 trained only its
# head; the 00:12 seed-7 run was killed). Waits for retry2 (last job in the chain). Resume-safe.
# Log: results/logs_vision/r2_retry3.log
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/r2_retry3.log
mkdir -p results/logs_vision
until grep -q 'R2 RETRY2 COMPLETE' results/logs_vision/r2_retry2.log 2>/dev/null; do sleep 300; done
echo "=== R2 RETRY3 $(date) ===" >>"$LOG"
COMMON="training.batch_size=1 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline"
for name in cil_cord_is3_seed7; do
  if [ -f "results/$name/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; continue; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py method=is3 scenario=cil_cord model=layoutlmv3_base seed=7 $COMMON \
    >"results/logs_vision/$name.log" 2>&1
  [ -f "results/$name/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
done
echo "=== R2 RETRY3 COMPLETE $(date) ===" >>"$LOG"
