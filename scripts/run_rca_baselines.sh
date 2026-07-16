#!/usr/bin/env bash
# RCA Tier B chain: instrumented core-6 baseline re-runs (scripts/rca_baselines.py).
# Waits for every earlier GPU chain (seed sweep, read-side gate) and any live train.py —
# one dataset builder at a time. Resume-safe: skips a method whose output JSON exists.
# Launch detached:  nohup setsid bash scripts/run_rca_baselines.sh > results/rca/chain.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "run_seeds_conservation.sh" > /dev/null; do sleep 300; done
while pgrep -f "run_readside_gate01.sh" > /dev/null; do sleep 300; done
while pgrep -f "scripts/train.py" > /dev/null; do sleep 300; done
while pgrep -f "scripts/gate0_knn_probe.py" > /dev/null; do sleep 300; done

mkdir -p results/rca
for m in naive ewc lwf er der_pp colar; do
  out="results/rca/dil_${m}_seed42_rca.json"
  [ -f "$out" ] && { echo "SKIP $out"; continue; }
  uv run python scripts/rca_baselines.py --method "$m" --seed 42 \
    --batch-size 2 --gradient-checkpointing --num-workers 0 \
    --output "$out" > "results/rca/dil_${m}_seed42_rca.log" 2>&1
  echo "DONE $m exit=$?"
done
echo "RCA BASELINES CHAIN COMPLETE"
