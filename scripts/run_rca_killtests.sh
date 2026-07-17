#!/usr/bin/env bash
# RCA kill-tests chain (docs/RCA_KILLTESTS_PREREG_2026-07.md). Queues behind every live GPU
# chain (read-side gate, train.py, other rca runners), resume-safe per output artifact.
# Launch detached:  nohup setsid bash scripts/run_rca_killtests.sh > results/rca/killtests/chain.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/rca/killtests

while pgrep -f "run_readside_gate01.sh" > /dev/null; do sleep 300; done
while pgrep -f "gate0_knn_probe.py" > /dev/null; do sleep 300; done
while pgrep -f "scripts/train.py" > /dev/null; do sleep 300; done
while pgrep -f "scripts/rca_baselines.py --method" > /dev/null; do sleep 300; done

run_rca() {  # $1=method $2=extra-args $3=output
  [ -f "$3" ] && { echo "SKIP $3"; return; }
  uv run python scripts/rca_baselines.py --method "$1" --seed 42 \
    --batch-size 2 --gradient-checkpointing --num-workers 0 $2 --output "$3"
  echo "DONE $3 exit=$?"
}

# 1. Readout suite: naive retrain + logit dump (kill-tests #1/#2, train stage)
if [ ! -f results/rca/killtests/train_meta.json ]; then
  uv run python scripts/rca_readout_fixes.py --stage train
  echo "DONE readout train exit=$?"
else
  echo "SKIP readout train"
fi

# 2. Frozen-trunk naive (kill-test #3)
run_rca naive "--freeze-trunk" results/rca/dil_naive_frozen_seed42_rca.json

# 3. Marginal-KL anchor (kill-test #4)
run_rca marginal_kl "" results/rca/dil_marginal_kl_seed42_rca.json

# 4. Balanced-softmax logit adjustment (kill-test #5)
run_rca logit_adjust "" results/rca/dil_logit_adjust_seed42_rca.json

# 5. Corrections (CPU — smoke gate inside hard-stops on retrain drift)
if [ ! -f results/rca/killtests/readout_fixes.json ]; then
  uv run python scripts/rca_readout_fixes.py --stage correct
  echo "DONE readout correct exit=$?"
else
  echo "SKIP readout correct"
fi

echo "KILLTESTS CHAIN COMPLETE"
