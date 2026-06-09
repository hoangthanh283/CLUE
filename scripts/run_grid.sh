#!/usr/bin/env bash
# Run the full baseline + selected-method grid.
# Methods × scenarios × seeds = N runs.
#
# Estimated total (excluding pilot, generalization):
#   Phase 3 (W5-6): 6 methods × 3 scenarios × 3 seeds = 54 runs (~54 GPU-hours)
#   Phase 4 (W7-8): 4 methods × 3 scenarios × 3 seeds = 36 runs (~43 GPU-hours)
#   Phase 5 (W9-11): 1 method × 3 scenarios × 3 seeds + ablations = ~57 runs (~74 GPU-hours)
#
# Usage:
#   bash scripts/run_grid.sh              # full grid (after pilot complete)
#   PHASE=3 bash scripts/run_grid.sh      # only core baselines
#   PHASE=4 bash scripts/run_grid.sh      # only advanced
#   METHODS="naive joint" bash scripts/run_grid.sh

set -euo pipefail

PHASE="${PHASE:-all}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed}"
SEEDS="${SEEDS:-42 123 7}"

case "$PHASE" in
    3|core)    DEFAULT_METHODS="naive joint ewc lwf er der_pp" ;;
    4|advanced) DEFAULT_METHODS="l2p dualprompt coda_prompt o_lora" ;;
    5|selected) DEFAULT_METHODS="doccl" ;;
    all|*)     DEFAULT_METHODS="naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora doccl" ;;
esac
METHODS="${METHODS:-$DEFAULT_METHODS}"

echo "=== DocCL Grid Run ==="
echo "Phase:     $PHASE"
echo "Methods:   $METHODS"
echo "Scenarios: $SCENARIOS"
echo "Seeds:     $SEEDS"
echo

START_TIME=$(date +%s)
TOTAL_RUNS=0
SKIPPED=0
FAILED=0

for method in $METHODS; do
    for scenario in $SCENARIOS; do
        for seed in $SEEDS; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            run_name="${scenario}_${method}_seed${seed}"
            done_marker="results/${run_name}/.done"

            if [ -f "$done_marker" ]; then
                echo "  [skip] $run_name (already done)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "  [run]  $run_name"
            if python scripts/train.py \
                method="$method" \
                scenario="$scenario" \
                seed="$seed"; then
                mkdir -p "results/${run_name}"
                touch "$done_marker"
            else
                echo "  [FAIL] $run_name — see W&B for details"
                FAILED=$((FAILED + 1))
                # don't abort, just log and continue
            fi
        done
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "=== Grid complete ==="
echo "Total runs:    $TOTAL_RUNS"
echo "Skipped:       $SKIPPED"
echo "Failed:        $FAILED"
echo "Elapsed:       $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
echo
echo "Next: python scripts/analyze_results.py --project doccl-aaai2027"
