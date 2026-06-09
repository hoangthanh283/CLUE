#!/usr/bin/env bash
# Run the full pilot study: 4 conditions × 3 seeds = 12 sequential runs.
# Each run trains on FUNSD → CORD → SROIE (3 tasks, naive sequential).
#
# Estimated total: ~18 GPU-hours on RTX 4090.
# Logs to W&B project: doccl-aaai2027 with tag pilot/{condition}/seed{seed}.
#
# Usage:
#   bash scripts/run_pilot.sh                  # all 4 conditions, 3 seeds
#   CONDITIONS="c4_full" bash scripts/run_pilot.sh   # subset
#   SEEDS="42" bash scripts/run_pilot.sh             # single seed for debugging

set -euo pipefail

CONDITIONS="${CONDITIONS:-c1_bert c2_no_text c3_no_image c4_full}"
SEEDS="${SEEDS:-42 123 7}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-results/pilot}"

mkdir -p "$OUTPUT_DIR"

echo "=== DocCL Pilot Study ==="
echo "Conditions: $CONDITIONS"
echo "Seeds:      $SEEDS"
echo "Output:     $OUTPUT_DIR"
echo

START_TIME=$(date +%s)

for cond in $CONDITIONS; do
    for seed in $SEEDS; do
        echo
        echo "--- Running condition=$cond seed=$seed ---"
        # Skip if result file exists (allows resuming after interruption)
        result_file="$OUTPUT_DIR/${cond}_seed${seed}.json"
        if [ -f "$result_file" ]; then
            echo "  Result file exists at $result_file — skipping (delete to re-run)"
            continue
        fi

        python -m doccl.pilot.run_pilot \
            --conditions "$cond" \
            --seeds "$seed" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$OUTPUT_DIR"

        echo "  Done: $cond seed=$seed → $result_file"
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "=== All pilot runs complete in $((ELAPSED/3600))h $((ELAPSED%3600/60))m ==="
echo
echo "Next: python -m doccl.pilot.analyze --pilot_dir $OUTPUT_DIR"
