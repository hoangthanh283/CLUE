#!/usr/bin/env bash
# Run the full pilot study: 4 conditions × 3 seeds = 12 sequential runs.
# Each run trains on FUNSD → CORD → SROIE (3 tasks, naive sequential) and saves a
# JSON to results/pilot/ (the pilot runner does NOT use W&B — no offline concern).
#
# Estimated total: ~18 GPU-hours on RTX 4090.
#
# Usage:
#   bash scripts/run_pilot.sh                          # all 4 conditions, 3 seeds
#   CONDITIONS="c4_full" bash scripts/run_pilot.sh     # subset
#   SEEDS="42" bash scripts/run_pilot.sh               # single seed for debugging
#   ORDER="2 1 0" bash scripts/run_pilot.sh            # alternate task order (§6.1.3
#                                                      # stability check; pooled by analyze)

set -euo pipefail

CONDITIONS="${CONDITIONS:-c1_bert c2_no_text c3_no_image c4_full}"
SEEDS="${SEEDS:-42 123 7}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-results/pilot}"
ORDER="${ORDER:-}"   # empty = default FUNSD→CORD→SROIE; e.g. "2 1 0" = reversed
# Limited-VRAM: GRAD_CKPT=1, lower BATCH_SIZE and probe sizes.
GRAD_CKPT="${GRAD_CKPT:-}"
CKA_N="${CKA_N:-500}"
FISHER_N="${FISHER_N:-200}"

MEM_ARGS="--cka_n_samples $CKA_N --fisher_n_samples $FISHER_N"
[ -n "$GRAD_CKPT" ] && MEM_ARGS="$MEM_ARGS --gradient_checkpointing"

mkdir -p "$OUTPUT_DIR"

# Match the result-file suffix produced by run_pilot.py for non-default orders.
ORDER_ARGS=""
ORDER_SUFFIX=""
if [ -n "$ORDER" ] && [ "$ORDER" != "0 1 2" ]; then
    ORDER_ARGS="--task_order $ORDER"
    ORDER_SUFFIX="_ord$(echo "$ORDER" | tr -d ' ')"
fi

echo "=== DocCL Pilot Study ==="
echo "Conditions: $CONDITIONS"
echo "Seeds:      $SEEDS"
echo "Order:      ${ORDER:-default (0 1 2)}"
echo "Output:     $OUTPUT_DIR"
echo

START_TIME=$(date +%s)

for cond in $CONDITIONS; do
    for seed in $SEEDS; do
        echo
        echo "--- Running condition=$cond seed=$seed order=${ORDER:-default} ---"
        result_file="$OUTPUT_DIR/${cond}_seed${seed}${ORDER_SUFFIX}.json"
        if [ -f "$result_file" ]; then
            echo "  Result file exists at $result_file — skipping (delete to re-run)"
            continue
        fi

        # shellcheck disable=SC2086
        python -m doccl.pilot.run_pilot \
            --conditions "$cond" \
            --seeds "$seed" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$OUTPUT_DIR" \
            $MEM_ARGS $ORDER_ARGS

        echo "  Done: $cond seed=$seed → $result_file"
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "=== All pilot runs complete in $((ELAPSED/3600))h $((ELAPSED%3600/60))m ==="
echo
echo "Next: python -m doccl.pilot.analyze --pilot_dir $OUTPUT_DIR"
