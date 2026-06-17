#!/usr/bin/env bash
# Study 1 — Generalization (vary backbone). Show the finding + DocCL's ranking
# transfer across architecturally diverse encoders, without re-running everything.
#
# Default subset: 4 most-informative methods (naive=lower, joint=upper,
#   der_pp=best-baseline placeholder — swap to the anchor's actual best baseline,
#   doccl=ours) × 2 vision-free backbones (LiLT, BROS) × {cil_cord, dil} × 3 seeds,
#   plus a LiLT-only dil_xlingual pass (cross-lingual is meaningless on English BROS).
# ≈ 4×2×2×3 + 4×1×3 ≈ 60 runs.
#
# Usage:
#   bash scripts/run_generalization.sh

set -euo pipefail

BACKBONES="${BACKBONES:-lilt_base bros_base}"
METHODS="${METHODS:-naive joint der_pp doccl}"
SCENARIOS="${SCENARIOS:-cil_cord dil dil_xlingual}"
SEEDS="${SEEDS:-42 123 7}"
# Backbones able to run the cross-lingual XFUND scenario (multilingual vocab).
MULTILINGUAL_BACKBONES="${MULTILINGUAL_BACKBONES:-lilt_base}"

echo "=== DocCL Generalization Study (Study 1) ==="
echo "Backbones: $BACKBONES"
echo "Methods:   $METHODS"
echo "Scenarios: $SCENARIOS"
echo "Seeds:     $SEEDS"
echo

START_TIME=$(date +%s)

for backbone in $BACKBONES; do
    for method in $METHODS; do
        for scenario in $SCENARIOS; do
            # Cross-lingual XFUND only runs on multilingual backbones.
            if [ "$scenario" = "dil_xlingual" ] && \
               [[ " $MULTILINGUAL_BACKBONES " != *" $backbone "* ]]; then
                continue
            fi
            for seed in $SEEDS; do
                run_name="${backbone}_${scenario}_${method}_seed${seed}"
                done_marker="results/${run_name}/.done"
                if [ -f "$done_marker" ]; then
                    echo "  [skip] $run_name"
                    continue
                fi

                echo "  [run] $run_name"
                if python scripts/train.py \
                    method="$method" \
                    scenario="$scenario" \
                    model="$backbone" \
                    seed="$seed"; then
                    mkdir -p "results/${run_name}"
                    touch "$done_marker"
                fi
            done
        done
    done
done

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "=== Generalization complete in $((ELAPSED/3600))h ==="
