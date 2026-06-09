#!/usr/bin/env bash
# Generalization study: run subset of methods on LiLT and BROS backbones.
# Designed to overlap with paper-writing time in W12-13.
#
# Subset: 6 methods (naive, joint, best-replay=DER++, best-prompt=CODA-P,
#                    best-LoRA=O-LoRA, our-DocCL) × 3 scenarios × 3 seeds × 2 backbones
# = 108 runs, ~130 GPU-hours
#
# Usage:
#   bash scripts/run_generalization.sh

set -euo pipefail

BACKBONES="${BACKBONES:-lilt_base bros_base}"
METHODS="${METHODS:-naive joint der_pp coda_prompt o_lora doccl}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed}"
SEEDS="${SEEDS:-42 123 7}"

echo "=== DocCL Generalization Study ==="
echo "Backbones: $BACKBONES"
echo "Methods:   $METHODS"
echo "Scenarios: $SCENARIOS"
echo "Seeds:     $SEEDS"
echo
echo "WARNING: LiLT and BROS wrappers are stubs at the time of this script."
echo "Implement doccl/models/lilt_wrapper.py and bros_wrapper.py before running."
echo

START_TIME=$(date +%s)

for backbone in $BACKBONES; do
    for method in $METHODS; do
        for scenario in $SCENARIOS; do
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
