#!/usr/bin/env bash
# Run the baseline + selected-method grid (and the component-targeting ablation).
#
# Phases (methods × scenarios × seeds):
#   PHASE=3 (core)     : naive joint ewc lwf er der_pp        (~54 runs)
#   PHASE=4 (advanced) : l2p dualprompt coda_prompt o_lora    (~36 runs)
#   PHASE=5 (selected) : doccl  (the pilot-selected proposed method)
#   PHASE=ablation     : proposed method × {text,visual,layout,fusion,uniform}
#                        on ABLATION_SCENARIOS (Table 6.2)
#   PHASE=all          : all baselines + doccl
#
# W&B defaults to OFFLINE (Vast.ai). Set WANDB_MODE=online for live logging.
#
# Usage:
#   PHASE=3 bash scripts/run_grid.sh                 # core baselines now (pre-pilot OK)
#   PHASE=5 bash scripts/run_grid.sh                 # proposed method (after GATE A)
#   PHASE=ablation bash scripts/run_grid.sh          # component ablation (after GATE A)
#   METHODS="naive joint" bash scripts/run_grid.sh   # custom subset

set -euo pipefail

PHASE="${PHASE:-all}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed}"
SEEDS="${SEEDS:-42 123 7}"
WANDB_MODE="${WANDB_MODE:-offline}"

PROPOSED="${PROPOSED:-doccl}"
COMPONENTS="${COMPONENTS:-text visual layout fusion uniform}"
ABLATION_SCENARIOS="${ABLATION_SCENARIOS:-cil_cord}"

# Extra Hydra overrides appended to every run. For a limited-VRAM GPU:
#   EXTRA="training.batch_size=2 training.gradient_checkpointing=true"
EXTRA="${EXTRA:-}"

START_TIME=$(date +%s)
TOTAL_RUNS=0
SKIPPED=0
FAILED=0

# run <run_name> <hydra-overrides...>
run_one() {
    local run_name="$1"; shift
    local done_marker="results/${run_name}/.done"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    if [ -f "$done_marker" ]; then
        echo "  [skip] $run_name (already done)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    echo "  [run]  $run_name"
    # Optional hard memory cap (MEM_CAP, e.g. "9G"): run train.py inside a transient
    # systemd cgroup scope so the kernel OOM-kills ONLY this process group if it ever
    # exceeds the cap — the machine itself can never be driven OOM. Falls back to a
    # plain run if systemd-run is unavailable.
    local launcher=(python scripts/train.py)
    if [ -n "${MEM_CAP:-}" ] && command -v systemd-run >/dev/null 2>&1; then
        launcher=(systemd-run --user --scope -q
                  -p "MemoryMax=${MEM_CAP}" -p "MemorySwapMax=0"
                  python scripts/train.py)
    fi
    # shellcheck disable=SC2086
    if "${launcher[@]}" "$@" "wandb.mode=$WANDB_MODE" $EXTRA; then
        mkdir -p "results/${run_name}"
        touch "$done_marker"
    else
        echo "  [FAIL] $run_name"
        FAILED=$((FAILED + 1))
    fi
}

if [ "$PHASE" = "ablation" ]; then
    echo "=== DocCL Component-Targeting Ablation (Table 6.2) ==="
    echo "Proposed:   $PROPOSED   Components: $COMPONENTS"
    echo "Scenarios:  $ABLATION_SCENARIOS   Seeds: $SEEDS   W&B: $WANDB_MODE"
    echo
    for scenario in $ABLATION_SCENARIOS; do
        for seed in $SEEDS; do
            for comp in $COMPONENTS; do
                run_one "${scenario}_${PROPOSED}_seed${seed}_${comp}" \
                    method="$PROPOSED" scenario="$scenario" seed="$seed" \
                    "method.target_component=$comp"
            done
        done
    done
else
    case "$PHASE" in
        3|core)     DEFAULT_METHODS="naive joint ewc lwf er der_pp" ;;
        4|advanced) DEFAULT_METHODS="l2p dualprompt coda_prompt o_lora" ;;
        5|selected) DEFAULT_METHODS="doccl" ;;
        all|*)      DEFAULT_METHODS="naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora doccl" ;;
    esac
    METHODS="${METHODS:-$DEFAULT_METHODS}"

    echo "=== DocCL Grid Run ==="
    echo "Phase:      $PHASE    W&B: $WANDB_MODE"
    echo "Methods:    $METHODS"
    echo "Scenarios:  $SCENARIOS    Seeds: $SEEDS"
    echo
    for method in $METHODS; do
        for scenario in $SCENARIOS; do
            for seed in $SEEDS; do
                run_one "${scenario}_${method}_seed${seed}" \
                    method="$method" scenario="$scenario" seed="$seed"
            done
        done
    done
fi

ELAPSED=$(( $(date +%s) - START_TIME ))
echo
echo "=== Grid complete ==="
echo "Total runs: $TOTAL_RUNS   Skipped: $SKIPPED   Failed: $FAILED"
echo "Elapsed:    $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
echo
echo "Next: python scripts/analyze_results.py && python scripts/ingest_to_thesis.py"
