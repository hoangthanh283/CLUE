#!/usr/bin/env bash
# Read-side memory Gate 0->1 chain (plan: read-side memory for CoLaR).
# Waits for the running conservation seed sweep (never overlap a dataset builder), runs
# the no-training kNN separability probe (Gate 0), then — only if it passes — the
# colar_knn seed-42 Gate 1 runs, then the colar_meta m1/m3 control pair (independent
# track). Resume-safe: skips any run whose metrics.json exists.
# Launch detached:  nohup setsid bash scripts/run_readside_gate01.sh > results/readside_gate01.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "run_seeds_conservation.sh" > /dev/null; do sleep 300; done
while pgrep -f "scripts/train.py" > /dev/null; do sleep 300; done

# ── Gate 0: layer-k kNN separability probe (no training, ~30 min) ────────────────────
if [ ! -f results/gate0_knn_probe.json ]; then
  uv run python scripts/gate0_knn_probe.py > results/gate0_knn_probe.log 2>&1 \
    || { echo "GATE 0 PROBE CRASHED (see results/gate0_knn_probe.log)"; }
fi

run() {  # $1=method $2=extra-overrides $3=outdir-name
  local out="results/$3"
  [ -f "$out/metrics.json" ] && { echo "SKIP $out"; return; }
  rm -rf "$out"
  uv run python scripts/train.py method="$1" scenario=dil model=layoutlmv3_base seed=42 \
    $2 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 \
    wandb.mode=offline hydra.run.dir="$out" > "$out.log" 2>&1
  echo "DONE $out exit=$?"
}

GO=$(uv run python - <<'PY'
import json
try:
    best = json.load(open("results/gate0_knn_probe.json"))["best_per_task"]
    print(int(sum(v["f1"] >= 20 for v in best.values()) >= 2))
except Exception:
    print(0)
PY
)

# ── Gate 1: colar_knn at the CoLaR headline recipe (baseline colar r128 = AA 87.6) ───
if [ "$GO" = "1" ]; then
  run colar_knn "method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.knn_lambda=0.3" \
    dil_colar_knn_seed42_k4_d50_r128
  run colar_knn "method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.knn_lambda=1.0" \
    dil_colar_knn_seed42_k4_d50_r128_lam100
else
  echo "GATE 0 FAILED — colar_knn runs skipped (results/gate0_knn_probe.json)"
fi

# ── M1 control pair (k=8 default for VRAM headroom; m1 must reproduce plain CoLaR) ───
run colar_meta "method.docs_per_task=50 method.rank_r=128 method.meta_m=1" \
  dil_colar_meta_seed42_d50_r128_m1
run colar_meta "method.docs_per_task=50 method.rank_r=128 method.meta_m=3" \
  dil_colar_meta_seed42_d50_r128_m3

echo "READSIDE GATE CHAIN COMPLETE"
