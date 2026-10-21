#!/bin/bash
# Tier B calibration pilot (plan rev. 8): once Tier A is complete, run ONE image cell —
# SLCA CIFAR-100 seed 42 — at batch 16 with gradient checkpointing (the only recipe the 6 GB
# box holds) and report the AA delta against the existing batch-128 cell
# (results/cil_cifar100_slca_seed42_vit, AA 87.4). It stops there: whether to run the 18
# divergent image cells locally or rent a GPU is a user decision (go if |delta| <~ 1 pp).
set -u
cd "$(dirname "$0")/../.."
LOG=results/logs_vision/final12_tierB_pilot.log
exec 9>/tmp/claude-1000/final12_tierB_pilot.lock
flock -n 9 || { echo "pilot already running" >&2; exit 1; }
until grep -q 'TIER A COMPLETE' results/logs_vision/final12_tierA.log 2>/dev/null; do sleep 300; done
echo "=== TIER B PILOT $(date) ===" >>"$LOG"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENARIOS="cil_cifar100" METHODS="slca" SEEDS="42" BATCH_SIZE=16 GRAD_CKPT=true NUM_WORKERS=2 \
  bash scripts/run_vision_grid.sh
python3 - >>"$LOG" <<'PY'
import json, pathlib
ref = pathlib.Path("results/cil_cifar100_slca_seed42_vit/metrics.json")
new = pathlib.Path("results/cil_cifar100_slca_bs16_seed42_vit/metrics.json")
if not new.exists():
    print("PILOT FAIL: no metrics.json for the bs16 cell"); raise SystemExit
a, b = json.load(open(ref))["AA"], json.load(open(new))["AA"]
print(f"PILOT RESULT  SLCA CIFAR-100 s42: bs128 AA={a:.2f}  bs16 AA={b:.2f}  delta(bs16-bs128)={b-a:+.2f} pp")
print("PILOT VERDICT:", "GO-local (|delta| <= 1 pp)" if abs(b-a) <= 1.0 else "STOP — consider renting (|delta| > 1 pp)")
PY
echo "=== TIER B PILOT COMPLETE $(date) ===" >>"$LOG"
