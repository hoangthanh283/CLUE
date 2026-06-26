#!/usr/bin/env bash
# HGT alpha-ablation on dil — the go/no-go: does head-gradient TRANSFER (alpha>0) give
# positive BWT vs head-gradient PROTECTION (alpha=0, BWT~0)? Plus CUBER (whole-network),
# ER (buffer reference), naive (floor). Resume-safe; local-2060 VRAM-safe.
# Knobs: SEED, EPOCHS, BATCH_SIZE, SUBSPACE_K, BACKBONE_TRAINABLE.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-5}"; BATCH_SIZE="${BATCH_SIZE:-2}"
SUBSPACE_K="${SUBSPACE_K:-32}"; BB="${BACKBONE_TRAINABLE:-false}"
ROOT="${ROOT:-results/hgt_ablation}"; SCEN=dil
PY() { uv run python "$@"; }

common=(scenario="$SCEN" seed="$SEED" method.epochs="$EPOCHS"
        training.batch_size="$BATCH_SIZE" training.gradient_checkpointing=true
        training.num_workers=0 wandb.mode=offline)

run() {  # $1 method, $2 tag, rest overrides
  local method="$1" tag="$2"; shift 2
  local odir="$ROOT/$tag" rundir
  rundir="$odir/${SCEN}_${method}_seed${SEED}"
  if [[ -f "$rundir/metrics.json" ]]; then echo "SKIP $tag (done)"; return 0; fi
  echo "=== RUN $tag :: method=$method $* ==="
  if ! uv run python scripts/train.py method="$method" "${common[@]}" output_dir="$odir" "$@"; then
    echo "!!! RUN $tag FAILED — continuing"
  fi
}

echo "############ HGT alpha-ablation ($SCEN, seed $SEED, ${EPOCHS}ep, bb_trainable=$BB) ############"
run naive  naive
run er     er
run hgt    hgt_a0   method.transfer_alpha=0.0 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run hgt    hgt_a05  method.transfer_alpha=0.5 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run hgt    hgt_a1   method.transfer_alpha=1.0 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run cuber  cuber    method.transfer_alpha=0.5 method.subspace_k="$SUBSPACE_K"

echo "############ summary (AA / BWT) ############"
PY - "$ROOT" "$SCEN" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1:4]
print(f"{'variant':<12}{'method':<8}{'AA':>8}{'BWT':>8}")
for rundir in sorted(glob.glob(f"{root}/*/{scen}_*_seed{seed}")):
    tag = rundir.split("/")[-2]; meth = rundir.split("/")[-1].split("_seed")[0].replace(f"{scen}_", "")
    mp = os.path.join(rundir, "metrics.json"); aa = bwt = None
    if os.path.exists(mp):
        d = json.load(open(mp)); aa, bwt = d.get("AA"), d.get("BWT")
    f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "-")
    print(f"{tag:<12}{meth:<8}{f(aa):>8}{f(bwt):>8}")
print("\nGO/NO-GO: hgt_a05/a1 BWT > hgt_a0 BWT (transfer beats protection) AND ideally BWT>0.")
print("Compare AA to er (buffer reference) and cuber. Flat alpha -> head-transfer falsified.")
PYEOF
echo "DONE."
