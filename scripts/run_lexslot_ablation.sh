#!/usr/bin/env bash
# LexSlot ablation on dil — the go/no-go. Two diagnosis-driven axes:
#   slot_sharing soft-vs-off : does lexical slot-SHARING give positive transfer (S=0.31 pair)?
#   slot_depth head_late-vs-uniform : does TARGETING the locus matter (like DocCL 85-vs-42)?
# vs DocCL (the working method LexSlot must beat), ER (buffer ref), naive (floor).
# Resume-safe; local-2060 VRAM-safe.  Knobs: SEED, EPOCHS, BATCH_SIZE.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-5}"; BATCH_SIZE="${BATCH_SIZE:-2}"
ROOT="${ROOT:-results/lexslot_ablation}"; SCEN=dil
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

echo "############ LexSlot ablation ($SCEN, seed $SEED, ${EPOCHS}ep) ############"
run naive  naive
run er     er
run doccl  doccl            method.target_depth=all
run lexslot ls_soft_headlate method.slot_depth=head_late method.slot_sharing=soft
run lexslot ls_off_headlate  method.slot_depth=head_late method.slot_sharing=off
run lexslot ls_soft_uniform  method.slot_depth=uniform   method.slot_sharing=soft
run lexslot ls_soft_headonly method.slot_depth=head_only method.slot_sharing=soft

echo "############ summary (AA / BWT) ############"
PY - "$ROOT" "$SCEN" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1:4]
print(f"{'variant':<18}{'method':<9}{'AA':>8}{'BWT':>8}")
for rundir in sorted(glob.glob(f"{root}/*/{scen}_*_seed{seed}")):
    tag = rundir.split("/")[-2]; meth = rundir.split("/")[-1].split("_seed")[0].replace(f"{scen}_", "")
    mp = os.path.join(rundir, "metrics.json"); aa = bwt = None
    if os.path.exists(mp):
        d = json.load(open(mp)); aa, bwt = d.get("AA"), d.get("BWT")
    f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "-")
    print(f"{tag:<18}{meth:<9}{f(aa):>8}{f(bwt):>8}")
print("\nGO/NO-GO: (sharing) ls_soft_headlate BWT/AA > ls_off_headlate -> lexical sharing transfers;")
print("(targeting) ls_soft_headlate > ls_soft_uniform -> targeting matters for slots (cf DocCL 85 vs 42);")
print("does any lexslot beat DocCL? per-class-F1 of SHARED labels (total/tax) tells the transfer story.")
PYEOF
echo "DONE."
