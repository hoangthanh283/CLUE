#!/usr/bin/env bash
# LCA (Local Classifier Alignment, ICLR 2026) — effectiveness check on doc-IE.
#
# LCA trains the FULL backbone (SGD+cosine) per task, then TIES-merges backbones and
# re-aligns the classifier on sampled per-class Gaussian features. It is heavy (full
# finetune, like der_pp). We compare LCA against naive (forgetting floor) and der_pp
# (replay reference) on the chosen scenario, plus an LCA ablation with the align step OFF
# (ca_robust_weight=0 AND a degenerate align) is approximated by merge-only behaviour —
# but the clean ablation here is LCA-with-align vs LCA-no-align via ca_samples_per_cls.
#
# Knobs: SCENARIO (default dil), EPOCHS, CA_EPOCHS, BATCH_SIZE, SEED. Resume-safe.
set -euo pipefail
cd "$(dirname "$0")/.."

SCENARIO="${SCENARIO:-dil}"; SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"; CA_EPOCHS="${CA_EPOCHS:-10}"; BATCH_SIZE="${BATCH_SIZE:-2}"
ROOT="${ROOT:-results/lca_eval}"
PY() { uv run python "$@"; }

common=(scenario="$SCENARIO" seed="$SEED" training.batch_size="$BATCH_SIZE"
        training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline)

run() {  # $1 method, $2 tag, rest overrides
  local method="$1" tag="$2"; shift 2
  local odir="$ROOT/$tag" rundir
  rundir="$odir/${SCENARIO}_${method}_seed${SEED}"
  if [[ -f "$rundir/metrics.json" ]]; then echo "SKIP $tag (done)"; return 0; fi
  echo "=== RUN $tag :: method=$method $* ==="
  uv run python scripts/train.py method="$method" "${common[@]}" output_dir="$odir" "$@"
}

echo "############ LCA effectiveness ($SCENARIO, seed $SEED) ############"
# LCA full method (merge + align).
run lca   lca          method.epochs="$EPOCHS" method.ca_epochs="$CA_EPOCHS"
# LCA ablation: align OFF (merge only) — ca_samples_per_cls=0 path? Instead set ca_epochs=0
# so align is a no-op, isolating the merge contribution.
run lca   lca_nomerge_noalign  method.epochs="$EPOCHS" method.ca_epochs=0 method.merge_coef=0.0
# LCA with merge but no align (merge_coef=1, ca_epochs=0) — does align matter?
run lca   lca_merge_noalign    method.epochs="$EPOCHS" method.ca_epochs=0
# Baselines for context.
run naive naive        method.epochs="$EPOCHS"
run der_pp der_pp      method.epochs="$EPOCHS"

echo "############ summary ############"
PY - "$ROOT" "$SCENARIO" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1:4]
print(f"{'variant':<22}{'method':<10}{'AA':>8}{'BWT':>8}")
for rundir in sorted(glob.glob(f"{root}/*/{scen}_*_seed{seed}")):
    tag = rundir.split("/")[-2]; meth = rundir.split("/")[-1].split("_seed")[0].split(f"{scen}_")[-1]
    mp = os.path.join(rundir, "metrics.json")
    aa = bwt = None
    if os.path.exists(mp):
        d = json.load(open(mp)); aa, bwt = d.get("AA"), d.get("BWT")
    f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "-")
    print(f"{tag:<22}{meth:<10}{f(aa):>8}{f(bwt):>8}")
print("\nLCA (merge+align) vs lca_merge_noalign = does the alignment step help?")
print("vs naive (floor) and der_pp (replay reference). If LCA > naive and align > no-align,")
print("LCA's classifier alignment is effective on doc-IE.")
PYEOF
echo "DONE."
