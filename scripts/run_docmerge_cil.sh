#!/usr/bin/env bash
# DocMERGE on cil_cord — the decisive test of the count-aware (1/T) merge fix.
#
# cil_cord = 5 sessions of CORD with a GROWING head → DISJOINT logit rows per session.
# This is the ONLY setting where count-aware merging changes the math (proven a no-op on
# dil's dense shared rows). RCA also predicts merge could plausibly work here because
# disjoint rows don't cancel. We run memory (baseline, no merge) + the 4 fixed merge
# variants + one count_aware=False CONTROL on plain, so any change is attributable to the
# fix vs the scenario.
#
# Frozen-backbone → faster than dil's der_pp; we SKIP der_pp here (replay≈high on CORD is
# already known; the scientific question is merge-vs-memory on disjoint rows).
# Resume-safe (skips a variant whose metrics.json exists). VRAM-safe local recipe.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-3}"; BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"; FISHER_N="${FISHER_N:-32}"
ROOT="${ROOT:-results/docmerge_cil}"; SCEN=cil_cord; NT=5
PY() { uv run python "$@"; }

common=(scenario="$SCEN" seed="$SEED" method.epochs="$EPOCHS" method.n_tasks="$NT"
        training.batch_size="$BATCH_SIZE" training.gradient_checkpointing=true
        training.num_workers="$NUM_WORKERS" wandb.mode=offline)

run() {  # $1 tag (unique out subdir), rest: overrides
  local tag="$1"; shift
  local odir="$ROOT/$tag"; local rundir="$odir/${SCEN}_doc_merge_seed${SEED}"
  if [[ -f "$rundir/metrics.json" ]]; then echo "SKIP $tag (done)"; return 0; fi
  echo "=== RUN $tag :: $* ==="
  uv run python scripts/train.py method=doc_merge "${common[@]}" \
    method.fisher_n_samples="$FISHER_N" output_dir="$odir" "$@"
}

echo "############ DocMERGE cil_cord (5 sessions, disjoint rows, seed $SEED, ${EPOCHS}ep) ############"
run memory               method.consolidate=memory method.router=sparse
run merge_plain          method.consolidate=merge  method.merge_rule=plain
run merge_plain_classic  method.consolidate=merge  method.merge_rule=plain method.merge_count_aware=false
run merge_ties           method.consolidate=merge  method.merge_rule=ties
run merge_fisher         method.consolidate=merge  method.merge_rule=fisher
run both_plain           method.consolidate=both   method.merge_rule=plain method.router=sparse

echo "############ summary (AA / BWT / routing) ############"
PY - "$ROOT" "$SCEN" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1:4]
print(f"{'variant':<20}{'AA':>8}{'BWT':>8}{'route':>8}")
for rundir in sorted(glob.glob(f"{root}/*/{scen}_doc_merge_seed{seed}")):
    tag = rundir.split("/")[-2]
    mp = os.path.join(rundir, "metrics.json")
    aa = bwt = rt = None
    if os.path.exists(mp):
        d = json.load(open(mp)); aa, bwt = d.get("AA"), d.get("BWT")
    rp = os.path.join(rundir, "routing.json")
    if os.path.exists(rp): rt = json.load(open(rp)).get("overall_hit_rate")
    f = lambda x, p=2: (f"{x:.{p}f}" if isinstance(x, (int, float)) else "-")
    print(f"{tag:<20}{f(aa):>8}{f(bwt):>8}{f(rt,3):>8}")
print("\nCompare: merge_plain (fixed) vs merge_plain_classic (pre-fix) = the 1/T fix effect.")
print("memory = no-merge baseline. If a fixed merge variant > memory AND BWT improves,")
print("the fix rescues merge on disjoint rows. If merge_plain == merge_plain_classic, the")
print("fix is a no-op here too (deltas dense) and merge is dead for doc-IE.")
PYEOF
echo "DONE."
