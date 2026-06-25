#!/usr/bin/env bash
# DocMERGE feasibility on `dil` (FUNSD->SROIE->CORD) — the local-2060, VRAM-safe cut.
#
#   Step A: re-confirm the head-locus on THIS method (each run writes diag.json).
#   Step B: the consolidate x merge_rule bake-off vs der_pp.
#
# Go/no-go (from the plan): `both` AA >= memory-only AND >= merge-only, AND BWT > 0;
# a fisher>plain gap is bonus evidence the head-locus signal is useful.
#
# train.py derives the run dir as <output_dir>/<scenario>_<method>_seed<N>, so the
# doc_merge ablation variants (same scenario/method/seed) would COLLIDE in one dir.
# We give each variant its own output_dir subfolder so dirs and diag/routing.json
# stay separate. Resume-safe: skips a variant whose metrics.json already exists.
# Knobs (env): SEED, EPOCHS, BATCH_SIZE, GRAD_CKPT, NUM_WORKERS, SCENARIO, FISHER_N.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_CKPT="${GRAD_CKPT:-true}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SCENARIO="${SCENARIO:-dil}"
FISHER_N="${FISHER_N:-64}"
ROOT="${ROOT:-results/docmerge_feas}"
PY() { uv run python "$@"; }

common=(scenario="$SCENARIO" seed="$SEED"
        training.epochs="$EPOCHS" training.batch_size="$BATCH_SIZE"
        training.gradient_checkpointing="$GRAD_CKPT" training.num_workers="$NUM_WORKERS"
        wandb.mode=offline)

# $1 method, $2 variant tag (unique output_dir subfolder), rest: extra overrides
run() {
  local method="$1"; shift
  local tag="$1"; shift
  local odir="$ROOT/$tag"
  local rundir="$odir/${SCENARIO}_${method}_seed${SEED}"
  if [[ -f "$rundir/metrics.json" ]]; then
    echo "SKIP $tag ($rundir/metrics.json exists)"; return 0
  fi
  echo "=== RUN $tag :: method=$method $* ==="
  uv run python scripts/train.py method="$method" "${common[@]}" \
    output_dir="$odir" "$@"
}

echo "############ DocMERGE feasibility ($SCENARIO, seed $SEED, ${EPOCHS}ep) ############"

# Replay reference (the saturated upper line on this scenario).
run der_pp der_pp

# The ablation axis: memory-only, merge-only x{plain,ties,fisher}, both x{...}.
run doc_merge memory   method.consolidate=memory method.router=sparse method.fisher_n_samples="$FISHER_N"
for rule in plain ties fisher; do
  run doc_merge "merge_${rule}" method.consolidate=merge method.merge_rule="$rule" method.fisher_n_samples="$FISHER_N"
  run doc_merge "both_${rule}"  method.consolidate=both  method.merge_rule="$rule" method.router=sparse method.fisher_n_samples="$FISHER_N"
done

echo "############ summary (AA / BWT / routing / head-locus) ############"
PY - "$ROOT" "$SCENARIO" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for rundir in sorted(glob.glob(f"{root}/*/{scen}_*_seed{seed}")):
    tag = rundir.split("/")[-2]
    m = mm = None
    mp = os.path.join(rundir, "metrics.json")
    if os.path.exists(mp):
        d = json.load(open(mp)); mm = d.get("metrics", d)
    rp = os.path.join(rundir, "routing.json")
    routing = json.load(open(rp))["overall_hit_rate"] if os.path.exists(rp) else None
    dp = os.path.join(rundir, "diag.json")
    hb = None
    if os.path.exists(dp):
        d = json.load(open(dp)); pb = d.get("per_boundary", {})
        hb = {k: round(v["head_over_backbone"], 1) for k, v in pb.items()}
    aa = mm.get("AA") if mm else None
    bwt = mm.get("BWT") if mm else None
    rows.append((tag, aa, bwt, routing, hb))
print(f"{'variant':<14}{'AA':>8}{'BWT':>8}{'route':>8}   head/backbone")
for tag, aa, bwt, routing, hb in rows:
    aas = f"{aa:.2f}" if isinstance(aa, (int, float)) else "-"
    bws = f"{bwt:.2f}" if isinstance(bwt, (int, float)) else "-"
    rts = f"{routing:.3f}" if isinstance(routing, (int, float)) else "-"
    print(f"{tag:<14}{aas:>8}{bws:>8}{rts:>8}   {hb if hb else ''}")
PYEOF
echo "DONE."
