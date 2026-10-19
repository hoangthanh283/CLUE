"""Pass 2 (ledger-only part) + Pass 3 + Pass 5 analysis for CoLaR / LexSlot.

Reads rca/colar_lexslot_ledger.csv and emits:
- rca/pass2_gap_decomposition.md    — replay_gap, plastic_gap per cell
- rca/pass3_flaw_map.md             — plasticity x recovery 2x2 per cell
- rca/pass5_sign_rule.md            — per-arm per-scenario mean/sd/sign-agreement
- rca/pros_cons.md                  — distilled strengths/weaknesses

The ridge terms (head_gap, trunk_gap, coverage_gap) need the probe
(scripts/rca_colar_probe.py) and are left as TODO cells for Pass 2b.
"""
from __future__ import annotations
import csv, statistics as st, pathlib, collections, math

ROWS = list(csv.DictReader(open("rca/colar_lexslot_ledger.csv")))
def f(x):
    try: return float(x)
    except: return float("nan")
for r in ROWS:
    for k in ("AA","BWT","AF","FWT","diag_mean","final_old_mean","final_new",
              "recency_slope","boundary_drop","longrun_drop"):
        r[k] = f(r.get(k))
    r["recoveries"] = int(f(r.get("recoveries")) or 0)

# Reference baselines, from Reports sheet + all_runs.csv (3-seed means).
BASELINES = {
  ("dil","layoutlmv3"):          {"joint":88.7,"naive":41.3,"er":87.9,"der_pp":88.0,"slca":None,"ranpac":None,"best_any":88.4},
  ("cil_cord","layoutlmv3"):     {"joint":33.3,"naive":19.0,"er":16.0,"der_pp":18.2,"slca":None,"ranpac":None,"best_any":19.0},
  ("mixed","layoutlmv3"):        {"joint":67.5,"naive":33.9,"er":60.3,"der_pp":60.8,"slca":None,"ranpac":None,"best_any":62.2},
  ("cil_cifar100","vit"):        {"joint":89.5,"naive":11.7,"er":42.3,"der_pp":43.6,"er_b2000":76.2,"der_pp_b2000":76.5,"slca":87.5,"ranpac":85.3,"simplecil":81.7,"best_any":87.5},
  ("cil_imagenet_r","vit"):      {"joint":77.6,"naive":9.3, "er":21.6,"der_pp":20.2,"er_b2000":54.8,"der_pp_b2000":52.8,"slca":65.0,"ranpac":58.6,"simplecil":48.2,"best_any":65.0},
}

out = pathlib.Path("rca"); out.mkdir(exist_ok=True)

# ---------- Pass 2 (ledger part): replay_gap + plastic_gap per scenario ----------
with open(out/"pass2_gap_decomposition.md","w") as fo:
    fo.write("# Pass 2 — Gap decomposition (ledger-only terms)\n\n")
    fo.write("For each cell: `plastic_gap = joint - AA` (what the method leaves on the table) ")
    fo.write("and `replay_gap = AA - naive` (what replay bought). ")
    fo.write("Ridge-based terms (head/trunk/coverage) require the Pass-2b probe; marked TODO.\n\n")
    fo.write("| family | scenario | backbone | seed | AA | joint | naive | plastic_gap | replay_gap | head_gap | trunk_gap | coverage_gap |\n")
    fo.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in ROWS:
        b = BASELINES.get((r["scenario"], r["backbone"]), {})
        pg = b.get("joint", float("nan")) - r["AA"] if not math.isnan(r["AA"]) else float("nan")
        rg = r["AA"] - b.get("naive", float("nan")) if not math.isnan(r["AA"]) else float("nan")
        fo.write(f"| {r['family']} | {r['scenario']} | {r['backbone']} | {r['seed']} | "
                 f"{r['AA']:.1f} | {b.get('joint','?')} | {b.get('naive','?')} | "
                 f"{pg:+.1f} | {rg:+.1f} | TODO | TODO | TODO |\n")

# ---------- Pass 3: 2x2 flaw map (plasticity high/low x recovers y/n) ----------
with open(out/"pass3_flaw_map.md","w") as fo:
    fo.write("# Pass 3 — Flaw map (plasticity × recovery)\n\n")
    fo.write("- **High plasticity** = diag_mean ≥ 90 (new tasks are learned well).\n")
    fo.write("- **Recovers** = ≥1 off-diagonal `M[t][k] > M[k][k] + 1`.\n")
    fo.write("- **Boundary drop** = immediate `M[k][k] - M[k+1][k]`; **long-run drop** = `M[k][k] - M[T][k]`.\n\n")
    fo.write("| family | scenario | seed | diag | final_old | plastic | recov | boundary | long-run | slope |\n")
    fo.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in ROWS:
        plastic = "HIGH" if r["diag_mean"] >= 90 else "LOW"
        rec = "yes" if r["recoveries"] else "no"
        fo.write(f"| {r['family']} | {r['scenario']} | {r['seed']} | "
                 f"{r['diag_mean']:.1f} | {r['final_old_mean']:.1f} | {plastic} | {rec} ({r['recoveries']}) | "
                 f"{r['boundary_drop']:.1f} | {r['longrun_drop']:.1f} | {r['recency_slope']:+.2f} |\n")

# ---------- Pass 5: per-arm aggregation, sign agreement vs plain CoLaR ----------
def agg(key):
    buckets = collections.defaultdict(list)
    for r in ROWS:
        buckets[(r[key], r["scenario"], r["backbone"])].append(r)
    return buckets

by_arm = agg("family")
# For each arm with 3 seeds, compute mean/sd and sign agreement vs plain colar same scenario+backbone
plain = {(r["scenario"], r["backbone"], r["seed"]): r["AA"]
         for r in ROWS if r["family"] == "colar" and r["docs_per_task"] in ("50","500",None)
         and (r["rank_r"] in (None,"128","") or r["rank_r"]=="128")}
# Lenient: any colar entry per (scenario,backbone,seed) — pick first
plain2 = {}
for r in ROWS:
    if r["family"]=="colar":
        plain2.setdefault((r["scenario"], r["backbone"], r["seed"]), r["AA"])
plain = plain2

with open(out/"pass5_sign_rule.md","w") as fo:
    fo.write("# Pass 5 — Multi-seed verdict (mean, SD, sign rule vs plain CoLaR)\n\n")
    fo.write("| arm | scenario | backbone | n | mean AA | SD | per-seed AA | Δ vs plain | sign-agree 3/3 |\n")
    fo.write("|---|---|---|---|---|---|---|---|---|\n")
    for (arm, sc, bb), rs in sorted(by_arm.items()):
        aas = [x["AA"] for x in rs if not math.isnan(x["AA"])]
        if not aas: continue
        m = st.mean(aas); sd = st.pstdev(aas) if len(aas)>1 else 0.0
        per = ", ".join(f"s{x['seed']}={x['AA']:.1f}" for x in rs)
        deltas = []
        for x in rs:
            p = plain.get((sc, bb, x["seed"]))
            if p is not None: deltas.append(x["AA"] - p)
        if deltas:
            dstr = ", ".join(f"{d:+.1f}" for d in deltas)
            sign = "yes" if (all(d>0 for d in deltas) or all(d<0 for d in deltas)) and len(deltas)>=3 else ("n/a" if len(deltas)<3 else "no")
        else: dstr, sign = "", ""
        fo.write(f"| {arm} | {sc} | {bb} | {len(aas)} | {m:.2f} | {sd:.2f} | {per} | {dstr} | {sign} |\n")

print("pass2 + pass3 + pass5 written to rca/")
