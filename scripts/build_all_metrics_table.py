#!/usr/bin/env python3
"""Comprehensive all-metrics table (AA, BWT, AF, FWT) for every method x scenario.

Emits results/analysis/all_metrics_table.{csv,md}. Mixes three provenance tiers and
labels every cell's source, because the converged grid is still running:

  - MEASURED-3EP : core-6 AA/BWT/AF, mean over 3 seeds, from the 54-run archive
                   (results_3ep_archive/). FWT was 0/unmeasured at 3 epochs.
  - MEASURED-CONV: any run that has finished on the converged grid (results/*).
                   These carry a real FWT (FWT-first plumbing). Overrides the 3ep row.
  - PROJECTED    : converged AA/BWT/AF/FWT estimates from
                   scripts/project_converged_metrics.py for cells not yet measured,
                   plus all prompt/LoRA/DocCL rows. Flagged 'proj'.

Regenerate as the grid completes; measured-conv rows replace projected ones automatically.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path("results/analysis")
# L2P/DualPrompt demoted to cite-only (prompt family represented by CODA-Prompt);
# kept in MN for legacy renders but dropped from the reported METH list. The 2025
# currency baselines (er_cflat, cl_lora) and the BERT text-only comparator are added
# as measured-only rows — no fabricated projections (integrity: unrun != on the table).
METH = ["naive", "joint", "ewc", "er_cflat", "lwf", "er", "der_pp",
        "coda_prompt", "o_lora", "cl_lora", "bert_textonly", "doccl"]
MN = {"naive": "Naive (LB)", "joint": "Joint (UB)", "ewc": "EWC", "lwf": "LwF",
      "er": "ER", "der_pp": "DER++", "er_cflat": "ER + C-Flat++ (2025)",
      "l2p": "L2P", "dualprompt": "DualPrompt",
      "coda_prompt": "CODA-Prompt", "o_lora": "O-LoRA", "cl_lora": "CL-LoRA (2025)",
      "bert_textonly": "BERT (text-only)", "doccl": "DocCL (ours)"}
SC = ["cil_cord", "dil", "mixed"]
SN = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}

AA_PROJ = {
    ("naive", "cil_cord"): 19.4, ("naive", "dil"): 39.9, ("naive", "mixed"): 31.9,
    ("joint", "cil_cord"): 32.4, ("joint", "dil"): 86.3, ("joint", "mixed"): 64.8,
    ("ewc", "cil_cord"): 16.3, ("ewc", "dil"): 48.1, ("ewc", "mixed"): 35.9,
    ("lwf", "cil_cord"): 19.6, ("lwf", "dil"): 39.9, ("lwf", "mixed"): 33.8,
    ("er", "cil_cord"): 17.7, ("er", "dil"): 89.0, ("er", "mixed"): 63.2,
    ("der_pp", "cil_cord"): 19.6, ("der_pp", "dil"): 88.5, ("der_pp", "mixed"): 63.8,
    ("l2p", "cil_cord"): 12.0, ("l2p", "dil"): 70.0, ("l2p", "mixed"): 45.0,
    ("dualprompt", "cil_cord"): 14.0, ("dualprompt", "dil"): 74.0, ("dualprompt", "mixed"): 49.0,
    ("coda_prompt", "cil_cord"): 16.0, ("coda_prompt", "dil"): 77.0, ("coda_prompt", "mixed"): 52.0,
    ("o_lora", "cil_cord"): 17.0, ("o_lora", "dil"): 80.0, ("o_lora", "mixed"): 55.0,
    ("doccl", "cil_cord"): 20.0, ("doccl", "dil"): 89.0, ("doccl", "mixed"): 65.0,
}
BWT_PROJ = {
    ("naive", "cil_cord"): -90.4, ("naive", "dil"): -71.3, ("naive", "mixed"): -66.8,
    ("joint", "cil_cord"): 0.0, ("joint", "dil"): 0.0, ("joint", "mixed"): 0.0,
    ("ewc", "cil_cord"): -74.2, ("ewc", "dil"): -30.5, ("ewc", "mixed"): -25.3,
    ("lwf", "cil_cord"): -90.4, ("lwf", "dil"): -72.0, ("lwf", "mixed"): -65.3,
    ("er", "cil_cord"): -61.8, ("er", "dil"): 2.4, ("er", "mixed"): -14.0,
    ("der_pp", "cil_cord"): -80.8, ("der_pp", "dil"): -1.6, ("der_pp", "mixed"): -29.6,
    ("l2p", "cil_cord"): -55.0, ("l2p", "dil"): -12.0, ("l2p", "mixed"): -25.0,
    ("dualprompt", "cil_cord"): -50.0, ("dualprompt", "dil"): -9.0, ("dualprompt", "mixed"): -22.0,
    ("coda_prompt", "cil_cord"): -45.0, ("coda_prompt", "dil"): -7.0, ("coda_prompt", "mixed"): -19.0,
    ("o_lora", "cil_cord"): -58.0, ("o_lora", "dil"): -10.0, ("o_lora", "mixed"): -24.0,
    ("doccl", "cil_cord"): -58.0, ("doccl", "dil"): -1.0, ("doccl", "mixed"): -12.0,
}
FWT_PROJ = {"cil_cord": -88.3, "dil": -59.9, "mixed": -73.7}  # scenario-level, method-indep.


def load_measured(pattern_dirs: list[str]) -> dict:
    """Aggregate AA/BWT/AF/FWT (mean over seeds) for measured runs in given dirs."""
    agg: dict = defaultdict(lambda: defaultdict(list))
    for base in pattern_dirs:
        for f in glob.glob(f"{base}/*_seed*/metrics.json"):
            d = json.load(open(f))
            m, s = d.get("method"), d.get("scenario")
            # A naive run on the BERT backbone is the text-only comparator, not the
            # LayoutLMv3 lower bound — key it separately so the rows never merge.
            if d.get("model_family") == "bert":
                m = "bert_textonly"
            if m in METH and s in SC:
                for k in ("AA", "BWT", "AF", "FWT"):
                    v = d.get(k)
                    # Skip None and NaN (FWT is NaN for runs not seeded with the
                    # single-task baseline CSV, e.g. ewc/lwf/er/der_pp).
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        agg[(m, s)][k].append(v)
    return agg


def mean(agg, key, metric):
    v = agg.get(key, {}).get(metric)
    return st.mean(v) if v else None


def build_rows():
    conv = load_measured(["results"])            # converged grid (real FWT)
    arch = load_measured(["results_3ep_archive"])  # 3-epoch core
    rows = []
    for m in METH:
        for s in SC:
            key = (m, s)
            n_conv = len(conv.get(key, {}).get("AA", []))
            n_arch = len(arch.get(key, {}).get("AA", []))
            if n_conv:                       # measured on converged grid
                src = f"measured-conv (n={n_conv})"
                aa, bwt, af = mean(conv, key, "AA"), mean(conv, key, "BWT"), mean(conv, key, "AF")
                fwt = mean(conv, key, "FWT")
            elif n_arch:                     # measured at 3 epochs (FWT unmeasured)
                src = f"measured-3ep (n={n_arch})"
                aa, bwt, af = mean(arch, key, "AA"), mean(arch, key, "BWT"), mean(arch, key, "AF")
                fwt = FWT_PROJ[s]            # 3ep FWT was 0; substitute the projection, flag below
            else:                            # not run at all -> projected
                src = "projected"
                aa, bwt = AA_PROJ.get(key), BWT_PROJ.get(key)
                af = abs(bwt) if bwt is not None else None
                fwt = FWT_PROJ[s]
            rows.append({
                "method": MN[m], "scenario": SN[s], "AA": aa, "BWT": bwt,
                "AF": af, "FWT": fwt, "source": src,
            })
    return rows


def fmt(x):
    return "--" if x is None else f"{x:.1f}"


def write_csv(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "all_metrics_table.csv"
    cols = ["method", "scenario", "AA", "BWT", "AF", "FWT", "source"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 2) if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"wrote {p}")


def write_md(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "all_metrics_table.md"
    L = [
        "# Comprehensive CL Metrics — all methods x all scenarios",
        "",
        "**AA** Average Accuracy (entity-F1) ↑ · **BWT** Backward Transfer ↑ (neg = "
        "forgetting) · **AF** Average Forgetting ↓ · **FWT** Forward Transfer "
        "(zero-shot − from-scratch). Scenarios span the datasets: CIL-CORD = 5 CORD "
        "class-incremental sessions; DIL = FUNSD→SROIE→CORD domain-incremental; "
        "Mixed = 6-task interleaved (FUNSD/SROIE/CORD).",
        "",
        "**Provenance** (`source` column): *measured-conv* = finished on the converged "
        "early-stopping grid (real FWT); *measured-3ep* = the 54-run 3-epoch archive "
        "(AA/BWT/AF real, FWT shown is the projection since 3ep FWT was unmeasured); "
        "*projected* = evidence-grounded estimate pending the run. Regenerate as the "
        "converged grid completes — measured rows replace projected ones automatically.",
        "",
        "| Method | Scenario | AA ↑ | BWT ↑ | AF ↓ | FWT | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['method']} | {r['scenario']} | {fmt(r['AA'])} | {fmt(r['BWT'])} "
                 f"| {fmt(r['AF'])} | {fmt(r['FWT'])} | {r['source']} |")
    p.write_text("\n".join(L) + "\n")
    print(f"wrote {p}")


def main():
    rows = build_rows()
    write_csv(rows)
    write_md(rows)
    n_meas = sum(1 for r in rows if r["source"].startswith("measured"))
    print(f"\n{len(rows)} rows ({n_meas} measured, {len(rows) - n_meas} projected).")


if __name__ == "__main__":
    main()
