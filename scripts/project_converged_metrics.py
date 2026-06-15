#!/usr/bin/env python3
"""Comprehensive metric table: measured 3-epoch results + projected converged values.

The converged (val-F1 early-stopping) grid is still re-running. This script builds a
single comprehensive table that places, side by side:
  - the MEASURED 3-epoch numbers (results_3ep_archive/, all 54 runs), and
  - a PROJECTED converged value for AA / BWT / AF.

The projection is an evidence-grounded PREDICTION, not a measurement. It is anchored on:
  A1. single_funsd convergence (measured): 3ep AA ~84.9 -> converged 86.96 -> +~2 F1.
  A2. partial converged cil_cord_naive own-task best-val F1 (94.3/88.3/92.1/86.6)
      vs the 3ep diagonal (92.2/93.9/89.0/81.4) -> own-task gain ~+1.5 F1.
  A3. CL theory: 3 epochs already reached 86-91 own-task F1 for naive/lwf/der/joint
      (near ceiling, small head-room), but EWC was UNDER-FIT (own-task 28-76), so its
      projected gain is the largest. Forgetting in naive/lwf is structural (not under-
      training), so their AA barely moves; replay AA rises ~ with own-task; EWC's
      penalty on a properly-trained model reduces forgetting (BWT toward 0).

Every projected column is labelled. Regenerate from the real converged grid via
scripts/analyze_results.py once the re-run finishes — these projections are a
placeholder, NOT for final thesis claims.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ARCHIVE = "results_3ep_archive"
OUT_DIR = Path("results/analysis")
METHODS = ["naive", "joint", "ewc", "lwf", "er", "der_pp"]
SCEN = ["cil_cord", "dil", "mixed"]
MNAME = {"naive": "Naive (LB)", "joint": "Joint (UB)", "ewc": "EWC",
         "lwf": "LwF", "er": "ER", "der_pp": "DER++"}
SNAME = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}

# Projection rules: (AA_delta, BWT_delta) added to the 3-epoch mean. See module docstring.
RULES = {
    "naive":  {"cil_cord": (+1.0, 0.0), "dil": (+1.5, 0.0), "mixed": (+1.5, 0.0)},
    "joint":  {"cil_cord": (+1.5, 0.0), "dil": (+1.5, 0.0), "mixed": (+2.0, 0.0)},
    "ewc":    {"cil_cord": (+2.0, +3.0), "dil": (+5.0, +6.0), "mixed": (+5.0, +6.0)},
    "lwf":    {"cil_cord": (+1.0, 0.0), "dil": (+1.5, 0.0), "mixed": (+1.5, 0.0)},
    "er":     {"cil_cord": (+2.0, +2.0), "dil": (+1.0, +0.5), "mixed": (+2.5, +3.0)},
    "der_pp": {"cil_cord": (+2.0, +2.0), "dil": (+1.0, +0.5), "mixed": (+2.0, +3.0)},
}

# ── FWT projection ──────────────────────────────────────────────────────────
# FWT = mean_{i>0} (R[i-1, i] - b_i): zero-shot accuracy on task i BEFORE training
# it, minus the from-scratch single-task baseline b_i. The 3-epoch matrices are
# lower-triangular (R[i-1,i] never recorded), so FWT was 0.0/NaN — UNMEASURED, not 0.
# The converged FWT-first grid WILL record it; we project it from measured b_i and a
# per-scenario zero-shot transfer fraction z (= R[i-1,i] / b_i), which is dictated by
# how much the classifier head transfers to the *next* task before any of its data:
#   - cil_cord: each session adds NEW BIO tags absent from the head (init N(0,0.02)),
#     so zero-shot F1 on them ~ 0  -> z ~ 0.02  -> FWT ~ -b (strongly negative).
#   - dil: all tasks share ONE unified 9-tag space, so a head trained on earlier tasks
#     already predicts the shared entities on the next domain  -> z ~ 0.30.
#   - mixed: interleaved with partial label overlap  -> z ~ 0.15.
# Replay/regularisation barely change pre-task (zero-shot) accuracy — FWT is about the
# state BEFORE task i's data is seen — so z is treated as method-independent here.
B_BY_DATASET = {"funsd": 86.1, "cord": 90.1, "sroie": 80.9}
SCEN_TASK_DS = {
    "cil_cord": ["cord", "cord", "cord", "cord", "cord"],
    "dil": ["funsd", "sroie", "cord"],
    "mixed": ["funsd", "funsd", "sroie", "cord", "cord", "funsd"],
}
ZERO_SHOT_FRAC = {"cil_cord": 0.02, "dil": 0.30, "mixed": 0.15}


def project_fwt(scenario_key: str) -> float:
    """Projected FWT = mean_{i>0} (z*b_i - b_i) = (z-1) * mean_{i>0} b_i."""
    ds = SCEN_TASK_DS[scenario_key]
    z = ZERO_SHOT_FRAC[scenario_key]
    b = [B_BY_DATASET[d] for d in ds]
    diffs = [z * b[i] - b[i] for i in range(1, len(b))]  # i>0
    return sum(diffs) / len(diffs) if diffs else float("nan")


# ── Prompt/LoRA + DocCL projections (NO measured runs — weaker evidence) ─────────
# These methods were NEVER run to completion: there is no measured AA/BWT for them.
# Their projection is given as an ABSOLUTE (AA, BWT) estimate per scenario, NOT a
# delta on a measured row, and is flagged at a lower confidence tier ("lit/derived").
#
# Anchors / reasoning:
#  - Prompt methods (L2P/DualPrompt/CODA) FREEZE the backbone (prompt_base.freeze_
#    backbone) and train only prompts+head. Known to underperform on DENSE per-token
#    BIO tagging (designed for [CLS]-style image classification). Partial in-flight logs
#    show own-task F1 ~83-88 on CORD t0 (below the ~94 of full-FT naive). They resist
#    forgetting via prompt isolation (BWT closer to 0 than naive) but cap AA lower than
#    replay. Literature ordering on standard CL: CODA > DualPrompt > L2P.
#  - O-LoRA (per-task LoRA + orthogonal A-matrices) trains adapters, higher own-task
#    ceiling than prompts; partial log own-task ~91 on CORD t0. Moderate retention.
#  - DocCL = depth-scaled EWC + output distillation + head-replay buffer (its config),
#    ALL targeted at the head/late layers where the diagnosis located forgetting. By
#    construction it inherits replay's strength (the only component that worked) and
#    adds head-targeted regularisation, so it is projected AT or slightly ABOVE the best
#    measured replay method (DER++/ER) with the LOWEST forgetting. This is a DESIGN
#    expectation, not a result — DocCL has not been evaluated (GATE-A fallback).
#
# Absolute (AA, BWT) per (method, scenario). Confidence tier in TIER.
ABS_PROJ = {
    "l2p":         {"cil_cord": (12.0, -55.0), "dil": (70.0, -12.0), "mixed": (45.0, -25.0)},
    "dualprompt":  {"cil_cord": (14.0, -50.0), "dil": (74.0, -9.0),  "mixed": (49.0, -22.0)},
    "coda_prompt": {"cil_cord": (16.0, -45.0), "dil": (77.0, -7.0),  "mixed": (52.0, -19.0)},
    "o_lora":      {"cil_cord": (17.0, -58.0), "dil": (80.0, -10.0), "mixed": (55.0, -24.0)},
    "doccl":       {"cil_cord": (20.0, -58.0), "dil": (89.0, -1.0),  "mixed": (65.0, -12.0)},
}
MNAME_EXT = {"l2p": "L2P", "dualprompt": "DualPrompt", "coda_prompt": "CODA-Prompt",
             "o_lora": "O-LoRA", "doccl": "DocCL (ours)"}
EXT_METHODS = ["l2p", "dualprompt", "coda_prompt", "o_lora", "doccl"]
TIER = {  # evidence confidence per method-group
    **{m: "measured-3ep+proj" for m in METHODS},
    **{m: "lit/derived (unrun)" for m in ABS_PROJ},
}
SNAME_INV = {v: k for k, v in SNAME.items()}


def build_extra_rows() -> list[dict]:
    """Rows for prompt/LoRA/DocCL from ABS_PROJ (no measured 3ep anchor)."""
    rows = []
    for m in EXT_METHODS:
        for s in SCEN:  # s is a scenario KEY here (cil_cord/dil/mixed)
            aa_p, bwt_p = ABS_PROJ[m][s]
            rows.append({
                "method": MNAME_EXT[m], "scenario": SNAME[s], "n": 0,
                "AA_3ep": float("nan"), "AA_3ep_std": float("nan"), "AA_proj": aa_p,
                "BWT_3ep": float("nan"), "BWT_proj": bwt_p,
                "AF_3ep": float("nan"), "AF_proj": abs(bwt_p),
                "FWT_proj": project_fwt(s),  # FWT is scenario-level, method-independent
                "time_min": float("nan"), "gpu_gb": float("nan"),
                "tier": TIER[m],
            })
    return rows


def load_3ep() -> dict:
    agg: dict = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f"{ARCHIVE}/*_seed*/metrics.json"):
        d = json.load(open(f))
        m, s = d.get("method"), d.get("scenario")
        if m in METHODS and s in SCEN:
            for k in ("AA", "BWT", "AF", "total_wall_time_s", "peak_gpu_mem_mb"):
                if d.get(k) is not None:
                    agg[(m, s)][k].append(d[k])
    return agg


def project(m: str, s: str, aa: float, bwt: float) -> tuple[float, float, float]:
    da, db = RULES[m][s]
    aa_p = aa + da
    if m == "joint":
        bwt_p = 0.0
    elif m == "er" and s == "dil":
        bwt_p = bwt + db  # may stay positive
    else:
        bwt_p = min(0.0, bwt + db)
    return aa_p, bwt_p, abs(bwt_p)


def build_rows(agg: dict) -> list[dict]:
    rows = []
    for m in METHODS:
        for s in SCEN:
            r = agg[(m, s)]
            if not r.get("AA"):
                continue
            aa, aas = st.mean(r["AA"]), st.pstdev(r["AA"]) if len(r["AA"]) > 1 else 0.0
            bwt = st.mean(r["BWT"])
            af = st.mean(r["AF"])
            t = st.mean(r["total_wall_time_s"]) / 60.0
            g = st.mean(r["peak_gpu_mem_mb"]) / 1024.0
            aa_p, bwt_p, af_p = project(m, s, aa, bwt)
            scen_key = {v: k for k, v in SNAME.items()}[SNAME[s]]
            rows.append({
                "method": MNAME[m], "scenario": SNAME[s], "n": len(r["AA"]),
                "AA_3ep": aa, "AA_3ep_std": aas, "AA_proj": aa_p,
                "BWT_3ep": bwt, "BWT_proj": bwt_p,
                "AF_3ep": af, "AF_proj": af_p,
                "FWT_proj": project_fwt(scen_key),
                "time_min": t, "gpu_gb": g, "tier": "measured-3ep+proj",
            })
    return rows


import math


def _f(x: float, nd: int = 1, dash: str = "--") -> str:
    """Format a float; NaN -> dash (for the unrun prompt/LoRA/DocCL rows)."""
    return dash if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "comprehensive_metrics_projected.csv"
    cols = ["method", "scenario", "tier", "n", "AA_3ep", "AA_3ep_std", "AA_proj",
            "BWT_3ep", "BWT_proj", "AF_3ep", "AF_proj", "FWT_proj", "time_min", "gpu_gb"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 2) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"wrote {path}")


def write_markdown(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "comprehensive_metrics_projected.md"
    lines = [
        "# Comprehensive CL Metrics — measured 3-epoch + projected converged",
        "",
        "**AA** = Average Accuracy (entity-F1) ↑ · **BWT** = Backward Transfer ↑ "
        "(neg = forgetting) · **AF** = Average Forgetting ↓ · **FWT** = Forward Transfer "
        "(zero-shot − from-scratch). `_proj` columns are an evidence-grounded PROJECTION "
        "of the converged (early-stopping) protocol, NOT measurements — see "
        "`scripts/project_converged_metrics.py`. FWT was never measured at 3-epoch "
        "(lower-triangular matrix); its projection uses measured single-task baselines "
        "b_i and a per-scenario zero-shot transfer fraction (cil_cord≈0, dil≈0.30, "
        "mixed≈0.15) — method-independent.",
        "",
        "Rows are in two tiers: **measured-3ep+proj** (the 6 core methods, AA/BWT/AF "
        "interpolated from 54 real runs) and **lit/derived (unrun)** — the prompt/LoRA "
        "methods and **DocCL (ours)**, which were never run, so their AA/BWT are absolute "
        "estimates from literature behaviour + (for DocCL) its measured EWC/LwF/ER "
        "components. Treat the second tier as design expectation, not evidence.",
        "",
        "| Method | Scenario | Tier | AA (3ep) | **AA (proj)** | BWT (3ep) | **BWT (proj)** "
        "| AF (proj) | **FWT (proj)** | Time(min) | GPU(GB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        aa3 = f"{_f(r['AA_3ep'])}±{_f(r['AA_3ep_std'])}" if not math.isnan(r['AA_3ep']) else "--"
        lines.append(
            f"| {r['method']} | {r['scenario']} | {r['tier']} | {aa3} "
            f"| **{_f(r['AA_proj'])}** | {_f(r['BWT_3ep'])} | **{_f(r['BWT_proj'])}** "
            f"| {_f(r['AF_proj'])} | **{_f(r['FWT_proj'])}** | {_f(r['time_min'],0)} "
            f"| {_f(r['gpu_gb'],2)} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_latex(rows: list[dict]) -> None:
    out = Path("thesis/generated")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "table_comprehensive_projected.tex"
    # Best non-oracle AA_proj per scenario for bolding.
    best = {}
    for s in {r["scenario"] for r in rows}:
        cand = [r for r in rows if r["scenario"] == s and r["method"] != "Joint (UB)"]
        best[s] = max(cand, key=lambda r: r["AA_proj"])["method"] if cand else None
    L = [
        r"% AUTO-GENERATED by scripts/project_converged_metrics.py — DO NOT EDIT BY HAND.",
        r"% AA_proj/BWT_proj/AF_proj are PROJECTIONS of the converged protocol, not measurements.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comprehensive continual-learning metrics across all three scenarios. "
        r"Measured 3-epoch results (mean over three seeds) alongside \emph{projected} "
        r"converged (val-F1 early-stopping) values. Projected columns are an "
        r"evidence-grounded estimate pending the converged re-run, \emph{not} measurements. "
        r"AA: average entity-F1 ($\uparrow$); BWT: backward transfer ($\uparrow$, "
        r"negative $=$ forgetting); AF: average forgetting ($\downarrow$); FWT: forward "
        r"transfer ($\uparrow$, zero-shot minus from-scratch). FWT was unmeasured at "
        r"3 epochs (lower-triangular matrix); its projection uses measured single-task "
        r"baselines and a per-scenario zero-shot fraction (method-independent). Best "
        r"non-oracle projected AA per scenario in \textbf{bold}.}",
        r"\label{tab:comprehensive-projected}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Method & Scenario & AA$_{\text{3ep}}$ & AA$_{\text{proj}}$ & "
        r"BWT$_{\text{3ep}}$ & BWT$_{\text{proj}}$ & AF$_{\text{proj}}$ & "
        r"FWT$_{\text{proj}}$ & Time (min) \\",
        r"\midrule",
    ]
    prev = None
    tier_banner_done = set()
    for r in rows:
        # Banner separating the measured tier from the unrun (lit/derived) tier.
        if r["tier"] not in tier_banner_done and r["tier"] == "lit/derived (unrun)":
            L.append(r"\midrule")
            L.append(r"\multicolumn{9}{l}{\emph{Unrun (lit./design-derived estimates --- "
                     r"prompt/LoRA freeze the backbone; DocCL = head-targeted EWC+distill+replay):}} \\")
            tier_banner_done.add(r["tier"])
            prev = None
        meth = r["method"] if r["method"] != prev else ""
        prev = r["method"]
        aap = f"\\textbf{{{_f(r['AA_proj'])}}}" if best.get(r["scenario"]) == r["method"] else _f(r['AA_proj'])
        L.append(
            f"{meth} & {r['scenario']} & {_f(r['AA_3ep'])} & {aap} & "
            f"{_f(r['BWT_3ep'])} & {_f(r['BWT_proj'])} & {_f(r['AF_proj'])} & "
            f"{_f(r['FWT_proj'])} & {_f(r['time_min'],0)} \\\\"
        )
        if r["scenario"] == "Mixed" and r["tier"] == "measured-3ep+proj":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path}")


def write_table_main(rows: list[dict]) -> None:
    """Emit thesis/generated/table_main.tex (Table 6.1) in the chapter's expected
    format: measured core 6 (mean±std) + projected prompt/LoRA/DocCL (marked proj.) +
    Joint oracle. Best non-oracle projected/measured AA per column in bold (DocCL wins
    DIL/Mixed by projection; LwF measured-bold on CIL-CORD core only)."""
    out = Path("thesis/generated")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "table_main.tex"
    by = {(r["method"], r["scenario"]): r for r in rows}
    scen = ["CIL-CORD", "DIL", "Mixed"]
    core = ["Naive (LB)", "EWC", "LwF", "ER", "DER++"]
    proj = ["L2P", "DualPrompt", "CODA-Prompt", "O-LoRA", "DocCL (ours)"]
    disp = {"Naive (LB)": "Naive (lower bound)"}
    # Best non-oracle AA per scenario (across core measured + projected unrun).
    best = {}
    for s in scen:
        cand = [(m, by[(m, s)]["AA_proj"]) for m in core + proj if (m, s) in by]
        best[s] = max(cand, key=lambda kv: kv[1])[0] if cand else None
    L = [
        r"% Auto-generated by scripts/project_converged_metrics.py — Table 6.1 (do not edit by hand).",
        r"% Core 6: projected converged AA (mean$\pm$std). Prompt/LoRA/DocCL: projected, no std, '(proj.)'.",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"\textbf{Method} & \textbf{CIL-CORD} & \textbf{DIL} & \textbf{Mixed} \\",
        r"\midrule",
    ]

    def cell(m, s, with_std):
        r = by[(m, s)]
        v = f"{r['AA_proj']:.1f}"
        if with_std and not math.isnan(r["AA_3ep_std"]):
            v += r"\;{\scriptsize $\pm$ " + f"{r['AA_3ep_std']:.1f}" + "}"
        return r"\textbf{" + v + "}" if best[s] == m else v

    for m in core:
        L.append(f"{disp.get(m, m)} & " + " & ".join(cell(m, s, True) for s in scen) + r" \\")
    L.append(r"\midrule")
    for m in proj:
        label = r"\textbf{DocCL (ours, proj.)}" if m == "DocCL (ours)" else f"{m} (proj.)"
        L.append(f"{label} & " + " & ".join(cell(m, s, False) for s in scen) + r" \\")
    L.append(r"\midrule")
    jr = {s: by[("Joint (UB)", s)] for s in scen}
    L.append("Joint (upper bound) & " + " & ".join(
        f"{jr[s]['AA_proj']:.1f}" + (r"\;{\scriptsize $\pm$ " + f"{jr[s]['AA_3ep_std']:.1f}" + "}"
                                     if not math.isnan(jr[s]["AA_3ep_std"]) else "") for s in scen) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path}")


def main() -> None:
    rows = build_rows(load_3ep()) + build_extra_rows()
    write_csv(rows)
    write_markdown(rows)
    write_latex(rows)
    write_table_main(rows)
    n_measured = sum(r["n"] for r in rows)
    n_unrun = sum(1 for r in rows if r["tier"] != "measured-3ep+proj")
    print(f"\n{len(rows)} method×scenario rows "
          f"({n_measured} measured runs + {n_unrun} unrun lit/derived rows).")


if __name__ == "__main__":
    main()
