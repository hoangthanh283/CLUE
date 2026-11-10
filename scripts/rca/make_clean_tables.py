"""Generate every number of the CVPR rewrite from result files (no hand-typed cells).

Writes into paper_cvpr/:
  vals.tex            \\newcommand macros for the headline cells (used by abstract, intro, teaser)
  tab_2x2.tex         memory x readout on CIL-CORD (+ isolation row)
  tab_main_clean.tex  comparison table (CIL-CORD AA, CIL-CORD full-label F1, Mixed AA, DIL AA)

Each cell lists its sources in priority order. A cell uses the CLEAN protocol
(results/clean_2026-11: fp32, held-out early stopping) once all three seeds exist there; until
then it falls back to the legacy run and is marked with a dagger. n < 3 prints per-seed values.

    uv run python scripts/rca/make_clean_tables.py
"""

import json
import re
from glob import glob
from pathlib import Path

import numpy as np

SEEDS = (42, 7, 123)
CLEAN, GATE, RESC = "results/clean_2026-11", "results/gates/cil_protocol_2026-11", "results/rescore"
OUT = Path("paper_cvpr")


def load(root: str, name: str, key: str = "AA", suffix: str = "*") -> list[float]:
    """Metric ``key`` of run ``<name>_seed<s>`` under ``root`` for every seed that exists."""
    vals = []
    for s in SEEDS:
        hits = sorted(glob(f"{root}/{name}_seed{s}{suffix}/**/metrics.json", recursive=True))
        if hits:
            v = json.loads(Path(hits[0]).read_text()).get(key)
            if v is not None:
                vals.append(float(v))
    return vals


def pick(sources, key="AA", drop_zero=False):
    """First source with 3 seeds wins; else the first non-empty one. -> (values, is_clean)."""
    got = [(load(r, n, key, sfx), r == CLEAN) for r, n, sfx in sources]
    if drop_zero:
        got = [([v for v in vs if v > 0.5], c) for vs, c in got]
    for vs, c in got:
        if len(vs) == 3:
            return vs, c
    for vs, c in got:
        if vs:
            return vs, c
    return [], False


def fmt(vs, clean, bold=False, mark=""):
    if not vs:
        return "--"
    if len(vs) >= 3:
        body = f"{np.mean(vs):.1f}{{\\pm}}{np.std(vs, ddof=1):.1f}"
    else:
        body = "/".join(f"{v:.1f}" for v in vs)
    if bold:
        body = f"\\mathbf{{{body}}}"
    sup = ",".join(x for x in ("\\dagger" if not clean else "", mark) if x)
    return f"${body}^{{{sup}}}$" if sup else f"${body}$"


def src(clean_name, *legacy):
    """Sources for one cell: the clean run first, then (root, name[, suffix]) fallbacks."""
    out = [(CLEAN, clean_name, "*")] if clean_name else []
    return out + [(t[0], t[1], t[2] if len(t) > 2 else "*") for t in legacy]


# ---------------------------------------------------------------------------- cell registry
# "" suffix = the run dir is exactly <name>_seed<s> (avoids matching <name>_seed<s>_bros etc.)
C = {
    "naive": src("cil_cord_naive", (RESC, "cil_cord_naive", "")),
    "nobank": src("cil_cord_colar_sbr_noreplay", (GATE, "cil_cord_colar_sbr_noreplay")),
    "colar": src("cil_cord_colar", (RESC, "cil_cord_colar", "")),
    "sbr": src("cil_cord_colar_sbr", (GATE, "cil_cord_colar_sbr", "")),
    "cllora": src("cil_cord_cl_lora", (RESC, "cil_cord_cl_lora", "")),
    "recall": src("cil_cord_colar_sbr_recall_r64", (GATE, "cil_cord_colar_sbr_recall_r64")),
    "joint": src("cil_cord_joint", (RESC, "cil_cord_joint", "")),
    "er": src("cil_cord_er250", (RESC, "cil_cord_er", "")),
    "slot": src(None, (GATE, "cil_cord_colar_sbr_slot")),
    "sbr_bros": src("cil_cord_colar_sbr_bros"),
}
M = {  # Mixed
    "naive": src("mixed_naive", (RESC, "mixed_naive", "")),
    "colar": src("mixed_colar", (RESC, "mixed_colar", "")),
    "sbr": src("mixed_colar_sbr", (RESC, "mixed_colar_sbr", "")),
    "joint": src("mixed_joint", (RESC, "mixed_joint", "")),
    "er": src(None, (RESC, "mixed_er", "")),
    "slot": src(None, (GATE, "mixed_colar_sbr_slot")),
    "recall": src(None, (GATE, "mixed_colar_sbr_recall_r64c")),
}
D = {  # DIL (scoring unaffected by the restriction; legacy = fp32 grid, test-split stopping)
    "naive": src("dil_naive", ("results", "dil_naive", "")),
    "colar": src("dil_colar", ("results", "dil_colar_e10", "_k4_d50_r128")),
    "sbr": src("dil_colar_sbr", (GATE, "dil_colar_sbr2")),
    "joint": src("dil_joint", ("results", "dil_joint", "")),
    "er": src("dil_er", ("results", "dil_er", "")),
    "slot": src(None, (GATE, "dil_colar_sbr_slot")),
}
BASE = [  # (label, rescore/grid method key, drop exact-zero collapses)
    ("LwF", "lwf", True),
    ("DER++", "der_pp", False),
    ("CODA-Prompt", "coda_prompt", False),
    ("CL-LoRA (2025)", "cl_lora", False),
    ("CPFD (2023)", "cpfd", False),
    ("IS3 (2024)", "is3", True),
    ("LCA (2026)", "lca", False),
]


def macro(name, vs):
    """Two macros per headline cell: \\v<Name> (mean, or mean±sd at n=3) and \\n<Name> (mean)."""
    if not vs:
        return f"\\newcommand{{\\v{name}}}{{--}}\\newcommand{{\\n{name}}}{{0}}\n"
    pm = (
        f"{np.mean(vs):.1f}{{\\pm}}{np.std(vs, ddof=1):.1f}"
        if len(vs) >= 3
        else f"{np.mean(vs):.1f}"
    )
    return f"\\newcommand{{\\v{name}}}{{{pm}}}\\newcommand{{\\n{name}}}{{{np.mean(vs):.1f}}}\n"


def main():
    cell = {k: pick(v) for k, v in C.items()}
    mix = {k: pick(v) for k, v in M.items()}
    dil = {k: pick(v) for k, v in D.items()}
    full = {
        k: pick(C[k], "full_label_f1") for k in ("naive", "colar", "sbr", "recall", "joint", "er")
    }

    # ---- vals.tex
    names = {
        "naive": "Naive", "nobank": "NoBank", "colar": "Colar", "sbr": "Sbr", "cllora": "ClLora",
        "recall": "Recall", "joint": "Joint", "er": "Er", "slot": "Slot", "sbr_bros": "SbrBros",
    }  # fmt: skip
    v = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    for k, n in names.items():
        v += macro(n, cell[k][0])
    for k, n in (("sbr", "SbrMix"), ("joint", "JointMix"), ("colar", "ColarMix"), ("er", "ErMix")):
        v += macro(n, mix[k][0])
    for k, n in (("sbr", "SbrDil"), ("joint", "JointDil"), ("colar", "ColarDil"), ("er", "ErDil")):
        v += macro(n, dil[k][0])
    for k, n in (("sbr", "SbrFull"), ("recall", "RecallFull"), ("joint", "JointFull")):
        v += macro(n, full[k][0])
    (OUT / "vals.tex").write_text(v)

    # ---- tab_2x2.tex
    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{lcc}\n\\toprule\n & \\textbf{Shared softmax} & \\textbf{Session-blocked readout} \\\\\n\\midrule\n"
    t += f"No memory & {fmt(*cell['naive'])} & {fmt(*cell['nobank'])} \\\\\n"
    t += f"Whole-document memory (CoLaR) & {fmt(*cell['colar'])} & {fmt(*cell['sbr'], bold=True)} \\\\\n"
    t += f"Per-session isolation, no memory & {fmt(*cell['cllora'])} & {fmt(*cell['recall'], bold=True)} \\\\\n"
    t += f"\\midrule\nJoint training (reference) & \\multicolumn{{2}}{{c}}{{{fmt(*cell['joint'])}}} \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_2x2.tex").write_text(t)

    # ---- tab_main_clean.tex
    def row(label, c, f, m, d, bold=False):
        lab = f"\\textbf{{{label}}}" if bold else label
        return f"{lab} & {c} & {f} & {m} & {d} \\\\\n"

    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{lcccc}\n\\toprule\n"
    t += " & \\multicolumn{2}{c}{\\textbf{CIL-CORD}} & \\textbf{Mixed} & \\textbf{DIL} \\\\\n"
    t += "\\textbf{Method} & AA & full-label F1 & AA & AA \\\\\n\\midrule\n"
    t += row(
        "Naive", fmt(*cell["naive"]), fmt(*full["naive"]), fmt(*mix["naive"]), fmt(*dil["naive"])
    )
    t += row(
        "ER, matched memory", fmt(*cell["er"]), fmt(*full["er"]), fmt(*mix["er"]), fmt(*dil["er"])
    )
    for label, key, dz in BASE:
        mark = "c" if dz else ""
        cc = pick(src(None, (RESC, f"cil_cord_{key}", "")), drop_zero=dz)
        mm = pick(src(None, (RESC, f"mixed_{key}", "")), drop_zero=dz)
        dd = pick(src(None, ("results", f"dil_{key}", "")))
        t += row(label, fmt(*cc, mark=mark), "--", fmt(*mm, mark=mark), fmt(*dd))
    t += "\\midrule\n"
    t += row(
        "CoLaR", fmt(*cell["colar"]), fmt(*full["colar"]), fmt(*mix["colar"]), fmt(*dil["colar"])
    )
    t += row(
        "CoLaR-SBR",
        fmt(*cell["sbr"]),
        fmt(*full["sbr"]),
        fmt(*mix["sbr"], bold=True),
        fmt(*dil["sbr"]),
        bold=True,
    )
    t += row(
        "\\quad + LexSlot head slots",
        fmt(*cell["slot"]),
        "--",
        fmt(*mix["slot"]),
        fmt(*dil["slot"]),
    )
    t += row(
        "SBR-Recall",
        fmt(*cell["recall"], bold=True),
        fmt(*full["recall"]),
        fmt(*mix["recall"], mark="b"),
        "--",
        bold=True,
    )
    t += "\\midrule\n"
    t += row(
        "Joint training (reference)",
        fmt(*cell["joint"]),
        fmt(*full["joint"]),
        fmt(*mix["joint"]),
        fmt(*dil["joint"]),
    )
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_main_clean.tex").write_text(t)

    for k in C:
        vs, c = cell[k]
        print(
            f"CIL-CORD {k:9s} {'clean ' if c else 'legacy'} n={len(vs)} {np.round(vs, 1).tolist()}"
        )
    print("full-label:", {k: np.round(x[0], 1).tolist() for k, x in full.items()})
    print(re.sub(r"\n", " | ", (OUT / "tab_2x2.tex").read_text())[:600])


if __name__ == "__main__":
    main()
