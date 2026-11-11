"""Generate every number of the CVPR rewrite from result files (no hand-typed cells).

Writes into paper_cvpr/:
  vals.tex            \\newcommand macros for the headline cells (used by abstract, intro, teaser)
  tab_2x2.tex         memory x readout on CIL-CORD (+ isolation row)
  tab_main_clean.tex  comparison table (CIL-CORD AA, CIL-CORD full-label F1, Mixed AA, DIL AA)
  tab_dil_grid.tex    DIL AA of the classical methods on four backbones (legacy grid)

Each cell lists its sources: the CLEAN run (results/clean_2026-11: fp32, held-out early
stopping) first, then legacy fallbacks. A cell uses the clean source ONLY when all three clean
seeds exist; otherwise it uses the first non-empty legacy source, marked with a dagger (a clean
source with one or two seeds is ignored, so the two protocols never mix in one cell). n < 3
prints per-seed values. Full-label F1 has no legacy source: it prints the clean seeds that exist,
with n marked when n < 3, else "--". A collapse mark (c) appears only on a cell from which an
exact-zero legacy seed was actually dropped.

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


def choose(got):
    """The one selection rule (shared with make_retention_fig.py). ``got`` = [(values, is_clean)]
    in source order -> index of the source to use, or None. A clean source is used only with all
    three seeds; otherwise the first non-empty legacy source."""
    for i, (vs, c) in enumerate(got):
        if c and len(vs) == len(SEEDS):
            return i
    for i, (vs, c) in enumerate(got):
        if not c and len(vs):
            return i
    return None


def pick(sources, key="AA", drop_zero=False):
    """-> (values, is_clean, dropped). ``drop_zero`` removes exact-zero (fp16-collapsed) seeds
    from LEGACY sources only; ``dropped`` is True only if a value was actually removed."""
    got, drops = [], []
    for r, n, sfx in sources:
        vs, c = load(r, n, key, sfx), r == CLEAN
        kept = [v for v in vs if v > 0.5] if drop_zero and not c else vs
        got.append((kept, c))
        drops.append(len(kept) < len(vs))
    i = choose(got)
    return (got[i][0], got[i][1], drops[i]) if i is not None else ([], False, False)


def pick_clean_only(sources, key):
    """For a metric only the clean protocol writes (full-label F1): the clean seeds that exist."""
    vs = [v for r, n, sfx in sources if r == CLEAN for v in load(r, n, key, sfx)]
    return vs, True, False


def fmt(vs, clean, dropped=False, bold=False, mark="", show_n=False):
    if not vs:
        return "--"
    if len(vs) >= 3:
        body = f"{np.mean(vs):.1f}{{\\pm}}{np.std(vs, ddof=1):.1f}"
    else:
        body = "/".join(f"{v:.1f}" for v in vs)
    if bold:
        body = f"\\mathbf{{{body}}}"
    marks = ("\\dagger" if not clean else "", "c" if dropped else "", mark,
             f"n={len(vs)}" if show_n and len(vs) < 3 else "")  # fmt: skip
    sup = ",".join(x for x in marks if x)
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
    ("IS3 (2024)", "is3", False),  # never mixed precision: its zeros are genuine
    ("LCA (2026)", "lca", False),
]


def macro(name, vs):
    """Two macros per headline cell: \\v<Name> (mean, or mean±sd at n=3) and \\n<Name> (mean)."""
    if not vs:
        return f"\\newcommand{{\\v{name}}}{{--}}\\newcommand{{\\n{name}}}{{--}}\n"
    pm = (
        f"{np.mean(vs):.1f}{{\\pm}}{np.std(vs, ddof=1):.1f}"
        if len(vs) >= 3
        else f"{np.mean(vs):.1f}"
    )
    return f"\\newcommand{{\\v{name}}}{{{pm}}}\\newcommand{{\\n{name}}}{{{np.mean(vs):.1f}}}\n"


DIL_METHODS = (("Naive", "naive"), ("EWC", "ewc"), ("LwF", "lwf"), ("ER", "er"),
               ("DER++", "der_pp"), ("Joint", "joint"))  # fmt: skip
DIL_BACKBONES = (("LayoutLMv3", ""), ("LiLT", "_lilt"), ("BROS", "_bros"), ("BERT", "_bert"))


def write_dil_grid():
    """Table 1: DIL AA, legacy grid ``results/dil_<method>_seed<s><backbone suffix>`` (all cells
    legacy protocol, stated once in the caption, so no per-cell dagger). A cell with no run is --."""
    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{l" + "c" * len(DIL_METHODS) + "}\n\\toprule\n"
    t += "Backbone & " + " & ".join(m for m, _ in DIL_METHODS) + " \\\\\n\\midrule\n"
    for bname, sfx in DIL_BACKBONES:
        cells = [fmt(load("results", f"dil_{key}", "AA", sfx), True) for _, key in DIL_METHODS]
        t += f"{bname} & " + " & ".join(cells) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_dil_grid.tex").write_text(t)


def main():
    cell = {k: pick(v) for k, v in C.items()}
    mix = {k: pick(v) for k, v in M.items()}
    dil = {k: pick(v) for k, v in D.items()}
    full = {
        k: pick_clean_only(C[k], "full_label_f1")
        for k in ("naive", "colar", "sbr", "recall", "joint", "er")
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
    t += f"Per-session low-rank adapters, no memory & {fmt(*cell['cllora'])} & {fmt(*cell['recall'], bold=True)} \\\\\n"
    t += f"\\midrule\nJoint training (reference) & \\multicolumn{{2}}{{c}}{{{fmt(*cell['joint'])}}} \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_2x2.tex").write_text(t)

    # ---- tab_main_clean.tex
    def fl(k, **kw):  # full-label F1 cell: clean seeds only, n marked when < 3
        return fmt(*full[k], show_n=True, **kw)

    def row(label, c, f, m, d, bold=False):
        lab = f"\\textbf{{{label}}}" if bold else label
        return f"{lab} & {c} & {f} & {m} & {d} \\\\\n"

    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{lcccc}\n\\toprule\n"
    t += " & \\multicolumn{2}{c}{\\textbf{CIL-CORD}} & \\textbf{Mixed} & \\textbf{DIL} \\\\\n"
    t += "\\textbf{Method} & AA & full-label F1 & AA & AA \\\\\n\\midrule\n"
    t += row(
        "Naive", fmt(*cell["naive"]), fl("naive"), fmt(*mix["naive"]), fmt(*dil["naive"])
    )
    t += row(
        "ER", fmt(*cell["er"]), fl("er"), fmt(*mix["er"]), fmt(*dil["er"])
    )
    for label, key, dz in BASE:
        # Clean source first (same names the queue writes), so this row and the 2x2 agree.
        cc = pick(src(f"cil_cord_{key}", (RESC, f"cil_cord_{key}", "")), drop_zero=dz)
        mm = pick(src(f"mixed_{key}", (RESC, f"mixed_{key}", "")), drop_zero=dz)
        dd = pick(src(f"dil_{key}", ("results", f"dil_{key}", "")))
        ff = pick_clean_only(src(f"cil_cord_{key}"), "full_label_f1")
        t += row(label, fmt(*cc), fmt(*ff, show_n=True), fmt(*mm), fmt(*dd))
    t += "\\midrule\n"
    t += row(
        "CoLaR", fmt(*cell["colar"]), fl("colar"), fmt(*mix["colar"]), fmt(*dil["colar"])
    )
    t += row(
        "CoLaR-SBR",
        fmt(*cell["sbr"]),
        fl("sbr"),
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
        fl("recall"),
        fmt(*mix["recall"], mark="b"),
        "--",
        bold=True,
    )
    t += "\\midrule\n"
    t += row(
        "Joint training (reference)",
        fmt(*cell["joint"]),
        fl("joint"),
        fmt(*mix["joint"]),
        fmt(*dil["joint"]),
    )
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_main_clean.tex").write_text(t)

    write_dil_grid()

    for k in C:
        vs, c, _ = cell[k]
        print(
            f"CIL-CORD {k:9s} {'clean ' if c else 'legacy'} n={len(vs)} {np.round(vs, 1).tolist()}"
        )
    print("full-label:", {k: np.round(x[0], 1).tolist() for k, x in full.items()})
    print(re.sub(r"\n", " | ", (OUT / "tab_2x2.tex").read_text())[:600])


if __name__ == "__main__":
    main()
