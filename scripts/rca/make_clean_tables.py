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


def _run_file(root: str, name: str, seed: int, suffix: str, fname: str) -> Path | None:
    hits = sorted(glob(f"{root}/{name}_seed{seed}{suffix}/**/{fname}", recursive=True))
    return Path(hits[0]) if hits else None


def load_seeds(root: str, name: str, key: str = "AA", suffix: str = "*") -> dict[int, float]:
    """{seed: metric ``key``} of run ``<name>_seed<s>`` under ``root`` for every seed that has it."""
    out = {}
    for s in SEEDS:
        f = _run_file(root, name, s, suffix, "metrics.json")
        v = json.loads(f.read_text()).get(key) if f else None
        if v is not None:
            out[s] = float(v)
    return out


def load(root: str, name: str, key: str = "AA", suffix: str = "*") -> list[float]:
    """Metric ``key`` of run ``<name>_seed<s>`` under ``root`` for every seed that exists."""
    return list(load_seeds(root, name, key, suffix).values())


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


def _candidates(sources, key, drop_zero):
    got, drops = [], []
    for r, n, sfx in sources:
        vs, c = load(r, n, key, sfx), r == CLEAN
        kept = [v for v in vs if v != 0.0] if drop_zero and not c else vs
        got.append((kept, c))
        drops.append(len(kept) < len(vs))
    return got, drops


def pick(sources, key="AA", drop_zero=False):
    """-> (values, is_clean, dropped). ``drop_zero`` removes exact-zero (fp16-collapsed) seeds
    from LEGACY sources only; ``dropped`` is True only if a value was actually removed."""
    got, drops = _candidates(sources, key, drop_zero)
    i = choose(got)
    return (got[i][0], got[i][1], drops[i]) if i is not None else ([], False, False)


def chosen_source(sources, key="AA", drop_zero=False):
    """The (root, name, suffix) whose seeds ``pick`` reports for this cell, or None."""
    i = choose(_candidates(sources, key, drop_zero)[0])
    return sources[i] if i is not None else None


def matrix_choice(sources) -> tuple[tuple | None, dict[int, np.ndarray]]:
    """Retention matrices {seed: R} chosen by the tables' rule (``choose``): the clean source
    only with all three seeds, otherwise the first non-empty legacy source. -> (source, mats)."""
    per_source = []
    for root, name, sfx in sources:
        mats = {}
        for s in SEEDS:
            f = _run_file(root, name, s, sfx, "matrix.npy")
            if f:
                mats[s] = np.load(f)
        per_source.append((mats, root == CLEAN))
    i = choose([(list(m.values()), c) for m, c in per_source])
    return (sources[i], per_source[i][0]) if i is not None else (None, {})


def matrices_by_seed(sources) -> dict[int, np.ndarray]:
    return matrix_choice(sources)[1]


def matrices(sources) -> list[np.ndarray]:
    return list(matrices_by_seed(sources).values())


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
    legacy protocol, stated once in the caption, so no per-cell dagger). A cell with no run is --.
    """
    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{l" + "c" * len(DIL_METHODS) + "}\n\\toprule\n"
    t += "Backbone & " + " & ".join(m for m, _ in DIL_METHODS) + " \\\\\n\\midrule\n"
    for bname, sfx in DIL_BACKBONES:
        cells = [fmt(load("results", f"dil_{key}", "AA", sfx), True) for _, key in DIL_METHODS]
        t += f"{bname} & " + " & ".join(cells) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_dil_grid.tex").write_text(t)


# ---------------------------------------------------------------------------- supplement
HDR = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
ROOT_NAME = {CLEAN: "clean", GATE: "gate", RESC: "rescore", "results": "grid"}


def protocol_note(src_tuple, seeds) -> str:
    """'seeds 42/7, rescore$^\\dagger$' for the source a cell or matrix was read from."""
    if src_tuple is None or not seeds:
        return "no run"
    root = src_tuple[0]
    tag = ROOT_NAME.get(root, root) + ("" if root == CLEAN else "$^{\\dagger}$")
    return f"seed{'s' if len(seeds) > 1 else ''} {'/'.join(map(str, seeds))}, {tag}"


def write_matrix(fname, title, sources):
    """Mean retention matrix R[i,j] (rows: after session i; columns: session j) of the source
    the tables use, with its seeds and protocol in the title row."""
    source, by_seed = matrix_choice(sources)
    note = protocol_note(source, list(by_seed))
    t = HDR
    if not by_seed:
        (OUT / fname).write_text(t + f"\\textbf{{{title}}}: no run\n")
        return
    mean = np.mean(list(by_seed.values()), axis=0)
    n = mean.shape[0]
    t += f"\\begin{{tabular}}{{l*{{{n}}}{{c}}}}\n"
    t += f"\\multicolumn{{{n + 1}}}{{l}}{{\\textbf{{{title}}} ({note})}} \\\\\n\\toprule\n"
    t += "$i\\backslash j$ & " + " & ".join(str(j + 1) for j in range(n)) + " \\\\\n\\midrule\n"
    for r in range(n):
        cells = [
            f"{mean[r, j]:.1f}" if j <= r and np.isfinite(mean[r, j]) else "" for j in range(n)
        ]
        t += f"{r + 1} & " + " & ".join(cells) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / fname).write_text(t)


def _h(sources_one, seeds):
    """Mean training-loop hours per run (``total_wall_time_s`` of metrics.json)."""
    r, n, sfx = sources_one
    ts = [v for s, v in load_seeds(r, n, "total_wall_time_s", sfx).items() if s in seeds]
    return f"{np.mean(ts) / 3600:.1f}" if ts else "--"


def _tag(s) -> str:
    return ROOT_NAME.get(s[0], s[0]) + ("" if s[0] == CLEAN else "$^{\\dagger}$")


def _seed_row(label, sources, drop_zero=False):
    """label & source & s42 & s7 & s123 & h/run (the seeds ``pick`` reports). No precision
    column: ``training_hparams.amp`` records the config flag, not whether the loop honoured it
    (seed-42 baselines predate AMP wiring; IS3 never autocasts), see tab:legacyprotocol."""
    s = chosen_source(sources, drop_zero=drop_zero)
    if s is None:
        return f"{label} & -- & -- & -- & -- & -- \\\\\n"
    vals = load_seeds(s[0], s[1], suffix=s[2])
    cells = []
    for sd in SEEDS:
        if sd not in vals:
            cells.append("--")
        elif drop_zero and s[0] != CLEAN and vals[sd] == 0.0:
            cells.append(f"${vals[sd]:.1f}^{{c}}$")
        else:
            cells.append(f"{vals[sd]:.1f}")
    return f"{label} & {_tag(s)} & " + " & ".join(cells) + f" & {_h(s, vals)} \\\\\n"


SEED_HDR = "Row & Source & 42 & 7 & 123 & h/run \\\\\n"


ROW_LABEL = {"naive": "Naive", "nobank": "SBR, no memory", "colar": "CoLaR",
             "sbr": "CoLaR-SBR", "cllora": "CL-LoRA", "recall": "SBR-Recall",
             "joint": "Joint training", "er": "ER", "slot": "CoLaR-SBR + LexSlot"}  # fmt: skip


def write_perseed():
    """Every headline cell's per-seed values, from the source the main tables use."""
    t = HDR + "\\begin{tabular}{llcccc}\n\\toprule\n" + SEED_HDR
    for scen, reg, pre in (("CIL-CORD", C, "cil_cord"), ("Mixed", M, "mixed"), ("DIL", D, "dil")):
        t += f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{{scen}}}}} \\\\\n"
        for k, sources in reg.items():
            if k == "cllora" or (k == "sbr_bros" and not chosen_source(sources)):
                continue  # CL-LoRA is listed once, with the baselines
            t += _seed_row(ROW_LABEL.get(k, k), sources)
        for label, key, dz in BASE:
            root = "results" if pre == "dil" else RESC
            t += _seed_row(label, src(f"{pre}_{key}", (root, f"{pre}_{key}", "")), dz)
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_perseed.tex").write_text(t)


def write_agree():
    """Session-restricted AA next to full-label F1 for every clean run that records both."""
    rows = []
    for f in sorted(glob(f"{CLEAN}/*/metrics.json")):
        m = json.loads(Path(f).read_text())
        if "smoke" in f or m.get("full_label_f1") is None:
            continue
        rows.append((m["scenario"], m["method"], m["seed"], m["AA"], m["full_label_f1"]))
    t = HDR + "\\begin{tabular}{llcccc}\n\\toprule\n"
    t += "Scenario & Method & Seed & AA & Full-label F1 & Difference \\\\\n\\midrule\n"
    for sc, me, sd, aa, fl in rows:
        lab = ROW_LABEL.get(me.replace("colar_sbr", "sbr"), me).replace("_", "\\_")
        t += f"{sc.replace('_', '-').upper()} & {lab} & {sd} & {aa:.1f} & {fl:.1f} & {fl - aa:+.1f} \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_agree.tex").write_text(t)


def write_slot():
    """CoLaR-SBR with and without LexSlot head slots, per scenario and seed."""
    t = HDR + "\\begin{tabular}{llccc}\n\\toprule\n"
    t += "Scenario & CoLaR-SBR source & 42 & 7 & 123 \\\\\n\\midrule\n"
    pairs = (("CIL-CORD", C["sbr"]), ("Mixed", M["sbr"]),
             ("Mixed", src(None, (GATE, "mixed_colar_sbr2"))), ("DIL", D["sbr"]))  # fmt: skip
    slots = {"CIL-CORD": C["slot"], "Mixed": M["slot"], "DIL": D["slot"]}
    for scen, sbr_src in pairs:
        s, ls = chosen_source(sbr_src), chosen_source(slots[scen])
        a = load_seeds(s[0], s[1], suffix=s[2]) if s else {}
        b = load_seeds(ls[0], ls[1], suffix=ls[2]) if ls else {}
        tag = _tag(s) if s else "--"
        cell = []
        for sd in SEEDS:
            if sd in a and sd in b:
                cell.append(f"{a[sd]:.1f} / {b[sd]:.1f} ({b[sd] - a[sd]:+.1f})")
            elif sd in b:
                cell.append(f"-- / {b[sd]:.1f}")
            else:
                cell.append("--")
        t += f"{scen} & {tag} & " + " & ".join(cell) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_slot.tex").write_text(t)


CONFLICT_ROWS = (  # CIL-CORD, session-restricted: what separates readout from memory
    ("CoLaR, shared softmax", C["colar"]),
    ("CoLaR, shared softmax (gate run)", src(None, (GATE, "cil_cord_colar", "_k4_d50_r128"))),
    ("Replay loss sliced to classes alive at storage", src(None, (RESC, "cil_cord_colar_r_sliced", ""))),
    ("Replay loss sliced (gate run)", src(None, (GATE, "cil_cord_colar_r_sliced", "_k4_d50_r128"))),
    ("Replay and current losses sliced", src(None, (GATE, "cil_cord_colar_r_sliced2", "_k4_d50_r128"))),
    ("\\quad + model pseudo-labels written into the memory",
     src(None, (GATE, "cil_cord_colar_r_sliced2_pseudo", "_k4_d50_r128"))),
    ("\\quad + pseudo-labels in both directions",
     src(None, (GATE, "cil_cord_colar_r_bilc", "_k4_d50_r128"))),
    ("True-label memory refresh (not deployable)",
     src(None, (GATE, "cil_cord_colar_r_oracle", "_k4_d50_r128"))),
    ("SBR, no memory", C["nobank"]),
    ("SBR, no memory, layers 1..11 frozen", src(None, (GATE, "cil_cord_colar_sbr_frozen"))),
    ("CoLaR-SBR", C["sbr"]),
    ("Joint training (reference)", C["joint"]),
)  # fmt: skip


def write_conflict():
    t = HDR + "\\begin{tabular}{llcccc}\n\\toprule\n" + SEED_HDR + "\\midrule\n"
    for label, sources in CONFLICT_ROWS:
        t += _seed_row(label, sources)
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_conflict.tex").write_text(t)


DELTA_RUNS = (  # (label, run name, suffix): SBR-Recall delta capacity on CIL-CORD
    ("rank 8, Q/K/V, memory", "cil_cord_colar_sbr_recall", ""),
    ("rank 8, Q/K/V, no memory", "cil_cord_colar_sbr_recall_d0", ""),
    ("rank 64, Q/K/V + dense, no memory", "cil_cord_colar_sbr_recall_r64", ""),
)


def write_delta():
    """Per-session diagonal R[j,j] and final row R[T,j] of every SBR-Recall capacity run."""
    t = HDR + "\\begin{tabular}{lc*{5}{c}c*{5}{c}c}\n\\toprule\n"
    t += " & & \\multicolumn{5}{c}{Diagonal $R[j,j]$} & & \\multicolumn{5}{c}{Final row $R[5,j]$} & \\\\\n"
    t += "Delta & Seed & 1 & 2 & 3 & 4 & 5 & & 1 & 2 & 3 & 4 & 5 & AA \\\\\n\\midrule\n"
    for label, name, sfx in DELTA_RUNS:
        for sd, mat in matrices_by_seed([(GATE, name, sfx)]).items():
            dg = " & ".join(f"{mat[j, j]:.1f}" for j in range(mat.shape[0]))
            fr = " & ".join(f"{v:.1f}" for v in mat[-1])
            t += f"{label} & {sd} & {dg} & & {fr} & {np.mean(mat[-1]):.1f} \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_delta.tex").write_text(t)


def write_supp():
    write_matrix("supp_mat_colar.tex", "CoLaR, shared softmax, CIL-CORD", C["colar"])
    write_matrix("supp_mat_sbr.tex", "CoLaR-SBR, CIL-CORD", C["sbr"])
    write_matrix("supp_mat_recall.tex", "SBR-Recall, CIL-CORD", C["recall"])
    write_matrix("supp_mat_sbr_mix.tex", "CoLaR-SBR, Mixed", M["sbr"])
    write_matrix("supp_mat_recall_mix_nomem.tex", "SBR-Recall, Mixed, no memory",
                 src(None, (GATE, "mixed_colar_sbr_recall_r64b")))  # fmt: skip
    write_matrix("supp_mat_recall_mix_mem.tex", "SBR-Recall, Mixed, 50-document memory",
                 M["recall"])  # fmt: skip
    write_perseed()
    write_conflict()
    write_agree()
    write_slot()
    write_delta()


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
    t += f"Low-rank adapters, no memory & {fmt(*cell['cllora'])} & {fmt(*cell['recall'], bold=True)} \\\\\n"
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
    t += row("Naive", fmt(*cell["naive"]), fl("naive"), fmt(*mix["naive"]), fmt(*dil["naive"]))
    t += row("ER", fmt(*cell["er"]), fl("er"), fmt(*mix["er"]), fmt(*dil["er"]))
    for label, key, dz in BASE:
        # Clean source first (same names the queue writes), so this row and the 2x2 agree.
        cc = pick(src(f"cil_cord_{key}", (RESC, f"cil_cord_{key}", "")), drop_zero=dz)
        mm = pick(src(f"mixed_{key}", (RESC, f"mixed_{key}", "")), drop_zero=dz)
        dd = pick(src(f"dil_{key}", ("results", f"dil_{key}", "")))
        ff = pick_clean_only(src(f"cil_cord_{key}"), "full_label_f1")
        t += row(label, fmt(*cc), fmt(*ff, show_n=True), fmt(*mm), fmt(*dd))
    t += "\\midrule\n"
    t += row("CoLaR", fmt(*cell["colar"]), fl("colar"), fmt(*mix["colar"]), fmt(*dil["colar"]))
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
    write_supp()

    for k in C:
        vs, c, _ = cell[k]
        print(
            f"CIL-CORD {k:9s} {'clean ' if c else 'legacy'} n={len(vs)} {np.round(vs, 1).tolist()}"
        )
    print("full-label:", {k: np.round(x[0], 1).tolist() for k, x in full.items()})
    print(re.sub(r"\n", " | ", (OUT / "tab_2x2.tex").read_text())[:600])


if __name__ == "__main__":
    main()
