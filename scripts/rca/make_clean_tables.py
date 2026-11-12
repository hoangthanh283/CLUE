"""Generate every number of the CVPR rewrite from result files (no hand-typed cells).

Writes into paper_cvpr/:
  vals.tex            \\newcommand macros for the headline cells (used by abstract, intro, teaser)
  tab_2x2.tex         memory x readout on CIL-CORD (strict 2x2; AA and full-label F1)
  tab_main_clean.tex  comparison table (CIL-CORD AA, CIL-CORD full-label F1, Mixed AA, DIL AA)
  tab_dil_grid.tex    DIL AA of the classical methods on four backbones (legacy grid)
  tab_ladder.tex      the hypothesis ladder H0-H4 (slot rows render only once their runs exist)
  tab_mech.tex        where old-session errors go (from results/clean_2026-11/mechanism_analysis.json)
  tab_beyond.tex      the whole "beyond one stream" float, or an empty file while no row exists

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
import math
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


def up1(x: float) -> str:
    """Round an upper bound UP to one decimal, so "no more than / within X" stays true."""
    return f"{math.ceil(10 * x - 1e-9) / 10:.1f}"


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
             "joint": "Joint training", "er": "ER", "slot": "CoLaR-SBR + LexSlot",
             "sbr_recall_r64": "SBR-Recall", "sbr_noreplay": "SBR, no memory",
             "sbr_bros": "CoLaR-SBR (BROS)",
             # queued run families: labelled now so write_agree never drops them silently
             "colar_r_oracle": "Older stored documents relabelled at each boundary (not deployable)",
             "colar_r_oracle_pre": "True-label refresh before each session (not deployable)",
             "colar_mib": "CoLaR + MiB-style unbiased CE (shared softmax)",
             "er_sbr": "Raw-document memory + SBR", "cl_lora": "CL-LoRA",
             "er250": "ER (250 documents)",
             "sbr_const_o": "CoLaR-SBR, constant background logit", "ice_o": "ICE-O (our port)"}  # fmt: skip


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
    """Session-restricted AA next to full-label F1 for every clean CIL-CORD run that records both
    (any backbone: the row label names a non-LayoutLMv3 one). Runs of other scenarios (e.g.
    cil_funsd) stay out; an alias without a ROW_LABEL is printed under its raw alias."""
    rows = []
    for f in sorted(glob(f"{CLEAN}/cil_cord_*/metrics.json")):
        m = json.loads(Path(f).read_text())
        if "smoke" in f or m.get("full_label_f1") is None or m.get("scenario") != "cil_cord":
            continue
        if "_dump" in f or str(m.get("method", "")).endswith("_dump"):
            continue  # seed-42 prediction-dump re-runs: mechanism inputs, never table cells
        rows.append((m["scenario"], m["method"], m["seed"], m["AA"], m["full_label_f1"]))
    rows.sort(key=lambda r: (r[0], r[1], SEEDS.index(r[2]) if r[2] in SEEDS else len(SEEDS)))
    t = HDR + "\\begin{tabular}{llcccc}\n\\toprule\n"
    t += "Scenario & Method & Seed & AA & Full-label F1 & Difference \\\\\n\\midrule\n"
    for sc, me, sd, aa, fl in rows:
        lab = ROW_LABEL.get(me.replace("colar_sbr", "sbr"), me).replace("_", "\\_")
        t += f"{sc.replace('_', '-').upper()} & {lab} & {sd} & {aa:.1f} & {fl:.1f} & ${fl - aa:+.1f}$ \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_agree.tex").write_text(t)


# LexSlot exists only as fp32 gate runs, so the CoLaR-SBR side is read from the SAME gate protocol
# (never from the main-table source, which switches to clean when its three clean seeds land).
SLOT_PAIRS = (
    ("CIL-CORD", (GATE, "cil_cord_colar_sbr", ""), (GATE, "cil_cord_colar_sbr_slot", "*")),
    ("Mixed", (GATE, "mixed_colar_sbr2", "*"), (GATE, "mixed_colar_sbr_slot", "*")),
    ("DIL", (GATE, "dil_colar_sbr2", "*"), (GATE, "dil_colar_sbr_slot", "*")),
)


def write_slot():
    """CoLaR-SBR with and without LexSlot head slots, per scenario and seed, both sides from
    the fp32 gate runs (one protocol per row)."""
    t = HDR + "\\begin{tabular}{llccc}\n\\toprule\n"
    t += "Scenario & Protocol & 42 & 7 & 123 \\\\\n\\midrule\n"
    for scen, s, ls in SLOT_PAIRS:
        a, b = load_seeds(s[0], s[1], suffix=s[2]), load_seeds(ls[0], ls[1], suffix=ls[2])
        cell = []
        for sd in SEEDS:
            if sd in a and sd in b:
                cell.append(f"{a[sd]:.1f} / {b[sd]:.1f} (${b[sd] - a[sd]:+.1f}$)")
            elif sd in b:
                cell.append(f"-- / {b[sd]:.1f}")
            else:
                cell.append("--")
        t += f"{scen} & {_tag(s)} & " + " & ".join(cell) + " \\\\\n"
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
    ("Older stored documents relabelled at each boundary (not deployable)", None),  # clean seeds as they land
    ("Older stored documents relabelled (gate run)",
     src(None, (GATE, "cil_cord_colar_r_oracle", "_k4_d50_r128"))),
    ("SBR, no memory", C["nobank"]),
    ("SBR, no memory, layers 1..11 frozen", src(None, (GATE, "cil_cord_colar_sbr_frozen"))),
    ("CoLaR-SBR", C["sbr"]),
    ("Joint training (reference)", C["joint"]),
)  # fmt: skip


def write_conflict():
    t = HDR + "\\begin{tabular}{llcccc}\n\\toprule\n" + SEED_HDR + "\\midrule\n"
    for label, sources in CONFLICT_ROWS:
        if sources is None:  # per-seed listing of the clean oracle runs (any n; never a mean)
            vals = load_seeds(*ORACLE_CLEAN[:2], suffix=ORACLE_CLEAN[2])
            cells = " & ".join(f"{vals[s]:.1f}" if s in vals else "--" for s in SEEDS)
            t += f"{label} & clean & {cells} & {_h(ORACLE_CLEAN, vals)} \\\\\n"
            continue
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
            t += f"{label} & {sd}$^{{\\dagger}}$ & {dg} & & {fr} & {np.mean(mat[-1]):.1f} \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_delta.tex").write_text(t)


DROPPED = (("EWC", "ewc"), ("L2P", "l2p"), ("DualPrompt", "dualprompt"), ("O-LoRA", "o_lora"),
           ("ER + C-Flat++", "er_cflat"))  # fmt: skip
DROPPED_REF = (("Naive (reference)", "naive"), ("ER (kept)", "er"))


def write_dropped():
    """R9-H2: the first CIL-CORD grid (results/cil_cord_<m>_seed<s>, LayoutLMv3, legacy scoring)
    for the methods dropped when the comparison set was fixed, per seed, beside naive and ER of
    the same grid. The text says every dropped seed is below every naive seed: checked here."""
    t = HDR + "\\begin{tabular}{lccc}\n\\toprule\nMethod & 42 & 7 & 123 \\\\\n\\midrule\n"
    vals = {}
    for label, key in DROPPED_REF + DROPPED:
        vals[key] = load_seeds("results", f"cil_cord_{key}", suffix="")
        t += f"{label} & " + " & ".join(
            f"{vals[key][s]:.1f}" if s in vals[key] else "--" for s in SEEDS) + " \\\\\n"  # fmt: skip
        if key == "er":
            t += "\\midrule\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "supp_dropped.tex").write_text(t)
    drop = [x for _, k in DROPPED for x in vals[k].values()]
    if not vals["naive"] or not drop or max(drop) >= min(vals["naive"].values()):
        _warn("a dropped method reaches a naive seed on the first CIL-CORD grid (H2 disclosure)")


def write_supp_vals():
    """Macros that describe the mean retention matrices of tab:matrices, so the appendix prose
    follows the matrices when their source switches protocol (supp_vals.tex)."""

    def mean(sources):
        mats = list(matrices_by_seed(sources).values())
        return np.mean(mats, axis=0) if mats else None

    def m(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"

    t = HDR
    co, sb, rc = mean(C["colar"]), mean(C["sbr"]), mean(C["recall"])
    if co is not None:
        n = co.shape[0]
        rises = [co[i + 1, j] - co[i, j] for j in range(n) for i in range(j, n - 1)]
        t += m("colarDropOne", f"{co[0, 0] - co[1, 0]:.0f}") + m(
            "colarDropTwo", f"{co[1, 1] - co[2, 1]:.0f}"
        )
        t += m("colarLateDiagMin", f"{min(np.diag(co)[2:]):.0f}")
        t += m("colarLateDiagMax", f"{max(np.diag(co)[2:]):.0f}")
        t += m("colarMaxRise", f"{max(rises):.0f}")
    if sb is not None:
        loss = np.diag(sb)[:-1] - sb[-1, :-1]
        t += m("sbrLossMin", f"{loss.min():.0f}") + m("sbrLossMax", f"{loss.max():.0f}")
        t += m("sbrLossMaxSession", str(int(loss.argmax()) + 1))
    if rc is not None:
        t += m("recallDevMax", up1(np.abs(np.diag(rc) - rc[-1]).max()))
    nb = matrices(C["nobank"])
    if nb:  # final F1 of every old session of no-memory SBR, any seed (an upper bound)
        t += m("nobankOldMax", f"{math.ceil(max(mt[-1, :-1].max() for mt in nb)):.0f}")
    (OUT / "supp_vals.tex").write_text(t)


def write_supp():
    write_matrix("supp_mat_colar.tex", "CoLaR, shared softmax, CIL-CORD", C["colar"])
    write_matrix("supp_mat_sbr.tex", "CoLaR-SBR, CIL-CORD", C["sbr"])
    write_matrix("supp_mat_recall.tex", "SBR-Recall, CIL-CORD", C["recall"])
    write_matrix("supp_mat_sbr_mix.tex", "CoLaR-SBR, Mixed", M["sbr"])
    write_matrix("supp_mat_recall_mix_nomem.tex", "SBR-Recall, Mixed, no memory",
                 src(None, (GATE, "mixed_colar_sbr_recall_r64b")))  # fmt: skip
    write_matrix("supp_mat_recall_mix_mem.tex", "SBR-Recall, Mixed, 50-document memory",
                 M["recall"])  # fmt: skip
    write_supp_vals()
    write_perseed()
    write_conflict()
    write_agree()
    write_slot()
    write_delta()
    write_dropped()


def _clean(k, key="AA"):
    """{seed: value} of the CLEAN source of headline cell ``k`` (its first source)."""
    r, n, sfx = C[k][0]
    return load_seeds(r, n, key, sfx) if r == CLEAN else {}


def down1(x: float) -> str:
    """Round a lower bound DOWN to one decimal, so "at least X" stays true."""
    return f"{math.floor(10 * x + 1e-9) / 10:.1f}"


def _warn(msg: str) -> None:
    print(f"WARNING (text claim no longer holds, edit the prose): {msg}")


def _defs(vals: dict) -> str:
    return "".join(f"\\newcommand{{\\{k}}}{{{x}}}\n" for k, x in vals.items())


def recall_macros() -> str:
    """Per-seed facts about the SBR-Recall CIL-CORD cell that the text states (rule 7):
    \\nRecallJointGap = max over seeds AND over both metrics (AA, full-label F1) of
    |Recall - joint| (seed-matched clean runs); \\nRecallMaxLoss / \\nRecallMaxGain = largest
    fall / rise of any session below / above its diagonal at any later boundary, on any seed of
    the chosen source."""
    gaps = []
    for key in ("AA", "full_label_f1"):
        ra, rb = _clean("recall", key), _clean("joint", key)
        gaps += [abs(ra[s] - rb[s]) for s in SEEDS if s in ra and s in rb]
    moves = [
        mat[i, j] - mat[j, j]
        for mat in matrices(C["recall"])
        for j in range(mat.shape[0])
        for i in range(j, mat.shape[0])
    ]
    return _defs({  # "--" when the runs are missing, so the paper still builds
        "nRecallJointGap": up1(max(gaps)) if gaps else "--",
        "nRecallMaxLoss": up1(max(0.0, -min(moves))) if moves else "--",
        "nRecallMaxGain": up1(max(moves)) if moves else "--",
    })  # fmt: skip


ORACLE_CLEAN = (CLEAN, "cil_cord_colar_r_oracle", "*")
ORACLE_GATE = (GATE, "cil_cord_colar_r_oracle", "_k4_d50_r128")
LABEL_REPAIRS = (  # label-side repairs of the shared softmax (rows of tab:conflict)
    (RESC, "cil_cord_colar_r_sliced", ""), (GATE, "cil_cord_colar_r_sliced", "_k4_d50_r128"),
    (GATE, "cil_cord_colar_r_sliced2", "_k4_d50_r128"),
    (GATE, "cil_cord_colar_r_sliced2_pseudo", "_k4_d50_r128"),
    (GATE, "cil_cord_colar_r_bilc", "_k4_d50_r128"), ORACLE_GATE, ORACLE_CLEAN,
)  # fmt: skip
_AVG_ROWS = ("micro avg", "macro avg", "weighted avg")


def _per_seed(d: dict, seeds, fmt_one="{:.1f}") -> str:
    """Rule 6 for a macro: the mean when all three seeds exist, else the per-seed values."""
    vs = [d[s] for s in seeds]
    if not vs:
        return "--"
    return (
        fmt_one.format(np.mean(vs)) if len(vs) == len(SEEDS) else "/".join(map(fmt_one.format, vs))
    )


def _span(vs, fmt_one="{:.0f}") -> str:
    lo, hi = fmt_one.format(min(vs)), fmt_one.format(max(vs))
    return lo if lo == hi else f"{lo} to {hi}"


def oracle_macros() -> str:
    """The true-label memory refresh (clean runs only; rule 6: a mean only once all three
    seeds exist, else the per-seed values with \\nOracleScope naming the seeds), seed-matched
    against clean CoLaR (shared softmax), joint training and CoLaR-SBR, under both metrics."""
    o, of = load_seeds(*ORACLE_CLEAN[:2], suffix="*"), load_seeds(
        *ORACLE_CLEAN[:2], key="full_label_f1", suffix="*"
    )
    seeds = [s for s in SEEDS if s in o and s in of]
    co, cof = _clean("colar"), _clean("colar", "full_label_f1")
    jo, jof = _clean("joint"), _clean("joint", "full_label_f1")
    sb, sbf = _clean("sbr"), _clean("sbr", "full_label_f1")
    seeds = [s for s in seeds if all(s in d for d in (co, cof, jo, jof, sb, sbf))]
    n = len(seeds)
    v = {"nOracleScope": "three seeds" if n == 3 else (
        f"seed{'s' if n > 1 else ''} {'/'.join(map(str, seeds))}, {('one', 'two')[n - 1]} seed"
        f"{'s' if n > 1 else ''} so far" if n else "no run")}  # fmt: skip
    for name, d in (("nOracle", o), ("nOracleFull", of), ("nColarSeed", co),
                    ("nColarSeedFull", cof), ("nJointSeed", jo), ("nJointSeedFull", jof),
                    ("nSbrSeed", sb), ("nSbrSeedFull", sbf)):  # fmt: skip
        v[name] = _per_seed(d, seeds)
    for name, a, c, j in (("nOracleShare", o, co, jo), ("nOracleShareFull", of, cof, jof)):
        sh = [100 * (a[s] - c[s]) / (j[s] - c[s]) for s in seeds]
        v[name] = (_span(sh) if n == 3 else "/".join(f"{x:.0f}" for x in sh)) if sh else "--"
    for s in seeds:  # the text: below joint training and below every seed of CoLaR-SBR
        if not (o[s] < min(jo[s], *sb.values()) and of[s] < min(jof[s], *sbf.values())):
            _warn(f"oracle seed {s} is not below joint and CoLaR-SBR on both metrics")
    v["nOracleN"] = ("no seed", "one seed", "two seeds", "three seeds")[n]  # scope in prose
    for s in seeds:  # "partly": above CoLaR and below joint training, same seed, both metrics
        if not (co[s] < o[s] < jo[s] and cof[s] < of[s] < jof[s]):
            _warn(f"oracle seed {s} is not between CoLaR and joint training on both metrics")
    mats = {s: np.load(_run_file(CLEAN, ORACLE_CLEAN[1], s, "*", "matrix.npy")) for s in seeds}
    for k in ("LateMin", "LateMax", "NextMin", "NextMax", "Last", "LastShare"):
        v[f"nOracle{k}"] = "--"  # so the paper builds before any clean oracle run exists
    if mats:
        t = next(iter(mats.values())).shape[0]
        late = [m[j, j] for m in mats.values() for j in range(2, t)]
        nxt = [m[j + 1, j] for m in mats.values() for j in range(2, t - 1)]
        # ranges rounded outward (R8-L3): floor the low end, ceil the high end
        v["nOracleLateMin"], v["nOracleLateMax"] = (
            f"{math.floor(min(late))}",
            f"{math.ceil(max(late))}",
        )
        v["nOracleNextMin"], v["nOracleNextMax"] = (
            f"{math.floor(min(nxt))}",
            f"{math.ceil(max(nxt))}",
        )
        v["nOracleLast"] = _per_seed({s: m[-1, -1] for s, m in mats.items()}, seeds)
        if any(m[j + 1, j] <= m[j, j] for m in mats.values() for j in range(2, t - 1)):
            _warn("an oracle session 3..T-1 does not rise at the next boundary")
        f = _run_file(CLEAN, ORACLE_CLEAN[1], seeds[0], "*", "per_class_f1.json")
        sup = json.loads(f.read_text()) if f else {}
        cnt = [sum(x["support"] for c, x in sup[str(i)].items() if c not in _AVG_ROWS)
               for i in range(t)] if sup else []  # fmt: skip
        v["nOracleLastShare"] = f"{100 * cnt[-1] / sum(cnt):.0f}" if cnt else "--"
    logs = sorted(glob(f"{CLEAN}/{ORACLE_CLEAN[1]}_seed42.log"))
    counts = re.findall(r"refreshed \d+ docs, (\d+) tokens relabelled", Path(logs[0]).read_text()) if logs else []  # fmt: skip
    v["nOracleRelabel"] = "/".join(counts) if counts else "--"  # seed 42
    # R8-H1: the refreshes after sessions 2..T-1 act on training (before sessions 3..T); the
    # refresh after the last session changes no training. The documents of the session just
    # finished are never in the refreshed set (log: "refreshed 50/100/150/200 docs").
    v["nOracleRelabelTrain"] = "/".join(counts[:-1]) if len(counts) > 1 else "--"
    v["nOracleRelabelAfter"] = counts[-1] if counts else "--"
    g = load_seeds(*ORACLE_GATE[:2], suffix=ORACLE_GATE[2]).get(42)
    v["nOracleGate"] = f"{g:.1f}" if g is not None else "--"  # legacy gate run, seed 42
    return _defs(v)


def conflict_macros() -> str:
    """The sliced label-side controls and the best seed of any label-side repair (AA)."""

    def one(r):
        return load_seeds(r[0], r[1], suffix=r[2])

    v = {}
    sl1 = load(RESC, "cil_cord_colar_r_sliced", suffix="")
    sl2 = load(GATE, "cil_cord_colar_r_sliced2", suffix="_k4_d50_r128")
    slg = one((GATE, "cil_cord_colar_r_sliced", "_k4_d50_r128")).get(42)
    v["nSlicedReplay"] = f"{np.mean(sl1):.1f}" if sl1 else "--"  # rescore (fp16)
    v["nSlicedReplayGate"] = f"{slg:.1f}" if slg is not None else "--"  # gate (fp32), seed 42
    v["nSlicedBoth"] = f"{np.mean(sl2):.1f}" if sl2 else "--"  # gate (fp32)
    # R9-M3: the one same-protocol, H2-specific number: seed 42, sliced replay -> + sliced current
    s2g = one((GATE, "cil_cord_colar_r_sliced2", "_k4_d50_r128")).get(42)
    v["nSlicedBothGate"] = f"{s2g:.1f}" if s2g is not None else "--"
    if s2g is not None and slg is not None and s2g <= slg:
        _warn("slicing the current loss no longer adds to the sliced replay (seed 42)")
    allr = [x for r in LABEL_REPAIRS for x in one(r).values()]
    v["nLabelRepairMax"] = up1(max(allr)) if allr else "--"
    if allr and sbr_min(C) <= max(allr):
        _warn("a label-side repair reaches a CoLaR-SBR seed")
    pb = one((GATE, "cil_cord_colar_r_bilc", "_k4_d50_r128")).get(42)
    v["nPseudoBoth"] = f"{pb:.1f}" if pb is not None else "--"  # both sides, seed 42
    return _defs(v)


def sbr_min(reg) -> float:
    vs = pick(reg["sbr"])[0]
    return min(vs) if vs else math.inf


def agree_macros() -> str:
    """AA minus full-label F1 per group of clean runs, and a per-seed check that both metrics
    order the groups the same way: no memory < CoLaR (shared softmax) < true-label refresh <
    CoLaR-SBR / SBR-Recall / joint training (inside a group the order may differ)."""
    groups = (("naive", "nobank"), ("colar",), ("sbr", "recall", "joint"))
    d = {k: (_clean(k), _clean(k, "full_label_f1")) for g in groups for k in g}
    diff = {k: [a[s] - f[s] for s in SEEDS if s in a and s in f] for k, (a, f) in d.items()}
    low = diff["naive"] + diff["nobank"]
    hi = [abs(x) for k in groups[2] for x in diff[k]]
    v = {
        "nAgreeLowMin": down1(min(low)) if low else "--",
        "nAgreeLowMax": up1(max(low)) if low else "--",
        "nAgreeColarMin": down1(min(diff["colar"])) if diff["colar"] else "--",
        "nAgreeColarMax": up1(max(diff["colar"])) if diff["colar"] else "--",
        "nAgreeHighMax": up1(max(hi)) if hi else "--",
    }
    if (low and min(low) <= 0) or (diff["colar"] and min(diff["colar"]) <= 0):
        _warn("restricted AA is no longer above full-label F1 for a weak run (R8-L4)")
    orc = (load_seeds(*ORACLE_CLEAN[:2], suffix="*"),
           load_seeds(*ORACLE_CLEAN[:2], key="full_label_f1", suffix="*"))  # fmt: skip
    for s in SEEDS:
        for m in (0, 1):
            vals = [[d[k][m][s] for k in g if s in d[k][m]] for g in groups]
            if s in orc[m]:
                vals.insert(2, [orc[m][s]])
            vals = [x for x in vals if x]
            if any(max(a) >= min(b) for a, b in zip(vals, vals[1:], strict=False)):
                _warn(f"group order differs, seed {s}, {'AA' if m == 0 else 'full-label'}")
    return _defs(v)


SUBNAIVE = ("cpfd", "is3", "coda_prompt", "lca")


def baseline_macros() -> str:
    """Why four baselines end below naive training (M1): their later-session diagonals.
    \\nCpfdLateMax = CPFD's best diagonal after session 1 (any seed); \\nBaseWorstLateMax = the
    largest, over IS3 / CODA-Prompt / LCA and their seeds, of the worst later-session diagonal;
    \\nNaiveDiagMin = the smallest diagonal of naive training (every session, every seed)."""
    mats = {k: matrices(src(f"cil_cord_{k}", (RESC, f"cil_cord_{k}", ""))) for k in SUBNAIVE}
    naive = matrices(C["naive"])
    v = {"nCpfdLateMax": f"{max(np.diag(m)[1:].max() for m in mats['cpfd']):.1f}"
         if mats["cpfd"] else "--"}  # fmt: skip
    worst = [np.diag(m)[1:].min() for k in SUBNAIVE[1:] for m in mats[k]]
    v["nBaseWorstLateMax"] = up1(max(worst)) if worst else "--"
    v["nNaiveDiagMin"] = down1(min(np.diag(m).min() for m in naive)) if naive else "--"
    nmin = min(pick(C["naive"])[0], default=math.inf)
    for k in SUBNAIVE:
        if max(pick(src(f"cil_cord_{k}", (RESC, f"cil_cord_{k}", "")))[0], default=0) >= nmin:
            _warn(f"{k} is not below naive training on every seed")
    return _defs(v)


# ---------------------------------------------------------------------------- P11: the ladder
# One row per test of a rung. "aa"/"full": sources; full is read from clean runs only.
# kind: "cell" = rule 6 (clean only at 3 seeds, else legacy with dagger); "clean" = clean seeds
# that exist, n marked (a control that has no legacy cell in this table); "slot" = a queued run:
# the row renders ONLY when ``min_n`` clean seeds exist, and its verdict cell stays empty until
# SLOT_VERDICT names one (the generator then warns: write the verdict and the prose).
ORACLE_PRE_NAME = "cil_cord_colar_r_oracle_pre"  # PM 2026-11-12: colar_r_oracle_pre, _k4_d50_r128
# Lit review A comparators (PM 2026-11-12; run names not reported yet: set them when Dev does)
SBR_CONSTBG_NAME = "cil_cord_colar_sbr_const_o"  # CoLaR-SBR, every background logit fixed at 0
ICE_O_NAME = "cil_cord_ice_o"  # ICE-O port: frozen encoder, head per session, constant Other = 0
SLOT_VERDICT = {"er_sbr": "", "oracle_pre": "", "colar_mib": "", "sbr_constbg": "",
                "ice_o": ""}  # set after reading the runs  # fmt: skip
MIX_RECALL_NOMEM = (GATE, "mixed_colar_sbr_recall_r64b", "*")  # seed 42, fp32 gate, legacy
MIX_RECALL_MEM = (GATE, "mixed_colar_sbr_recall_r64c", "*")
LADDER = (  # (rung, hypothesis, prediction, intervention, key, aa, full, kind, verdict, min_n)
    ("", "", "", "Naive fine-tuning (reference)", "naive", C["naive"], C["naive"], "cell", "", 0),
    ("H0", "Only drift: forgetting is just harder",
     "a memory that retains under domain shift retains here; raw beats compressed",
     "whole-document memory (CoLaR), shared softmax", "colar", C["colar"], C["colar"], "cell",
     "refutes", 0),
    ("", "", "", "raw-document memory (ER), shared softmax$^{e}$", "er", C["er"], C["er"], "cell",
     "refutes", 0),
    ("", "", "", "raw-document memory, blocked readout", "er_sbr", src("cil_cord_er_sbr"),
     src("cil_cord_er_sbr"), "slot", "", 3),
    ("H1", "Stale labels in the memory",
     "true labels in the stored documents close the gap",
     "older stored documents relabelled at each boundary$^{r}$", "oracle",
     [ORACLE_CLEAN], [ORACLE_CLEAN], "clean", "partly", 0),
    ("", "", "", "every stored document relabelled before each session", "oracle_pre",
     src(ORACLE_PRE_NAME), src(ORACLE_PRE_NAME), "slot", "", 3),
    ("H2", "Current documents mislabel old entities",
     "a loss that lets ``O'' mean ``O or an old class'' closes the gap",
     "replay and current losses sliced to their classes", "sliced2",
     src(None, (GATE, "cil_cord_colar_r_sliced2", "_k4_d50_r128")), [], "cell", "open", 0),
    ("", "", "", "MiB-style unbiased cross-entropy, both directions", "colar_mib",
     src("cil_cord_colar_mib"), src("cil_cord_colar_mib"), "slot", "", 3),
    ("H3", "The shared background normalisation",
     "one background per session closes the gap without true labels, given a memory",
     "CoLaR memory, session-blocked readout (CoLaR-SBR)", "sbr", C["sbr"], C["sbr"], "cell",
     "supports", 0),
    ("", "", "", "session-blocked readout, no memory", "nobank", C["nobank"], C["nobank"], "cell",
     "supports", 0),
    ("", "", "", "CoLaR memory, blocked readout with a constant background logit",
     "sbr_constbg", src(SBR_CONSTBG_NAME), src(SBR_CONSTBG_NAME), "slot", "", 3),
    ("H4", "Isolation replaces memory where a routing key exists",
     "per-session deltas need no memory on one schema, and fail across schemas",
     "per-session deltas, blocked readout, no memory (SBR-Recall)", "recall", C["recall"],
     C["recall"], "cell", "supports", 0),
    ("", "", "", "frozen encoder, one head per session, constant Other logit, no memory (ICE-O)",
     "ice_o", src(ICE_O_NAME), src(ICE_O_NAME), "slot", "", 3),
    ("", "", "", "the same on Mixed (AA), no memory", "mix_nomem", [MIX_RECALL_NOMEM], [],
     "cell", "supports", 0),
    ("", "", "", "the same on Mixed (AA), 50-document memory", "mix_mem", [MIX_RECALL_MEM], [],
     "cell", "supports", 0),
    ("", "", "", "Joint training (reference)", "joint", C["joint"], C["joint"], "cell", "", 0),
)  # fmt: skip


def _gap_share(a: dict, lo: dict, hi: dict) -> list[float]:
    """Per seed (seeds in all three): share of the lo -> hi gap that ``a`` covers."""
    return [(a[s] - lo[s]) / (hi[s] - lo[s]) for s in SEEDS if s in a and s in lo and s in hi]


def verdict_ok(key: str, verdict: str) -> bool | None:
    """R9-L7: the per-seed condition behind every typed verdict word (None: no check exists).
    Every condition must hold on every seed, under both metrics where both exist."""
    m2 = ("AA", "full_label_f1")
    c = {k: [_clean(k, m) for m in m2] for k in ("naive", "colar", "sbr", "nobank", "recall", "joint")}
    orc = [load_seeds(*ORACLE_CLEAN[:2], key=m, suffix="*") for m in m2]
    mix = {k: load_seeds(*s[:2], suffix=s[2]).get(42) for k, s in
           (("nomem", MIX_RECALL_NOMEM), ("mem", MIX_RECALL_MEM))}  # fmt: skip
    i = (0, 1)
    checks = {
        ("colar", "refutes"): lambda: all(x < 0.5 for j in i for x in
                                          _gap_share(c["colar"][j], c["naive"][j], c["joint"][j])),
        ("er", "refutes"): lambda: max(pick(C["er"])[0]) < min(pick(C["colar"])[0]),
        ("oracle", "partly"): lambda: bool(orc[0]) and all(
            c["colar"][j][s] < orc[j][s] < c["joint"][j][s] for j in i for s in orc[j]),
        ("sliced2", "open"): lambda: True,  # legacy, cross-protocol: no verdict is claimed
        ("sbr", "supports"): lambda: all(x >= 0.5 for j in i for x in
                                         _gap_share(c["sbr"][j], c["colar"][j], c["joint"][j]))
        and all(min(c["sbr"][j].values()) > max(orc[j].values(), default=-1) for j in i),
        ("nobank", "supports"): lambda: all(x < 0.5 for j in i for x in
                                            _gap_share(c["nobank"][j], c["naive"][j], c["joint"][j])),
        ("recall", "supports"): lambda: all(abs(c["recall"][j][s] - c["joint"][j][s]) < 3
                                            for j in i for s in c["recall"][j] if s in c["joint"][j]),
        ("mix_nomem", "supports"): lambda: mix["nomem"] < mix["mem"]
        and mix["nomem"] < min(pick(M["sbr"])[0]),
        ("mix_mem", "supports"): lambda: mix["mem"] > mix["nomem"],
    }  # fmt: skip
    f = checks.get((key, verdict))
    return None if f is None else bool(f())


def write_ladder() -> None:
    errors = []
    t = HDR + "\\begin{tabular}{@{}l p{2.6cm} p{3.6cm} p{4.1cm} c c l@{}}\n\\toprule\n"
    t += ("Rung & Hypothesis & Prediction if true & Intervention (CIL-CORD unless stated) & "
          "Full-label F1 & AA & Verdict \\\\\n\\midrule\n")  # fmt: skip
    for rung, hyp, pred, interv, key, aa_s, full_s, kind, verdict, min_n in LADDER:
        if kind == "slot":
            full = pick_clean_only(full_s, "full_label_f1")
            aa = pick_clean_only(aa_s, "AA")
            if len(aa[0]) < min_n:
                if aa[0]:
                    print(f"slot {key}: {len(aa[0])} clean seed(s), renders at {min_n}")
                continue
            verdict = SLOT_VERDICT.get(key, "")
        elif kind == "clean":
            aa, full = pick_clean_only(aa_s, "AA"), pick_clean_only(full_s, "full_label_f1")
        else:
            aa = pick(aa_s)
            full = pick_clean_only(full_s, "full_label_f1") if full_s else ([], True, False)
        # R9-L7: every rendered rung row carries a verdict, and the verdict's per-seed condition
        if key not in ("naive", "joint"):  # the two references carry none
            ok = verdict_ok(key, verdict) if verdict else False
            if not verdict:
                errors.append(f"ladder row {key} renders with no verdict: set SLOT_VERDICT")
            elif ok is None:
                errors.append(f"ladder row {key}: no check for verdict '{verdict}' (verdict_ok)")
            elif not ok:
                errors.append(f"ladder row {key}: verdict '{verdict}' fails its per-seed check")
        if rung or key == "joint":
            t += "\\midrule\n"
        # n is marked on every cell with fewer than three seeds, clean or legacy (R9-L1)
        t += (f"{rung} & {hyp} & {pred} & {interv} & {fmt(*full, show_n=True)} & "
              f"{fmt(*aa, show_n=True)} & {verdict} \\\\\n")  # fmt: skip
    if errors:
        raise SystemExit("make_clean_tables: ladder verdicts:\n  " + "\n  ".join(errors))
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_ladder.tex").write_text(t)


# ---------------------------------------------------------------------------- P11: mechanism
MECH_JSON = Path(CLEAN) / "mechanism_analysis.json"
MECH_ROWS = (  # (column label, method alias in metrics.json, analysis section, readout)
    ("Naive", "naive", "masked_splits", "shared"),  # R9-H1: the shared-softmax signature, no memory
    ("CoLaR", "colar", "masked_splits", "shared"),
    ("Relab.$^{r}$", "colar_r_oracle", "masked_splits", "shared"),
    # per-token outputs on the unmasked test set (seed-42 re-runs): render only once they exist
    ("CoLaR$^{d}$", "colar_dump", "full_label", "shared"),
    ("Naive$^{d}$", "naive_dump", "full_label", "shared"),
    ("No mem.", "colar_sbr_noreplay", "masked_splits", "blocked"),
    ("CoLaR-SBR", "colar_sbr", "masked_splits", "blocked"),
    ("SBR-Recall", "colar_sbr_recall_r64", "masked_splits", "blocked"),
)


def _mech_runs(res: dict, method: str, section: str) -> list[dict]:
    """Whole run records (``aa_vs_full`` and ``section``) of one method, in file order."""
    return [r for r in res.get("runs", {}).values()
            if r.get("method") == method and r.get(section)]  # fmt: skip


def _rng(vals, pct=False, upper=False) -> str:
    """min to max over seeds, rounded outward (``upper``: both ends are upper bounds, so both
    round up); one seed prints its value."""
    vs = [100 * x if pct else x for x in vals if x is not None]
    if not vs:
        return ""
    if len(vs) == 1:
        return f"${up1(vs[0]) if upper else f'{vs[0]:.1f}'}$"
    lo, hi = (up1(min(vs)) if upper else down1(min(vs))), up1(max(vs))
    return f"${lo}$" if lo == hi else f"${lo}$--${hi}$"


def mech_stats(res: dict, method: str, section: str) -> dict | None:
    runs = _mech_runs(res, method, section)
    if not runs:
        return None
    rs = [r[section] for r in runs]
    old = [r["old"] for r in rs]
    sp = [o["spans"] for o in old]
    # precision is undefined with no predicted old span (naive: TP = FP = 0); seqeval prints 0
    out = {"n": len(rs), "recall": [s["recall"] for s in sp],
           "precision": [s["precision"] if s["tp"] + s["fp"] else None for s in sp],
           "p_upper": any(s.get("fp_is_lower_bound") for s in sp),
           "tp_old": [s["tp"] for s in sp],
           # span-level indicator over ALL sessions' missed spans (App. B): not a bound
           "indicator": [(r.get("aa_vs_full", {}).get("implied") or {}).get("min_share_missed_spans_all_O")
                         for r in runs]}  # fmt: skip
    errs = [o.get("errors") for o in old]
    if all(errs):
        for k in ("share_to_O", "share_to_same_session", "share_to_other_session"):
            out[k] = [e[k] for e in errs]
    rout = [o.get("routing") for o in old]
    if all(rout):
        out["owner"] = [x["routed_to_owner"] / x["n_entity_tokens"] for x in rout]
        if all("gold_O_claimed_by_own_block" in x for x in rout):  # own block, own split
            out["false_claim"] = [x["gold_O_claimed_by_own_block"] / x["n_gold_O_tokens"]
                                  for x in rout]  # fmt: skip
    return out


def write_mechanism() -> str:
    """tab_mech.tex (old-session spans and tokens of the final model) + macros for the prose."""
    res = json.loads(MECH_JSON.read_text()) if MECH_JSON.exists() else {}
    # transposed (R9 layout): one column per run, grouped by readout; one row per measure
    st, cols = {}, []
    for label, method, section, readout in MECH_ROWS:
        s = mech_stats(res, method, section)
        if s is None:
            continue
        st[method] = s
        p = _rng(s["precision"], upper=s["p_upper"])
        if s["p_upper"] and p:  # FP a lower bound (per-class reports): precision an upper bound
            p = "$\\le$" + p
        cols.append((label, readout, [f"${s['n']}$", _rng(s["recall"]), p,
                     _rng(s["indicator"], True), _rng(s.get("owner", []), True),
                     _rng(s.get("share_to_O", []), True),
                     _rng(s.get("share_to_same_session", []), True),
                     _rng(s.get("share_to_other_session", []), True),
                     _rng(s.get("false_claim", []), True)]))  # fmt: skip
    n_sh = sum(c[1] == "shared" for c in cols)
    n_bl = len(cols) - n_sh
    t = HDR + "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\n"
    t += (f" & \\multicolumn{{{n_sh}}}{{c}}{{Shared softmax}} & "
          f"\\multicolumn{{{n_bl}}}{{c}}{{Blocked readout}} \\\\\n"
          f"\\cmidrule(lr){{2-{1 + n_sh}}}\\cmidrule(l){{{2 + n_sh}-{1 + len(cols)}}}\n")  # fmt: skip
    t += " & " + " & ".join(c[0] for c in cols) + " \\\\\n\\midrule\n"
    names = ("seeds $n$", "spans: R", "\\quad P", "\\quad missed at O$^{i}$", "tokens: owner",
             "\\quad wrong $\\to$ O", "\\quad $\\to$ same session", "\\quad $\\to$ other session",
             "false claims")  # fmt: skip
    for i, name in enumerate(names):
        t += name + " & " + " & ".join(c[2][i] for c in cols) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_mech.tex").write_text(t)

    v = {}

    def put(name, vals, pct=False, lo=True, up=False):
        """lo: the minimum (rounded down, or up when ``up``: an upper bound); else the max (up)."""
        vs = [100 * x if pct else x for x in vals or [] if x is not None]
        v[name] = ((up1 if up else down1)(min(vs)) if lo else up1(max(vs))) if vs else "--"

    for m, pre in (("naive", "Naive"), ("colar", "Colar"), ("colar_sbr_noreplay", "NoBank"),
                   ("colar_sbr", "Sbr"), ("colar_sbr_recall_r64", "Recall"),
                   ("colar_r_oracle", "Oracle")):  # fmt: skip
        s = st.get(m) or {}
        for k, short, pct in (("recall", "R", False), ("precision", "P", False),
                              ("owner", "Owner", True), ("share_to_O", "ToO", True),
                              ("share_to_same_session", "Same", True),
                              ("share_to_other_session", "Other", True),
                              ("false_claim", "False", True)):  # fmt: skip
            # R9-L8: an upper bound (precision from per-class reports) rounds up at both ends
            up = k == "precision" and s.get("p_upper")
            put(f"m{pre}{short}Min", s.get(k), pct, True, bool(up))
            put(f"m{pre}{short}Max", s.get(k), pct, False)
        ind = [100 * x for x in s.get("indicator") or [] if x is not None]  # whole percent
        v[f"m{pre}IndMin"] = f"{math.floor(min(ind))}" if ind else "--"
        v[f"m{pre}IndMax"] = f"{math.ceil(max(ind))}" if ind else "--"
    spans = res.get("gold_spans_per_session_test_split") or []
    v["mSessOneShare"] = f"{100 * spans[0] / sum(spans):.0f}" if spans else "--"
    v["mTestSpans"] = f"{sum(spans):,}".replace(",", "{,}") if spans else "--"
    # pooled per-session own-class counts vs the full-label F1 of the same final model
    gaps, gaps_nb, missed_o, last, imp_p, nb_bound, old_p_lb = [], [], [], [], [], [], []
    for r in res.get("runs", {}).values():
        a, me = r["aa_vs_full"], r["method"]
        if me.endswith("_dump") or a.get("pooled_own_class_f1_from_splits") is None:
            continue
        g = abs(a["pooled_own_class_f1_from_splits"] - a["full_label_f1"])
        (gaps_nb if me == "colar_sbr_noreplay" else gaps).append(g)
        if me == "colar_sbr_noreplay":  # the bound's failure case (per-token: no O errors)
            nb_bound.append((a.get("implied") or {}).get("min_share_missed_spans_all_O") or 0.0)
        if me == "colar":
            imp = a.get("implied") or {}
            missed_o.append(imp.get("min_share_missed_spans_all_O"))
            last.append(a["last_session_f1"])
            if imp.get("full_tp"):  # precision implied by the exact full-label F1, all sessions
                imp_p.append(100 * imp["full_tp"] / (imp["full_tp"] + imp["full_fp_implied"]))
                # R9-M1: a true lower bound on OLD-span precision: every old-class false positive
                # is one of the full test set's false positives, so FP_old <= FP_implied (all)
                tp_old = r["masked_splits"]["old"]["spans"]["tp"]
                old_p_lb.append(100 * tp_old / (tp_old + imp["full_fp_implied"]))
    v["mPoolGapMax"] = up1(max(gaps)) if gaps else "--"
    v["mPoolGapNoBankMax"] = up1(max(gaps_nb)) if gaps_nb else "--"
    v["mColarMissedOMin"] = (
        f"{math.floor(100 * min(missed_o))}" if all(missed_o) and missed_o else "--"
    )
    v["mColarLastMin"], v["mColarLastMax"] = (
        (down1(min(last)), up1(max(last))) if last else ("--", "--")
    )
    v["mNoBankBoundMax"] = f"{math.floor(100 * max(nb_bound))}" if nb_bound else "--"
    v["mColarImpPMin"], v["mColarImpPMax"] = (
        (down1(min(imp_p)), up1(max(imp_p))) if imp_p else ("--", "--")
    )
    v["mColarOldPLb"] = down1(min(old_p_lb)) if old_p_lb else "--"
    nb = st.get("colar_sbr_noreplay")
    if nb and max(nb.get("share_to_O", [1])) > 0:
        _warn("no-memory SBR now has predicted-O errors on old sessions")
    # H0 rests on: CoLaR's newest session ends far below every diagonal of naive training (per
    # seed), naive keeps no old span, CoLaR keeps some
    naive_diag = [np.diag(m).min() for m in matrices(C["naive"])]
    if last and naive_diag and max(last) >= min(naive_diag):
        _warn("a CoLaR seed's newest session reaches a naive-training diagonal (H0)")
    nv, co = st.get("naive") or {}, st.get("colar") or {}
    if max(nv.get("recall", [0])) > 0 or min(co.get("recall", [0])) <= 0:
        _warn("naive keeps an old span, or a CoLaR seed keeps none (H0)")
    return _defs(v)


def gap_macros() -> str:
    """CoLaR-SBR below joint training per seed (both metrics); Mixed SBR-Recall; frozen SBR."""
    v = {}
    for key, nm in (("AA", "Aa"), ("full_label_f1", "Full")):
        s, j = _clean("sbr", key), _clean("joint", key)
        g = [j[x] - s[x] for x in SEEDS if x in s and x in j]
        v[f"nSbrGap{nm}Min"] = down1(min(g)) if g else "--"
        v[f"nSbrGap{nm}Max"] = up1(max(g)) if g else "--"
    for nm, srcs in (("nRecallMixNoMem", MIX_RECALL_NOMEM), ("nRecallMixMem", MIX_RECALL_MEM),
                     ("nSbrFrozen", (GATE, "cil_cord_colar_sbr_frozen", "*"))):  # fmt: skip
        x = load_seeds(*srcs[:2], suffix=srcs[2]).get(42)
        v[nm] = f"{x:.1f}" if x is not None else "--"
    er, co = pick(C["er"])[0], pick(C["colar"])[0]
    if er and co and max(er) >= min(co):
        _warn("an ER seed reaches a CoLaR seed (H0 'raw is no better')")
    return _defs(v)


# ---------------------------------------------------------------------------- P11: beyond
BEYOND = (  # (stream, label, run name): a row renders only when its three clean seeds exist,
    # except the session-order rows, which are seed 42 by design (n marked)
    ("CIL-FUNSD", "Naive fine-tuning", "cil_funsd_naive", 3),
    ("CIL-FUNSD", "CoLaR, shared softmax", "cil_funsd_colar", 3),
    ("CIL-FUNSD", "CoLaR-SBR", "cil_funsd_colar_sbr_d10", 3),
    ("CIL-FUNSD", "SBR-Recall", "cil_funsd_colar_sbr_recall_r64", 3),
    ("CIL-FUNSD", "Joint training (reference)", "cil_funsd_joint", 3),
    ("CIL-CORD, BROS", "CoLaR-SBR", "cil_cord_colar_sbr_bros", 3),
    ("CIL-CORD, order 2", "SBR-Recall", "cil_cord_o2_colar_sbr_recall_r64", 1),
    ("CIL-CORD, order 3", "SBR-Recall", "cil_cord_o3_colar_sbr_recall_r64", 1),
)


def write_beyond() -> None:
    """tab_beyond.tex: the WHOLE float, or an empty file (renders nothing) while no row exists."""
    rows = ""
    for stream, label, name, min_n in BEYOND:
        aa = load(CLEAN, name, "AA")
        if len(aa) < min_n:
            continue
        fl = load(CLEAN, name, "full_label_f1")
        rows += (f"{stream} & {label} & {fmt(fl, True, show_n=True)} & "
                 f"{fmt(aa, True, show_n=True)} \\\\\n")  # fmt: skip
    if not rows:
        (OUT / "tab_beyond.tex").write_text(HDR)
        return
    t = HDR + "\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\n"
    t += ("\\caption{\\textbf{Beyond one stream} (clean protocol; mean $\\pm$ s.d. over three "
          "seeds, per-seed values when fewer; $^{n=k}$: $k$ seeds). CIL-FUNSD: FUNSD's entity "
          "types over two sessions, memory of $10$ documents per session. BROS: the method on a "
          "second backbone. Orders 2 and 3: CIL-CORD's sessions permuted.}\n"
          "\\label{tab:beyond}\n")  # fmt: skip
    t += "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llcc}\n\\toprule\n"
    t += "Stream & Method & Full-label F1 & AA \\\\\n\\midrule\n" + rows
    t += "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    (OUT / "tab_beyond.tex").write_text(t)


def main():
    cell = {k: pick(v) for k, v in C.items()}
    mix = {k: pick(v) for k, v in M.items()}
    dil = {k: pick(v) for k, v in D.items()}
    full = {
        k: pick_clean_only(C[k], "full_label_f1")
        for k in ("naive", "nobank", "colar", "sbr", "recall", "joint", "er")
    }

    # ---- vals.tex
    names = {
        "naive": "Naive", "nobank": "NoBank", "colar": "Colar", "sbr": "Sbr", "cllora": "ClLora",
        "recall": "Recall", "joint": "Joint", "er": "Er", "slot": "Slot", "sbr_bros": "SbrBros",
    }  # fmt: skip
    v = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    for k, n in names.items():
        v += macro(n, cell[k][0])
    for (_, sbr_g, slot_g), n in zip(SLOT_PAIRS, ("Gate", "GateMix", "GateDil"), strict=True):
        v += macro(f"Sbr{n}", load(sbr_g[0], sbr_g[1], suffix=sbr_g[2]))
        v += macro(f"Slot{n}", load(slot_g[0], slot_g[1], suffix=slot_g[2]))
    for k, n in (("sbr", "SbrMix"), ("joint", "JointMix"), ("colar", "ColarMix"), ("er", "ErMix")):
        v += macro(n, mix[k][0])
    for k, n in (("sbr", "SbrDil"), ("joint", "JointDil"), ("colar", "ColarDil"), ("er", "ErDil")):
        v += macro(n, dil[k][0])
    for k, n in (
        ("sbr", "SbrFull"),
        ("recall", "RecallFull"),
        ("joint", "JointFull"),
        ("nobank", "NoBankFull"),
        ("naive", "NaiveFull"),
        ("colar", "ColarFull"),
    ):
        v += macro(n, full[k][0])
    er_src = chosen_source(C["er"])  # the CIL-CORD ER buffer the \nEr cell was trained with
    v += f"\\newcommand{{\\nErCilBuf}}{{{250 if er_src and er_src[0] == CLEAN else 200}}}\n"
    v += recall_macros() + oracle_macros() + conflict_macros() + agree_macros()
    v += baseline_macros() + gap_macros() + write_mechanism()
    (OUT / "vals.tex").write_text(v)
    write_ladder()
    write_beyond()

    # ---- tab_2x2.tex
    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"

    def fl2(k, **kw):  # full-label F1 (the headline) before AA: clean seeds only, n when < 3
        return fmt(*full[k], show_n=True, **kw)

    t += "\\begin{tabular}{lcccc}\n\\toprule\n"
    t += " & \\multicolumn{2}{c}{\\textbf{Shared softmax}} & \\multicolumn{2}{c}{\\textbf{Session-blocked readout}} \\\\\n"
    t += "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & full-label F1 & AA & full-label F1 & AA \\\\\n\\midrule\n"
    t += f"No memory & {fl2('naive')} & {fmt(*cell['naive'])} & {fl2('nobank')} & {fmt(*cell['nobank'])} \\\\\n"
    t += (f"CoLaR memory & {fl2('colar')} & {fmt(*cell['colar'])} & "
          f"{fl2('sbr', bold=True)} & {fmt(*cell['sbr'], bold=True)} \\\\\n")  # fmt: skip
    # The reference is not a cell of the 2x2: one row spanning both readouts, labelled metrics.
    t += (f"\\midrule\nJoint training (reference) & \\multicolumn{{4}}{{c}}{{full-label F1 {fl2('joint')}, "
          f"AA {fmt(*cell['joint'])}}} \\\\\n")  # fmt: skip
    t += "\\bottomrule\n\\end{tabular}\n"
    (OUT / "tab_2x2.tex").write_text(t)

    # ---- tab_main_clean.tex
    def fl(k, **kw):  # full-label F1 cell: clean seeds only, n marked when < 3
        return fmt(*full[k], show_n=True, **kw)

    def row(label, c, f, m, d, bold=False):
        lab = f"\\textbf{{{label}}}" if bold else label
        return f"{lab} & {f} & {c} & {m} & {d} \\\\\n"

    t = "% AUTO-GENERATED by scripts/rca/make_clean_tables.py -- do not edit.\n"
    t += "\\begin{tabular}{lcccc}\n\\toprule\n"
    t += " & \\multicolumn{2}{c}{\\textbf{CIL-CORD}} & \\textbf{Mixed} & \\textbf{DIL} \\\\\n"
    t += "\\textbf{Method} & full-label F1 & AA & AA & AA \\\\\n\\midrule\n"
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
        fl("recall", bold=True),
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
