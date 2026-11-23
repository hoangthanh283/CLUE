#!/usr/bin/env python3
"""Image-side comparison table (frozen 12-row set, ViT-B/16) -> paper_cvpr/tab_image_core.tex.

Reads results/cil_{cifar100,imagenet_r}_<run>/metrics.json directly; a cell whose seeds are
not all on disk is rendered as ``pending'' (never imputed). Re-run after every image cell lands:

    uv run python scripts/rca/make_image_table.py

ponytail: one table, hard-coded row map; extend ROWS if the frozen set ever changes (it should not).
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from table_emph import emph

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_cvpr" / "tab_image_core.tex"
SEEDS = (42, 7, 123)
SCEN = ("cil_cifar100", "cil_imagenet_r")

# the three-part training scheme for exclusive sessions (negatives, balanced refit, slow encoder)
SBRX = "{sc}_colar_sbrx500r128_nw010_rf200_sl_seed{sd}_vit"
# label, {scenario: run-dir template with {sc} and {sd}} (None = not applicable), group
ROWS = [
    ("Naive", "{sc}_naive_seed{sd}_vit", "base"),
    ("LwF", "{sc}_lwf_seed{sd}_vit", "base"),
    ("ER (2{,}000 exemplars)", "{sc}_er_b2000_seed{sd}_vit", "base"),
    ("DER++ (2{,}000 exemplars)", "{sc}_der_pp_b2000_seed{sd}_vit", "base"),
    ("CODA-Prompt", "{sc}_coda_prompt_seed{sd}_vit", "base"),
    ("CL-LoRA (2025)", "{sc}_cl_lora_seed{sd}_vit", "base"),
    ("SLCA (2023)", "{sc}_slca_seed{sd}_vit", "base"),
    ("RanPAC (2023)", "{sc}_ranpac_seed{sd}_vit", "base"),
    ("LCA (2026)", "{sc}_lca_seed{sd}_vit", "base"),
    ("Latent memory (ours, uncompressed)", "{sc}_bank500_seed{sd}_vit_k4_d500", "ours"),
    ("CoLaR (ours)", "{sc}_colar500r128_seed{sd}_vit_k4_d500_r128", "ours"),
    ("CoLaR-SBR (ours)", "{sc}_colar_sbr500r128_seed{sd}_vit", "ours"),
    ("CoLaR-SBR, session-exclusive training (ours)", SBRX, "ours"),
    ("Joint training (reference)", "{sc}_joint_seed{sd}_vit", "oracle"),
]
# what each row keeps across sessions (Table 17's memory column; MB = 10^6 bytes: ER as computed
# in write_vals, ours as recorded by the runs as replay_memory_bytes)
STORED = {
    "Naive": "none",
    "LwF": "previous model",
    "ER (2{,}000 exemplars)": "2{,}000 images, 1{,}204 MB",
    "DER++ (2{,}000 exemplars)": "2{,}000 images + logits, 1{,}204 MB",
    "CODA-Prompt": "prompt pool",
    "CL-LoRA (2025)": "previous model, adapters",
    "SLCA (2023)": "class Gaussians",
    "RanPAC (2023)": "class statistics",
    "LCA (2026)": "session backbones, class Gaussians",
    "Latent memory (ours, uncompressed)": "5{,}000 latents, 1{,}513 MB",
    "CoLaR (ours)": "5{,}000 latents, 1{,}235 MB",
    "CoLaR-SBR (ours)": "5{,}000 latents, 1{,}235 MB",
    "CoLaR-SBR, session-exclusive training (ours)": "5{,}000 latents, 1{,}235 MB",
    "Joint training (reference)": "all data",
}
# Ridge-readout probe on the adapted latent-bank trunk (Pass 2b, seed 42): rp_full from probes.json.
PROBE = {
    "cil_cifar100": "cil_cifar100_rcaBank_seed42_vit",
    "cil_imagenet_r": "cil_imagenet_r_colar_pp_seed42_vit",
}


def aa(run: str) -> float | None:
    p = RES / run / "metrics.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return float(json.load(fh)["AA"])


def cell(tmpl: str | None, sc: str) -> str:
    if tmpl is None:
        return "--"
    vals = [v for sd in SEEDS if (v := aa(tmpl.format(sc=sc, sd=sd))) is not None]
    if not vals:
        return "\\emph{pending}"
    if len(vals) < len(SEEDS):
        m = st.mean(vals)
        return f"{m:.1f}\\;{{\\scriptsize ({len(vals)} seeds, pending)}}"
    return f"${st.mean(vals):.1f}{{\\pm}}{st.stdev(vals):.1f}$"  # the style of the document tables


def probe_cell(sc: str) -> str:
    p = RES / PROBE[sc] / "probes.json"
    if not p.exists():
        return "--"
    with open(p) as fh:
        v = json.load(fh).get("rp_full")
    return f"{float(v):.1f}" if v is not None else "--"


# Supplement: every image run outside the frozen set that is kept as evidence (same recipe).
EXTRA = [
    ("EWC", "{sc}_ewc_seed{sd}_vit"),
    ("ER (200 exemplars)", "{sc}_er_seed{sd}_vit"),
    ("DER++ (200 exemplars)", "{sc}_der_pp_seed{sd}_vit"),
    ("SLCA without classifier alignment", "{sc}_slca_noca_seed{sd}_vit"),
    ("SimpleCIL (frozen trunk, NCM)", "{sc}_simplecil_seed{sd}_vit"),
    (
        "Per-class Gaussian latent replay (falsified)",
        {"cil_cifar100": "{sc}_aglr_replay_seed{sd}_vit_keep100", "cil_imagenet_r": None},
    ),
]
OUT_SUPP = ROOT / "paper_cvpr" / "tab_image_supp.tex"
OUT_PROBE = ROOT / "paper_cvpr" / "tab_probe_decomp.tex"
# ponytail: the *_lexslot_seed42 probes exist too but were taken on the buffer-free LexSlot run,
# not the `_off` run behind the LexSlot row in tab:main, so they are left out of the paper.
PROBE_CELLS = [  # (label, run dir) -- Pass 2b, seed 42
    ("CIFAR-100, latent-memory run$^{v}$", "cil_cifar100_rcaBank_seed42_vit"),
    ("ImageNet-R, CoLaR run$^{v}$", "cil_imagenet_r_colar_pp_seed42_vit"),
    ("DIL, CoLaR", "dil_colar_pp_seed42"),
    ("CIL-CORD, CoLaR", "cil_cord_colar_pp_seed42"),
    ("Mixed, CoLaR", "mixed_colar_pp_seed42"),
]
PROBE_COLS = (
    "head",
    "rp_stored",
    "rp_full",
    "rp_full_frozen",
    "head_gap",
    "trunk_gap",
    "coverage_gap",
)


def metric(run: str, key: str) -> float | None:
    p = RES / run / "metrics.json"
    if not p.exists():
        return None
    with open(p) as fh:
        v = json.load(fh).get(key)
    return None if v is None or v != v else float(v)  # NaN -> None


def stat_cell(tmpl: str | None, sc: str, key: str) -> str:
    if tmpl is None:
        return "--"
    vals = [v for sd in SEEDS if (v := metric(tmpl.format(sc=sc, sd=sd), key)) is not None]
    if not vals:
        return "\\emph{pending}" if key == "AA" else "--"
    if len(vals) == 1:
        return f"{vals[0]:.1f}\\;{{\\scriptsize (1 seed)}}"
    tag = (
        f"$\\pm$ {st.stdev(vals):.1f}"
        if len(vals) == len(SEEDS)
        else f"({len(vals)} seeds, pending)"
    )
    return f"{st.mean(vals):.1f}\\;{{\\scriptsize {tag}}}"


def write_supp() -> None:
    lines = [
        "% Auto-generated by scripts/rca/make_image_table.py (do not edit by hand).",
        "% AA and BWT, mean $\\pm$ std over seeds 42/7/123; frozen reference set first, then the extra rows kept as evidence.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{CIFAR-100}} & \\multicolumn{2}{c}{\\textbf{ImageNet-R}} \\\\",
        "\\textbf{Method} & AA & BWT & AA & BWT \\\\",
        "\\midrule",
    ]
    rows = [(lab, tm, g) for lab, tm, g in ROWS] + [(lab, tm, "extra") for lab, tm in EXTRA]
    prev = None
    for label, tmpl, group in rows:
        if prev and group != prev:
            lines.append("\\midrule")
        prev = group
        tm = tmpl if isinstance(tmpl, dict) else dict.fromkeys(SCEN, tmpl)
        cells = [stat_cell(tm[sc], sc, k) for sc in SCEN for k in ("AA", "BWT")]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT_SUPP.write_text("\n".join(lines))


def write_probe() -> None:
    lines = [
        "% Auto-generated by scripts/rca/make_image_table.py from results/*/probes.json (Pass 2b, seed 42; do not edit by hand).",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Cell} & Head & Ridge, stored & Ridge, full & Ridge, frozen & Head gap & Trunk gap & Coverage gap \\\\",
        "\\midrule",
    ]
    for label, run in PROBE_CELLS:
        p = RES / run / "probes.json"
        if not p.exists():
            continue
        with open(p) as fh:
            d = json.load(fh)
        cells = ["--" if d.get(k) is None else f"{float(d[k]):.1f}" for k in PROBE_COLS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT_PROBE.write_text("\n".join(lines))


# P18: macros for the image appendix prose (supp.tex App. L), so no image number is typed there.
# key -> run template (both scenarios); names are letters only (LaTeX macro names).
VAL_RUNS = {
    "Naive": "{sc}_naive_seed{sd}_vit", "ErTwoH": "{sc}_er_seed{sd}_vit",
    "ErTwoK": "{sc}_er_b2000_seed{sd}_vit", "DerTwoH": "{sc}_der_pp_seed{sd}_vit",
    "DerTwoK": "{sc}_der_pp_b2000_seed{sd}_vit", "Bank": "{sc}_bank500_seed{sd}_vit_k4_d500",
    "Colar": "{sc}_colar500r128_seed{sd}_vit_k4_d500_r128", "Slca": "{sc}_slca_seed{sd}_vit",
    "SlcaNoca": "{sc}_slca_noca_seed{sd}_vit", "Ranpac": "{sc}_ranpac_seed{sd}_vit",
    "Lca": "{sc}_lca_seed{sd}_vit", "Joint": "{sc}_joint_seed{sd}_vit",
    "Sbr": "{sc}_colar_sbr500r128_seed{sd}_vit", "SbrX": SBRX,
}  # fmt: skip
DS = {"cil_cifar100": "Cifar", "cil_imagenet_r": "Imnr"}
OUT_VALS = ROOT / "paper_cvpr" / "image_vals.tex"


def seeds_of(tmpl: str, sc: str, key: str = "AA") -> dict[int, float]:
    return {sd: v for sd in SEEDS if (v := metric(tmpl.format(sc=sc, sd=sd), key)) is not None}


def _warn(msg: str) -> None:
    print(f"WARNING (image prose no longer holds, edit it): {msg}")


def write_vals() -> None:
    """image_vals.tex: \\img<Run><Dataset> = mean AA over three seeds ("--" otherwise), plus the
    per-seed facts the App. L prose states (each checked here on every seed)."""
    import math

    t = "% Auto-generated by scripts/rca/make_image_table.py (do not edit by hand).\n"
    v = {sc: {k: seeds_of(tm, sc) for k, tm in VAL_RUNS.items()} for sc in SCEN}
    for sc, ds in DS.items():
        for k, d in v[sc].items():
            val = f"{st.mean(d.values()):.1f}" if len(d) == len(SEEDS) else "--"
            t += f"\\newcommand{{\\img{k}{ds}}}{{{val}}}\n"
    c, r = v["cil_cifar100"], v["cil_imagenet_r"]
    for ours in ("Bank", "Colar"):
        if not all(c[ours][s] > max(c["ErTwoK"][s], c["DerTwoK"][s]) for s in SEEDS):
            _warn(f"{ours} is not above ER and DER++ (2,000) on every CIFAR-100 seed")
        for sc in SCEN:
            if not all(v[sc][b][s] > v[sc][ours][s] for b in ("Slca", "Ranpac") for s in SEEDS):
                _warn(f"SLCA or RanPAC is not above {ours} on every seed of {sc}")
        gap = max(abs(r[ours][s] - r["ErTwoK"][s]) for s in SEEDS)
        t += f"\\newcommand{{\\img{ours}ErGapImnr}}{{{math.ceil(10 * gap - 1e-9) / 10:.1f}}}\n"
    # CoLaR-SBR against CoLaR, seed-matched (the prose states the ImageNet-R gain on every seed)
    for sc, ds in DS.items():
        dd = [v[sc]["Sbr"][s] - v[sc]["Colar"][s] for s in SEEDS]
        t += f"\\newcommand{{\\imgSbrMinusColar{ds}Min}}{{{min(dd):+.1f}}}\n"
        t += f"\\newcommand{{\\imgSbrMinusColar{ds}Max}}{{{max(dd):+.1f}}}\n"
    # the session-exclusive training scheme against CoLaR-SBR (run-matched), SLCA and joint (means)
    mean = lambda sc, k: st.mean(v[sc][k].values())  # noqa: E731
    for sc, ds in DS.items():
        if len(v[sc]["SbrX"]) != len(SEEDS):
            continue
        dd = [v[sc]["SbrX"][s] - v[sc]["Sbr"][s] for s in SEEDS]
        t += f"\\newcommand{{\\imgSbrXMinusSbr{ds}Min}}{{{min(dd):+.1f}}}\n"
        t += f"\\newcommand{{\\imgSbrXMinusSbr{ds}Max}}{{{max(dd):+.1f}}}\n"
        t += f"\\newcommand{{\\imgSbrXMinusSlca{ds}}}{{{mean(sc, 'SbrX') - mean(sc, 'Slca'):+.1f}}}\n"
        t += f"\\newcommand{{\\imgJointMinusSbrX{ds}}}{{{mean(sc, 'Joint') - mean(sc, 'SbrX'):.1f}}}\n"
        t += (
            f"\\newcommand{{\\imgSlcaMinusSbrX{ds}}}{{{mean(sc, 'Slca') - mean(sc, 'SbrX'):.1f}}}\n"
        )
        for k, tm in (("SbrX", SBRX), ("Sbr", VAL_RUNS["Sbr"]), ("Slca", VAL_RUNS["Slca"])):
            af = seeds_of(tm, sc, "AF")
            if len(af) == len(SEEDS):
                t += f"\\newcommand{{\\img{k}Af{ds}}}{{{st.mean(af.values()):.1f}}}\n"
        if not all(v[sc]["SbrX"][s] > v[sc]["Ranpac"][s] for s in SEEDS):
            _warn(f"the session-exclusive scheme is not above RanPAC on every run of {sc}")
    # the scheme without its refit (negatives + slow encoder), where all three runs exist
    for sc, ds in DS.items():
        nr = seeds_of("{sc}_colar_sbrx500r128_nw010_sl_seed{sd}_vit", sc)
        if len(nr) == len(SEEDS) and len(v[sc]["SbrX"]) == len(SEEDS):
            t += f"\\newcommand{{\\imgNoRefit{ds}}}{{{st.mean(nr.values()):.1f}}}\n"
            t += f"\\newcommand{{\\imgNoRefitSd{ds}}}{{{st.stdev(nr.values()):.1f}}}\n"
            t += f"\\newcommand{{\\imgSbrXSd{ds}}}{{{st.stdev(v[sc]['SbrX'].values()):.1f}}}\n"
    if not all(v[sc]["Slca"][s] > v[sc]["Sbr"][s] for sc in SCEN for s in SEEDS):
        _warn("SLCA is not above CoLaR-SBR on every seed")
    if not all(r["Sbr"][s] > max(r["Ranpac"][s], r["ErTwoK"][s], r["Bank"][s]) for s in SEEDS):
        _warn("CoLaR-SBR is not above RanPAC, ER and the latent memory on every ImageNet-R seed")
    cb = max(abs(v[sc]["Colar"][s] - v[sc]["Bank"][s]) for sc in SCEN for s in SEEDS)
    t += f"\\newcommand{{\\imgColarBankGapMax}}{{{math.ceil(10 * cb - 1e-9) / 10:.1f}}}\n"
    for sc in SCEN:
        if not all(v[sc]["Slca"][s] > max(v[sc][k][s] for k in ("Bank", "Colar", "ErTwoK"))
                   for s in SEEDS):  # fmt: skip
            _warn(f"SLCA is not the highest final AA of the trajectory figure on {sc}")
    for k in ("Bank", "Colar"):
        b = {metric(VAL_RUNS[k].format(sc=sc, sd=sd), "replay_memory_bytes")
             for sc in SCEN for sd in SEEDS}  # fmt: skip
        b.discard(None)
        t += f"\\newcommand{{\\img{k}MB}}{{{max(b) / 1e6:,.0f}}}\n".replace(",", "{,}")
        if len(b) > 1:
            _warn(f"{k}: replay_memory_bytes differs across runs")
    # R13-M2: ER's buffer as implemented (doccl/methods/buffer.py stores each batch tensor; the
    # vision transform yields float32 pixel_values at the Resize size of doccl/data/vision.py, plus
    # one int64 label): bytes = buffer_size x (3 x H x W x 4 + 8), with buffer_size as recorded.
    import re

    size = re.search(r"Resize\(\((\d+), (\d+)\)\)", (ROOT / "doccl/data/vision.py").read_text())
    h, w = int(size.group(1)), int(size.group(2))
    bufs = {(json.loads((RES / VAL_RUNS["ErTwoK"].format(sc=sc, sd=sd) / "metrics.json").read_text())
             ["method_hparams"]["buffer_size"]) for sc in SCEN for sd in SEEDS}  # fmt: skip
    if len(bufs) != 1:
        _warn("ER-2000 runs record different buffer sizes")
    er_mb = bufs.pop() * (3 * h * w * 4 + 8) / 1e6
    t += f"\\newcommand{{\\imgErTwoKMB}}{{{er_mb:,.0f}}}\n".replace(",", "{,}")
    t += f"\\newcommand{{\\imgSide}}{{{h}}}\n"
    OUT_VALS.write_text(t)


ABL = (  # (negatives, refit, slow encoder, run stem) -- one run each, the same seed
    (0, 0, 0, "colar_sbr500r128"),
    (1, 0, 0, "colar_sbrx500r128_nw010"),
    (1, 1, 0, "colar_sbrx500r128_nw010_rf200"),
    (0, 0, 1, "colar_sbr500r128_sl"),
    (1, 0, 1, "colar_sbrx500r128_nw010_sl"),
    (1, 1, 1, "colar_sbrx500r128_nw010_rf200_sl"),
)


def write_ablation() -> None:
    """tab_image_ablation.tex: the three parts of the session-exclusive training scheme, one run
    each on the same seed (AA and AF); a cell without its run is ``pending``."""
    ck = lambda x: "\\checkmark" if x else ""  # noqa: E731
    t = "% Auto-generated by scripts/rca/make_image_table.py (do not edit by hand).\n"
    t += "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
    t += " & & & & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet-R} \\\\\n"
    t += "\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n"
    t += " & Negatives & Refit & Two rates & AA $\\uparrow$ & AF $\\downarrow$ & "
    t += "AA $\\uparrow$ & AF $\\downarrow$ \\\\\n\\midrule\n"
    for i, (n, r, sl, stem) in enumerate(ABL):
        cells = []
        for sc in SCEN:
            for key in ("AA", "AF"):
                # every run the row has: mean +- s.d. with three runs, the value with one
                vs = [
                    x for sd in SEEDS if (x := metric(f"{sc}_{stem}_seed{sd}_vit", key)) is not None
                ]
                # slow encoder without negatives: the late sessions are never learned, so AF
                # (what was learned and then lost) says nothing there; printed as -- with a mark
                if key == "AF" and (n, r, sl) == (0, 0, 1):
                    cells.append("--$^{u}$")
                elif not vs:
                    cells.append("\\emph{pending}")
                elif len(vs) == len(SEEDS) and key == "AA":
                    cells.append(f"${st.mean(vs):.1f}{{\\pm}}{st.stdev(vs):.1f}$")
                else:
                    assert len(vs) in (1, len(SEEDS)), (stem, sc, len(vs))  # no half-filled row
                    cells.append(f"${st.mean(vs):.1f}$")
        shade = "\\rowcolor{black!7} " if (n, r, sl) == (1, 1, 1) else ""  # the full scheme
        idx = ("I", "II", "III", "IV", "V", "VI")[i]
        t += f"{shade}{idx} & {ck(n)} & {ck(r)} & {ck(sl)} & " + " & ".join(cells) + " \\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    (ROOT / "paper_cvpr" / "tab_image_ablation.tex").write_text(t)


def main() -> None:
    write_ablation()
    write_supp()
    write_probe()
    write_vals()
    lines = [
        "% Auto-generated by scripts/rca/make_image_table.py (do not edit by hand).",
        "% AA and AF, mean $\\pm$ std over three runs; ``pending'' = not all runs on disk yet.",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{CIFAR-100} & \\multicolumn{2}{c}{ImageNet-R} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
        "Method & Kept across sessions & AA $\\uparrow$ & AF $\\downarrow$ & AA $\\uparrow$ & AF $\\downarrow$ \\\\",
        "\\midrule",
    ]
    # bold: best per column, Joint (the reference) aside; underlined: our best otherwise
    tms = [t if isinstance(t, dict) else dict.fromkeys(SCEN, t) for _, t, _ in ROWS]
    ours, ref = [g == "ours" for *_, g in ROWS], [g == "oracle" for *_, g in ROWS]

    def af_cell(tm, sc, is_ref):
        vals = [x for sd in SEEDS if (x := metric(tm.format(sc=sc, sd=sd), "AF")) is not None]
        return f"${st.mean(vals):.1f}$" if len(vals) == len(SEEDS) and not is_ref else "--"

    cols = []
    for sc in SCEN:
        cols.append(emph([cell(tm[sc], sc) for tm in tms], ours, ref))
        cols.append(
            emph(
                [af_cell(tm[sc], sc, r) for tm, r in zip(tms, ref, strict=True)],
                ours,
                ref,
                lower=True,
            )
        )
    prev = None
    for (label, _tmpl, group), *cells in zip(ROWS, *cols, strict=True):
        if prev and group != prev:
            lines.append("\\midrule")
        prev = group
        shade = "\\rowcolor{black!7} " if group == "ours" else ""  # ours, as in the main table
        lines.append(f"{shade}{label} & {STORED[label]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT.write_text("\n".join(lines))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
