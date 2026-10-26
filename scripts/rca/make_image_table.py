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

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_cvpr" / "tab_image_core.tex"
SEEDS = (42, 7, 123)
SCEN = ("cil_cifar100", "cil_imagenet_r")

# label, {scenario: run-dir template with {sc} and {sd}} (None = not applicable), group
ROWS = [
    ("Naive (lower bound)", "{sc}_naive_seed{sd}_vit", "base"),
    ("LwF", "{sc}_lwf_seed{sd}_vit", "base"),
    ("ER (2{,}000 exemplars)", "{sc}_er_b2000_seed{sd}_vit", "base"),
    ("DER++ (2{,}000 exemplars)", "{sc}_der_pp_b2000_seed{sd}_vit", "base"),
    ("CODA-Prompt", "{sc}_coda_prompt_seed{sd}_vit", "base"),
    ("CL-LoRA (2025)", "{sc}_cl_lora_seed{sd}_vit", "base"),
    ("SLCA (2023)", "{sc}_slca_seed{sd}_vit", "base"),
    ("RanPAC (2023)", "{sc}_ranpac_seed{sd}_vit", "base"),
    ("LCA (2026)", "{sc}_lca_seed{sd}_vit", "base"),
    (
        "\\textbf{Latent bank, 500/class (ours, uncompressed)}",
        "{sc}_bank500_seed{sd}_vit_k4_d500",
        "ours",
    ),
    (
        "\\textbf{CoLaR $r{=}128$, 500/class (ours)}",
        {"cil_cifar100": "{sc}_colar500r128_seed{sd}_vit_k4_d500_r128", "cil_imagenet_r": "{sc}_colar500r128_seed{sd}_vit_k4_d500_r128"},
        "ours",
    ),
    ("Joint (upper bound)", "{sc}_joint_seed{sd}_vit", "oracle"),
]
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
    return f"{st.mean(vals):.1f}\\;{{\\scriptsize $\\pm$ {st.stdev(vals):.1f}}}"


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
    ("CIFAR-100, latent bank", "cil_cifar100_rcaBank_seed42_vit"),
    ("ImageNet-R, latent bank", "cil_imagenet_r_colar_pp_seed42_vit"),
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


def main() -> None:
    write_supp()
    write_probe()
    lines = [
        "% Auto-generated by scripts/rca/make_image_table.py (do not edit by hand).",
        "% Metric: AA (mean $\\pm$ std over seeds 42/7/123); ``pending'' = not all seeds on disk yet.",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{CIFAR-100} & \\textbf{ImageNet-R} \\\\",
        "\\midrule",
    ]
    prev = None
    for label, tmpl, group in ROWS:
        if prev and group != prev:
            lines.append("\\midrule")
        prev = group
        tm = tmpl if isinstance(tmpl, dict) else dict.fromkeys(SCEN, tmpl)
        lines.append(f"{label} & " + " & ".join(cell(tm[sc], sc) for sc in SCEN) + " \\\\")
        if group == "ours" and label.startswith("\\textbf{CoLaR"):
            lines.append(
                "Ridge readout on the latent-bank trunk (probe, seed 42) & "
                + " & ".join(probe_cell(sc) for sc in SCEN)
                + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT.write_text("\n".join(lines))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
