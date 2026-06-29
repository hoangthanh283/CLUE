#!/usr/bin/env python3
"""Generate the LexSlot results figure (Figure 6.x) for Chapter 6 from MEASURED results.

Two stacked panels on the domain-incremental (DIL) scenario, LayoutLMv3, mean over the
seeds present (error bars = sample std):
  - top:    average accuracy (AA), LexSlot vs the strongest baselines vs the joint oracle
  - bottom: backward transfer (BWT), same methods (closer to 0 = less forgetting)

LexSlot is the proposed isolated, buffer-free slot memory (the ``_off'' config).
Reads results/dil_<method>_seed{42,123,7}/metrics.json and
results/dil_lexslot_seed{42,123,7}_off/metrics.json.

Emits thesis/figures/lexslot_dil_bars.pdf (committed, CI-safe).
Run from the CLUE/ root:  uv run python scripts/build_lexslot_figure.py
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("thesis/figures")
SEEDS = (42, 123, 7)

# (label, results-dir glob stem). LexSlot uses the isolated _off config (the selected method).
METHODS = [
    ("Naive", "dil_naive_seed{s}"),
    ("EWC", "dil_ewc_seed{s}"),
    ("ER", "dil_er_seed{s}"),
    ("DER++", "dil_der_pp_seed{s}"),
    ("C-Flat++", "dil_er_cflat_seed{s}"),
    ("LexSlot (ours)", "dil_lexslot_seed{s}_off"),
    ("Joint (oracle)", "dil_joint_seed{s}"),
]
COL = {
    "Naive": "#9e9e9e",
    "EWC": "#ff9800",
    "ER": "#2196f3",
    "DER++": "#3f51b5",
    "C-Flat++": "#00bcd4",
    "LexSlot (ours)": "#e91e63",  # highlight colour
    "Joint (oracle)": "#4caf50",
}


def _agg(stem: str, metric: str):
    """Mean and sample-std of `metric` over the seeds that have a metrics.json."""
    vals = []
    for s in SEEDS:
        p = Path("results") / stem.format(s=s) / "metrics.json"
        if p.exists():
            v = json.loads(p.read_text()).get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(v))
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def _panel(ax, metric, ylabel, zero_line=False):
    labels = [m[0] for m in METHODS]
    means, errs, cols = [], [], []
    for label, stem in METHODS:
        mu, sd = _agg(stem, metric)
        means.append(mu if mu is not None else float("nan"))
        errs.append(sd if sd is not None else 0.0)
        cols.append(COL[label])
    x = range(len(labels))
    bars = ax.bar(
        x, means, yerr=errs, capsize=3, color=cols, edgecolor="black", linewidth=0.5
    )
    # value labels
    for b, mu in zip(bars, means, strict=False):
        if mu == mu:  # not NaN
            ax.annotate(
                f"{mu:.1f}",
                (b.get_x() + b.get_width() / 2, mu),
                ha="center",
                va="bottom" if mu >= 0 else "top",
                fontsize=8,
                xytext=(0, 2 if mu >= 0 else -2),
                textcoords="offset points",
            )
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (ax_aa, ax_bwt) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    _panel(ax_aa, "AA", "Average accuracy (entity-F1)")
    _panel(ax_bwt, "BWT", "Backward transfer", zero_line=True)
    ax_aa.set_title(
        "Domain-incremental (DIL): LexSlot vs baselines (mean over seeds, $\\pm$std)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG / "lexslot_dil_bars.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    # report the numbers used (provenance / sanity)
    for label, stem in METHODS:
        aa_m, aa_s = _agg(stem, "AA")
        bw_m, bw_s = _agg(stem, "BWT")
        if aa_m is not None:
            n = len(glob.glob(str(Path("results") / stem.format(s="*") / "metrics.json")))
            print(f"  {label:16s} AA={aa_m:6.2f}±{aa_s:4.2f}  BWT={bw_m:6.2f}±{bw_s:4.2f} (n={n})")


if __name__ == "__main__":
    main()
