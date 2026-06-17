#!/usr/bin/env python3
"""Generate the benchmark comparison figures for Chapter 6 from measured results.

Reads results/*_<method>_seed*/metrics.json (converged grid) for the methods that
have finished, falls back to results_3ep_archive/ for the core-6 where the converged
runs are not in yet, and uses the projected values only for still-unrun methods.

Emits into thesis/figures/ (committed, CI-safe):
  - bench_aa_bars.pdf   : grouped AA bars per scenario (Fig 6.x, regenerated)
  - bench_bwt_bars.pdf  : grouped BWT bars per scenario (regenerated)
  - bench_doccl_vs_replay.pdf : focused DocCL vs replay vs oracle comparison (NEW)

DocCL is now MEASURED (9/9); these figures show the honest result: DocCL reaches the
joint oracle on DIL but does not beat replay.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("thesis/figures")
SC = ["cil_cord", "dil", "mixed"]
SN = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}
# Display order + labels for the full benchmark figures.
METHODS = ["naive", "ewc", "lwf", "er", "der_pp", "doccl", "joint"]
MN = {"naive": "Naive", "ewc": "EWC", "lwf": "LwF", "er": "ER",
      "der_pp": "DER++", "doccl": "DocCL", "joint": "Joint"}
COL = {"naive": "#9e9e9e", "ewc": "#ff9800", "lwf": "#cddc39", "er": "#2196f3",
       "der_pp": "#3f51b5", "doccl": "#e91e63", "joint": "#4caf50"}

# Projected fallback (only for methods with no measured run anywhere).
PROJ_AA = {("naive", "cil_cord"): 19.4, ("naive", "dil"): 39.9, ("naive", "mixed"): 31.9,
           ("joint", "cil_cord"): 32.4, ("joint", "dil"): 86.3, ("joint", "mixed"): 64.8}


def load(dirs):
    agg = defaultdict(lambda: defaultdict(list))
    for base in dirs:
        for f in glob.glob(f"{base}/*_seed*/metrics.json"):
            d = json.load(open(f))
            m, s = d.get("method"), d.get("scenario")
            if s in SC:
                for k in ("AA", "BWT"):
                    if d.get(k) is not None:
                        agg[(m, s)][k].append(d[k])
    return agg


CONV = load(["results"])
ARCH = load(["results_3ep_archive"])


def val(m, s, metric):
    """Prefer measured-converged, then 3-epoch archive, then projected (AA only)."""
    if CONV[(m, s)].get(metric):
        return st.mean(CONV[(m, s)][metric]), "conv"
    if ARCH[(m, s)].get(metric):
        return st.mean(ARCH[(m, s)][metric]), "3ep"
    if metric == "AA" and (m, s) in PROJ_AA:
        return PROJ_AA[(m, s)], "proj"
    return None, None


def grouped_bars(metric: str, out: str, ylabel: str, title: str, zero_line=False):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    n_m = len(METHODS)
    width = 0.8 / n_m
    x = range(len(SC))
    for i, m in enumerate(METHODS):
        vals, srcs = [], []
        for s in SC:
            v, src = val(m, s, metric)
            vals.append(v if v is not None else 0.0)
            srcs.append(src)
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=MN[m], color=COL[m],
                      edgecolor="black", linewidth=0.4)
        # hatch any cell that fell back to projected (none expected now for these methods)
        for b, src in zip(bars, srcs):
            if src == "proj":
                b.set_hatch("///")
    ax.set_xticks(list(x))
    ax.set_xticklabels([SN[s] for s in SC])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(ncol=4, fontsize=8, loc="lower center" if zero_line else "upper center",
              bbox_to_anchor=(0.5, 1.02 if not zero_line else -0.18), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / out}")


def doccl_vs_replay():
    """Focused figure: DocCL vs the best replay (ER/DER++) vs Joint oracle, AA per scenario."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = ["ER", "DER++", "DocCL", "Joint (oracle)"]
    keys = ["er", "der_pp", "doccl", "joint"]
    cols = [COL["er"], COL["der_pp"], COL["doccl"], COL["joint"]]
    width = 0.8 / len(groups)
    x = range(len(SC))
    for i, (g, k, c) in enumerate(zip(groups, keys, cols)):
        vals = [val(k, s, "AA")[0] or 0.0 for s in SC]
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=g, color=c, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SN[s] for s in SC])
    ax.set_ylabel("Average accuracy (entity-F1)")
    ax.set_title("DocCL vs.\\ replay vs.\\ oracle (measured)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "bench_doccl_vs_replay.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'bench_doccl_vs_replay.pdf'}")


def main():
    grouped_bars("AA", "bench_aa_bars.pdf",
                 "Average accuracy (entity-F1)",
                 "Average accuracy by method and scenario")
    grouped_bars("BWT", "bench_bwt_bars.pdf",
                 "Backward transfer (BWT)",
                 "Backward transfer by method and scenario", zero_line=True)
    doccl_vs_replay()


if __name__ == "__main__":
    main()
