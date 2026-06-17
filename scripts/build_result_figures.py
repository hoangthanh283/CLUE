#!/usr/bin/env python3
"""Generate the benchmark comparison figures for Chapter 6 from MEASURED converged results.

Reads results/*_<method>_seed*/metrics.json (the converged 54-run core + 9 DocCL runs).
Core grid is now complete, so these are the real apples-to-apples figures. Falls back to
results_3ep_archive/ only if a converged cell is missing.

Emits into thesis/figures/ (committed, CI-safe):
  - bench_aa_bars.pdf          : grouped AA bars per scenario (all methods)
  - bench_bwt_bars.pdf         : grouped BWT bars per scenario
  - bench_af_bars.pdf          : grouped Average-Forgetting bars per scenario (NEW)
  - bench_doccl_vs_replay.pdf  : focused DocCL vs replay vs oracle (AA)
  - bench_dil_closeup.pdf       : DIL near-oracle close-up, AA with oracle line (NEW)
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("thesis/figures")
SC = ["cil_cord", "dil", "mixed"]
SN = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}
METHODS = ["naive", "ewc", "lwf", "er", "der_pp", "doccl", "joint"]
MN = {"naive": "Naive", "ewc": "EWC", "lwf": "LwF", "er": "ER",
      "der_pp": "DER++", "doccl": "DocCL", "joint": "Joint"}
COL = {"naive": "#9e9e9e", "ewc": "#ff9800", "lwf": "#cddc39", "er": "#2196f3",
       "der_pp": "#3f51b5", "doccl": "#e91e63", "joint": "#4caf50"}


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def load(dirs):
    agg = defaultdict(lambda: defaultdict(list))
    for base in dirs:
        for f in glob.glob(f"{base}/*_seed*/metrics.json"):
            d = json.load(open(f))
            m, s = d.get("method"), d.get("scenario")
            if s in SC:
                for k in ("AA", "BWT", "AF"):
                    if _ok(d.get(k)):
                        agg[(m, s)][k].append(d[k])
    return agg


CONV = load(["results"])
ARCH = load(["results_3ep_archive"])


def val(m, s, metric):
    if CONV[(m, s)].get(metric):
        return st.mean(CONV[(m, s)][metric])
    if ARCH[(m, s)].get(metric):
        return st.mean(ARCH[(m, s)][metric])
    return None


def grouped(metric, out, ylabel, title, zero_line=False, methods=None):
    methods = methods or METHODS
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.8 / len(methods)
    x = range(len(SC))
    for i, m in enumerate(methods):
        vals = [val(m, s, metric) or 0.0 for s in SC]
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        ax.bar(offs, vals, width, label=MN[m], color=COL[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SN[s] for s in SC])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(ncol=len(methods), fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / out}")


def doccl_vs_replay():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    groups = [("ER", "er"), ("DER++", "der_pp"), ("DocCL", "doccl"), ("Joint (oracle)", "joint")]
    width = 0.8 / len(groups)
    x = range(len(SC))
    for i, (g, k) in enumerate(groups):
        vals = [val(k, s, "AA") or 0.0 for s in SC]
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        bars = ax.bar(offs, vals, width, label=g, color=COL[k], edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SN[s] for s in SC])
    ax.set_ylabel("Average accuracy (entity-F1)")
    ax.set_title("DocCL vs.\\ replay vs.\\ oracle (measured)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "bench_doccl_vs_replay.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'bench_doccl_vs_replay.pdf'}")


def dil_closeup():
    """DIL-only close-up: AA bars with the joint-oracle line, showing DocCL reaches it."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    order = ["naive", "lwf", "ewc", "doccl", "er", "der_pp"]
    vals = [val(m, "dil", "AA") or 0.0 for m in order]
    bars = ax.bar([MN[m] for m in order], vals, color=[COL[m] for m in order],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center",
                va="bottom", fontsize=8)
    oracle = val("joint", "dil", "AA")
    ax.axhline(oracle, color=COL["joint"], linestyle="--", linewidth=1.3,
               label=f"Joint oracle ({oracle:.1f})")
    ax.set_ylabel("Average accuracy (entity-F1)")
    ax.set_title("Domain-incremental (DIL): DocCL reaches the oracle, replay leads")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "bench_dil_closeup.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'bench_dil_closeup.pdf'}")


def main():
    grouped("AA", "bench_aa_bars.pdf", "Average accuracy (entity-F1)",
            "Average accuracy by method and scenario")
    grouped("BWT", "bench_bwt_bars.pdf", "Backward transfer (BWT)",
            "Backward transfer by method and scenario", zero_line=True)
    grouped("AF", "bench_af_bars.pdf", "Average forgetting (AF, lower better)",
            "Average forgetting by method and scenario")
    doccl_vs_replay()
    dil_closeup()


if __name__ == "__main__":
    main()
