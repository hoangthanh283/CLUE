#!/usr/bin/env python3
"""Generate the benchmark comparison figures for Chapter 6 from MEASURED converged results.

Reads results/*_<method>_seed*/metrics.json (the converged 54-run core + 9 DocCL runs).
Core grid is now complete, so these are the real apples-to-apples figures. Falls back to
results_3ep_archive/ only if a converged cell is missing.

Emits into thesis/figures/ (committed, CI-safe):
  - bench_aa_bars.pdf          : grouped AA bars per core scenario (all methods)
  - bench_bwt_bars.pdf         : grouped BWT bars per core scenario
  - bench_af_bars.pdf          : grouped Average-Forgetting bars per core scenario
  - bench_doccl_vs_replay.pdf  : focused DocCL vs replay vs oracle (AA)
  - bench_dil_closeup.pdf       : DIL near-oracle close-up, AA with oracle line
  - bench_newscenarios_aa.pdf  : AA + BWT on the two new scenarios (WildReceipt CIL,
                                 XFUND cross-lingual DIL) for the classical method set
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
# Core scenarios carry the full method suite (incl. DocCL). The grouped bench_* bars
# stay on these three so the chart is legible; the two NEW scenarios get their own
# figure (they only have classical-method runs).
SC = ["cil_cord", "dil", "mixed"]
SN = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}
# New scenarios + the methods that actually ran on them (no DocCL / 2025 / prompt-LoRA).
SC_NEW = ["cil_wildreceipt", "dil_xlingual"]
SN_NEW = {"cil_wildreceipt": "CIL-WildReceipt", "dil_xlingual": "DIL-XLing (XFUND)"}
NEW_METHODS = ["naive", "ewc", "lwf", "er", "der_pp", "l2p", "dualprompt", "joint"]
# Full reported method set for the core grouped bars (only those present render).
METHODS = [
    "naive",
    "ewc",
    "lwf",
    "er",
    "der_pp",
    "coda_prompt",
    "o_lora",
    "cl_lora",
    "bert_textonly",
    "doccl",
    "joint",
]
MN = {
    "naive": "Naive",
    "ewc": "EWC",
    "lwf": "LwF",
    "er": "ER",
    "der_pp": "DER++",
    "coda_prompt": "CODA-Prompt",
    "o_lora": "O-LoRA",
    "cl_lora": "CL-LoRA",
    "bert_textonly": "BERT",
    "l2p": "L2P",
    "dualprompt": "DualPrompt",
    "doccl": "DocCL",
    "joint": "Joint",
}
COL = {
    "naive": "#9e9e9e",
    "ewc": "#ff9800",
    "lwf": "#cddc39",
    "er": "#2196f3",
    "der_pp": "#3f51b5",
    "coda_prompt": "#00bcd4",
    "o_lora": "#795548",
    "cl_lora": "#8d6e63",
    "bert_textonly": "#607d8b",
    "l2p": "#009688",
    "dualprompt": "#00acc1",
    "doccl": "#e91e63",
    "joint": "#4caf50",
}


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def load(dirs, scenarios):
    """Aggregate AA/BWT/AF per (method, scenario). A naive run on a BERT backbone
    (model_family == 'bert') is surfaced as the separate 'bert_textonly' method, to
    match scripts/analyze_results.py."""
    agg = defaultdict(lambda: defaultdict(list))
    for base in dirs:
        for f in glob.glob(f"{base}/*_seed*/metrics.json"):
            d = json.load(open(f))
            m, s = d.get("method"), d.get("scenario")
            if d.get("model_family") == "bert":
                m = "bert_textonly"
            if s in scenarios:
                for k in ("AA", "BWT", "AF"):
                    if _ok(d.get(k)):
                        agg[(m, s)][k].append(d[k])
    return agg


_ALL_SC = SC + SC_NEW
CONV = load(["results"], _ALL_SC)
ARCH = load(["results_3ep_archive"], _ALL_SC)


def val(m, s, metric):
    if CONV[(m, s)].get(metric):
        return st.mean(CONV[(m, s)][metric])
    if ARCH[(m, s)].get(metric):
        return st.mean(ARCH[(m, s)][metric])
    return None


def _present(methods, scenarios):
    """Keep only methods that have at least one measured cell in the given scenarios."""
    return [m for m in methods if any(val(m, s, "AA") is not None for s in scenarios)]


def grouped(metric, out, ylabel, title, zero_line=False, methods=None, scenarios=None, names=None):
    scenarios = scenarios or SC
    names = names or SN
    methods = _present(methods or METHODS, scenarios)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.8 / max(len(methods), 1)
    x = range(len(scenarios))
    for i, m in enumerate(methods):
        vals = [val(m, s, metric) or 0.0 for s in scenarios]
        offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
        ax.bar(offs, vals, width, label=MN[m], color=COL[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([names[s] for s in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=28)
    if zero_line:
        ax.axhline(0, color="black", linewidth=0.6)
    ncol = min(len(methods), 6) or 1
    ax.legend(
        ncol=ncol, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16)
    )
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
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.6,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
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
    bars = ax.bar(
        [MN[m] for m in order],
        vals,
        color=[COL[m] for m in order],
        edgecolor="black",
        linewidth=0.5,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8
        )
    oracle = val("joint", "dil", "AA")
    ax.axhline(
        oracle,
        color=COL["joint"],
        linestyle="--",
        linewidth=1.3,
        label=f"Joint oracle ({oracle:.1f})",
    )
    ax.set_ylabel("Average accuracy (entity-F1)")
    ax.set_title("Domain-incremental (DIL): DocCL reaches the oracle, replay leads")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "bench_dil_closeup.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'bench_dil_closeup.pdf'}")


def new_scenarios():
    """Two-panel AA + BWT bars for the new scenarios (WildReceipt CIL, XFUND
    cross-lingual DIL). Only classical methods ran here — no DocCL / 2025 / prompt-LoRA."""
    methods = _present(NEW_METHODS, SC_NEW)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, metric, ylabel, zline in (
        (axes[0], "AA", "Average accuracy (entity-F1)", False),
        (axes[1], "BWT", "Backward transfer (BWT)", True),
    ):
        width = 0.8 / max(len(methods), 1)
        x = range(len(SC_NEW))
        for i, m in enumerate(methods):
            vals = [val(m, s, metric) or 0.0 for s in SC_NEW]
            offs = [xi - 0.4 + (i + 0.5) * width for xi in x]
            ax.bar(offs, vals, width, label=MN[m], color=COL[m], edgecolor="black", linewidth=0.4)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SN_NEW[s] for s in SC_NEW], fontsize=9)
        ax.set_ylabel(ylabel)
        if zline:
            ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(
        ncol=min(len(methods), 4) or 1,
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(1.05, 1.2),
    )
    fig.suptitle("Extended scenarios: scale (WildReceipt) and language shift (XFUND)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "bench_newscenarios_aa.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'bench_newscenarios_aa.pdf'}")


def main():
    grouped(
        "AA",
        "bench_aa_bars.pdf",
        "Average accuracy (entity-F1)",
        "Average accuracy by method and scenario",
    )
    grouped(
        "BWT",
        "bench_bwt_bars.pdf",
        "Backward transfer (BWT)",
        "Backward transfer by method and scenario",
        zero_line=True,
    )
    grouped(
        "AF",
        "bench_af_bars.pdf",
        "Average forgetting (AF, lower better)",
        "Average forgetting by method and scenario",
    )
    doccl_vs_replay()
    dil_closeup()
    new_scenarios()


if __name__ == "__main__":
    main()
