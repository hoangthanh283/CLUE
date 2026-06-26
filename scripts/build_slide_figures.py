"""Regenerate the DocCL_present.pptx chart images to match the current thesis.

Emits PNGs sized to the existing slide frames (so the slide XML geometry is untouched)
into the path given by --out (default: the unzipped pptx media dir). Numbers are the
locked thesis values (generated/table_main.tex, tab:forgetting-matrix, tab:cka-layer,
tab:bert-contrast, and the DIL ablation computed from results/all_runs.csv).

Usage:
    python scripts/build_slide_figures.py --out /tmp/pptx_work/ppt/media
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Colour-blind-friendly palette shared with the thesis backbone figures.
C_LAYOUT = "#0072B2"  # blue   — LayoutLMv3
C_BERT = "#999999"  # grey   — BERT (unimodal)
C_LILT = "#D55E00"  # vermillion — LiLT
C_BROS = "#009E73"  # green  — BROS
C_ACCENT = "#C2185B"  # magenta accent (DocCL / highlights)
C_ORACLE = "#2E7D32"  # oracle green

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


def _save_exact(fig, out: Path, w: int, h: int, dpi: int = 200) -> None:
    """Save then force exact WxH so the PNG drops into the existing frame undistorted."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    Image.open(out).convert("RGB").resize((w, h), Image.LANCZOS).save(out)


def forgetting_matrix(out: Path) -> None:
    """Slide 3 (Hook): single-backbone LayoutLMv3 forgetting matrix, corrected numbers."""
    w, h = 1199, 991
    fig, ax = plt.subplots(figsize=(w / 200, h / 200))
    R = np.array([[89.1, np.nan, np.nan], [3.3, 94.1, np.nan], [1.1, 2.5, 81.4]])
    im = ax.imshow(R, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    rows = ["after FUNSD", "after CORD", "after SROIE"]
    cols = ["eval\nFUNSD", "eval\nCORD", "eval\nSROIE"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows, fontsize=11)
    ax.set_xlabel("Evaluated task $j$", fontsize=11)
    ax.set_ylabel("Model after training task $i$", fontsize=11)
    for i in range(3):
        for j in range(3):
            v = R[i, j]
            if np.isnan(v):
                ax.text(j, i, "not\nseen", ha="center", va="center", color="#888",
                        style="italic", fontsize=10)
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=15,
                        fontweight="bold", color="white" if v < 55 else "black")
    # Hook annotation: FUNSD 89.1 -> 3.3 after one further task.
    ax.annotate("FUNSD: 89.1 → 3.3\nafter one further task", xy=(0, 1), xytext=(1.35, 0.25),
                ha="center", color=C_ACCENT, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.6))
    ax.set_title("Performance matrix $R[i,j]$ — naive sequential fine-tuning\n"
                 "diagonal stays high, retention collapses to $\\sim$1–3 F1",
                 fontsize=12.5, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Entity-F1 on the evaluated task", fontsize=10)
    fig.tight_layout()
    _save_exact(fig, out, w, h)


def cka_gradient(out: Path) -> None:
    """Slide 7 (Finding): 4-backbone CKA depth gradient collapsing onto one curve."""
    w, h = 1469, 935
    fig, ax = plt.subplots(figsize=(w / 200, h / 200))
    depths = ["text+layout\nembed.", "encoder\nL0 (early)", "encoder\nL6 (mid)",
              "encoder\nL11 (late)", "classifier\nhead"]
    x = np.arange(len(depths))
    series = [
        ("LayoutLMv3", [1.000, 0.963, 0.654, 0.246, 0.162], C_LAYOUT, "o", "-"),
        ("BERT (text-only)", [0.999, 0.985, 0.764, 0.287, 0.198], C_BERT, "s", "--"),
        ("LiLT", [0.999, 0.958, 0.453, 0.171, 0.128], C_LILT, "^", "-"),
        ("BROS", [0.993, 0.942, 0.636, 0.181, 0.141], C_BROS, "D", "-"),
    ]
    for name, ys, c, mk, ls in series:
        ax.plot(x, ys, marker=mk, ls=ls, lw=2.4, ms=8, color=c, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("linear CKA  (1.0 = no drift)", fontsize=11.5, style="italic")
    ax.set_xlabel("input encoders → output head (increasing depth)", fontsize=11, style="italic")
    ax.set_title("Per-layer drift (FUNSD→CORD): one curve, four backbones",
                 fontsize=13.5, pad=10)
    ax.annotate("1.00", xy=(0, 1.0), xytext=(0, 1.02), ha="center", color=C_LAYOUT, fontsize=10)
    ax.annotate("0.16", xy=(4, 0.162), xytext=(4, 0.05), ha="center", color=C_LAYOUT,
                fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=11.5, loc="upper right", ncol=2)
    fig.tight_layout()
    _save_exact(fig, out, w, h)


def aa_bars(out: Path) -> None:
    """Slide 12 (Results visualized): AA bars, ER / DER++ / DocCL / Joint.

    No internal legend — the slide supplies its own colour legend (runs 3–6), so the
    four bars/colours mirror the original deck design (C-Flat++ is named in the text).
    """
    w, h = 1425, 742
    fig, ax = plt.subplots(figsize=(w / 200, h / 200))
    scenarios = ["CIL-CORD", "DIL", "Mixed"]
    methods = ["ER", "DER++", "DocCL", "Joint (oracle)"]
    data = {
        "ER": [15.98, 87.85, 60.32],
        "DER++": [18.15, 88.02, 60.82],
        "DocCL": [16.11, 84.71, 57.64],
        "Joint (oracle)": [33.32, 88.73, 67.51],
    }
    # Match the original deck swatches: ER light-blue, DER++ indigo, DocCL magenta, Joint green.
    colors = ["#29ABE2", "#3949AB", "#E91E63", "#43A047"]
    x = np.arange(len(scenarios))
    nb = len(methods)
    width = 0.78 / nb
    for i, (m, c) in enumerate(zip(methods, colors)):
        xs = x + (i - (nb - 1) / 2) * width
        bars = ax.bar(xs, data[m], width, color=c, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, data[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average accuracy (entity-F1)", fontsize=12)
    fig.tight_layout()
    _save_exact(fig, out, w, h)


def ablation_bars(out: Path) -> None:
    """New slide: depth-targeting ablation on DIL — targeting beats uniform."""
    w, h = 1425, 742
    fig, ax = plt.subplots(figsize=(w / 200, h / 200))
    labels = ["Head-only", "Late-only", "Full DocCL", "Uniform\n(all depths)"]
    vals = [86.1, 86.2, 84.7, 42.6]
    colors = [C_ACCENT, C_ACCENT, "#7B1FA2", "#9E9E9E"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center",
                va="bottom", fontsize=14, fontweight="bold")
    ax.axhline(88.7, ls="--", lw=1.6, color=C_ORACLE)
    ax.text(3.4, 89.4, "Joint oracle 88.7", color=C_ORACLE, fontsize=10, ha="right")
    ax.axhline(41.3, ls=":", lw=1.4, color="#777")
    ax.text(3.4, 42.2, "Naive 41.3", color="#777", fontsize=9.5, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average accuracy on DIL (entity-F1)", fontsize=12)
    ax.set_title("Targeting the diagnosed head/late locus vs. spreading the budget uniformly",
                 fontsize=13, pad=10)
    ax.annotate("collapses\n(−half)", xy=(3, 42.6), xytext=(3, 64), ha="center",
                color="#616161", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#616161", lw=1.6))
    fig.tight_layout()
    _save_exact(fig, out, w, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/pptx_work/ppt/media"))
    args = ap.parse_args()
    forgetting_matrix(args.out / "image-3-1.png")
    cka_gradient(args.out / "image-7-1.png")
    aa_bars(args.out / "image-12-1.png")
    ablation_bars(args.out / "image-abl-1.png")
    print(f"Wrote 4 slide figures to {args.out}")


if __name__ == "__main__":
    main()
