#!/usr/bin/env python3
"""Redraw Figure 6.2 — the forgetting-matrix heatmap R[i,j], publication quality.

R[i,j] = entity-F1 on task j after training task i (C4 full model, seed 42, naive
sequential fine-tuning over FUNSD -> CORD -> SROIE). Source: results/pilot/c4_full_seed42.json.

Design goals (professional + convincing):
  - perceptually-uniform colormap (viridis), 0-100 fixed scale so colour = absolute F1;
  - lower-triangular matrix; un-evaluated future tasks (upper triangle) shown as hatched
    grey "not seen yet" cells, not blank/NaN;
  - every cell annotated with its F1, white/black text auto-chosen for contrast;
  - the DIAGONAL (just-learned) outlined to draw the eye to "learned well";
  - a thin arrow/annotation highlighting the collapse from diagonal to the cell below it;
  - clean task labels on both axes (train step rows, eval task columns) and a colourbar.

Emits thesis/figures/diag_forgetting_matrix.pdf (overwrites the old static figure).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SRC = Path("results/pilot/c4_full_seed42.json")
OUT = Path("thesis/figures/diag_forgetting_matrix.pdf")
TASKS = ["FUNSD", "CORD", "SROIE"]  # task_order [0,1,2]


def load_matrix() -> np.ndarray:
    if SRC.exists():
        M = np.array(json.load(open(SRC))["matrix"], dtype=float)
    else:  # fall back to the values quoted in the thesis table
        M = np.array([[88.0, np.nan, np.nan],
                      [1.8, 93.0, np.nan],
                      [1.9, 4.1, 80.5]])
    return M


def main() -> None:
    M = load_matrix()
    n = M.shape[0]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    # Heatmap of the measured (lower-triangular) cells on a fixed 0-100 scale.
    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("viridis").copy()
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    # Colourbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Entity-F1 on the evaluated task", fontsize=10)

    # Upper triangle (task not yet seen) → hatched grey "not seen" cells.
    for i in range(n):
        for j in range(n):
            if math.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e8e8e8",
                                       edgecolor="white", hatch="////", linewidth=0))
                ax.text(j, i, "not\nseen", ha="center", va="center", fontsize=7.5,
                        color="#9a9a9a", style="italic")
            else:
                # Annotate the F1; white text on dark cells, black on light.
                val = M[i, j]
                txt_col = "white" if val < 55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_col)

    # Outline the diagonal (just-learned) cells to say "learned well".
    for d in range(n):
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="#ffffff", linewidth=2.4))
        ax.add_patch(Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                               edgecolor="#222222", linewidth=0.8, linestyle=(0, (3, 2))))

    # Highlight the collapse: arrow from the diagonal FUNSD cell to the cell below it,
    # with the call-out placed in the clear upper-right area (no cell overlap).
    ax.annotate("", xy=(-0.02, 0.78), xytext=(-0.02, 0.22),
                arrowprops=dict(arrowstyle="-|>", color="#d32f2f", lw=2.4,
                                shrinkA=10, shrinkB=10))
    ax.annotate("FUNSD: $88.0 \\rightarrow 1.8$\nafter one further task",
                xy=(0.05, 0.9), xytext=(1.15, 0.35),
                fontsize=8.8, color="#d32f2f", fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#d32f2f", lw=0.8, alpha=0.6))

    # Axes: rows = training step, columns = evaluated task.
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"eval\n{t}" for t in TASKS], fontsize=10)
    ax.set_yticklabels([f"after {t}" for t in TASKS], fontsize=10)
    ax.set_xlabel("Evaluated task  $j$", fontsize=11)
    ax.set_ylabel("Model after training task  $i$", fontsize=11)
    ax.set_title("Performance matrix $R[i,j]$ — naive sequential fine-tuning\n"
                 "diagonal stays high, retention collapses to $\\sim$2--4 F1",
                 fontsize=11.5, pad=10)

    # Light gridlines between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}  (matrix from {'pilot data' if SRC.exists() else 'table fallback'})")


if __name__ == "__main__":
    main()
