"""Per-session retention on CIL-CORD (session-restricted scoring): F1 on session j after each
later session, mean over seeds, for CoLaR (shared softmax), CoLaR-SBR and SBR-Recall. Reads the
same sources, in the same priority order, as make_clean_tables.py.

    uv run python scripts/rca/make_retention_fig.py   ->  paper_cvpr/figures/retention_cilcord.pdf
"""

from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from make_clean_tables import SEEDS, C  # noqa: E402

PANELS = [("colar", "CoLaR"), ("sbr", "CoLaR-SBR"), ("recall", "SBR-Recall")]


def matrices(sources):
    """Retention matrices of the first source with all seeds, else the first non-empty one."""
    per_source = []
    for root, name, sfx in sources:
        mats = []
        for s in SEEDS:
            hits = sorted(glob(f"{root}/{name}_seed{s}{sfx}/**/matrix.npy", recursive=True))
            if hits:
                mats.append(np.load(hits[0]))
        per_source.append(mats)
    full = [m for m in per_source if len(m) == len(SEEDS)]
    return (full or [m for m in per_source if m] or [[]])[0]


def main():
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.45), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, 5))
    for ax, (key, title) in zip(axes, PANELS, strict=True):
        mats = matrices(C[key])
        if mats:
            mean = np.mean(mats, axis=0)
            n = mean.shape[0]
            for j in range(n):
                ax.plot(range(j + 1, n + 1), mean[j:, j], marker="o", ms=2, lw=0.9, color=colors[j])
        ax.set_title(f"{title} (n={len(mats)})", fontsize=6, pad=2)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, lw=0.3)
        ax.tick_params(length=1.5, pad=1)
    axes[0].set_ylabel("F1 on session $j$", labelpad=1)
    axes[1].set_xlabel(
        "sessions trained (one line per session $j$, from when it is learned)", labelpad=1
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.2, wspace=0.08)
    out = Path("paper_cvpr/figures/retention_cilcord.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
