"""Cross-backbone comparison figures for the forgetting diagnosis.

The per-condition pilot figures (cka_heatmap, fisher_bars, displacement_bars) contain
every condition but are laid out as one panel/group per condition, which buries the
backbone-vs-backbone story. This script produces three figures whose explicit purpose
is the cross-architecture comparison that the thesis claims ("forgetting is located in
the same place on LayoutLMv3, BERT, LiLT and BROS"):

  1. backbone_cka_gradient.pdf   — CKA vs normalised depth, one line per backbone.
                                    All collapse onto the same monotonic late+head curve.
  2. backbone_head_dominance.pdf — head vs max-non-head Fisher-weighted displacement,
                                    grouped bars, log scale. Head dominates on every one.
  3. backbone_metrics.pdf        — AA and BWT per backbone (paired bars, mean±std).

Reads the raw results/pilot/*_seed*.json directly (NOT aggregated.parquet, which pools
the reverse-order *_ord210 c4 runs into the LayoutLMv3 condition and would skew the
LayoutLMv3 bars/line). Canonical task order only: drops *_ord210 reverse-order runs and
degenerate all-zero runs, so every number matches the thesis cross-backbone tables.
Writes to results/pilot/figures/.

Usage:
    uv run python scripts/build_backbone_figures.py
    uv run python scripts/ingest_to_thesis.py       # copy into thesis/figures/
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The four backbones we compare (condition key -> display label). c4_full is the
# canonical full multimodal LayoutLMv3; the masked variants are the modality study,
# not the backbone study, so they are excluded from these cross-backbone figures.
BACKBONES = {
    "c4_full": "LayoutLMv3",
    "cb_bert": "BERT",
    "cl_lilt": "LiLT",
    "cr_bros": "BROS",
}
# Distinct, colour-blind-friendly palette, stable across all three figures.
COLORS = {
    "c4_full": "#0072B2",  # blue
    "cb_bert": "#999999",  # grey (unimodal baseline)
    "cl_lilt": "#D55E00",  # vermillion
    "cr_bros": "#009E73",  # green
}
# Normalised depth axis shared across backbones (different layer-name schemes map here).
DEPTH_ORDER = ["embeddings", "early (L0)", "mid (L6)", "late (L11)", "head"]


def _depth_of(layer: str) -> str | None:
    """Map a per-backbone CKA layer name to a normalised depth position."""
    if layer.endswith("classifier"):
        return "head"
    if layer.endswith("embeddings"):
        return "embeddings"
    if layer.endswith("patch_embed"):
        return None  # vision-only stream; not a shared depth point
    if "encoder.layer." in layer:
        n = int(layer.rsplit(".", 1)[1])
        if n == 0:
            return "early (L0)"
        if n == 6:
            return "mid (L6)"
        if n == 11:
            return "late (L11)"
    return None


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["condition"].isin(BACKBONES)]


def load_canonical() -> pd.DataFrame:
    """Long-format (condition, seed, metric, boundary, layer, group, value) from the
    canonical-order pilot JSONs only (excludes *_ord210 reverse-order + dead runs)."""
    rows: list[dict] = []
    for f in glob.glob("results/pilot/*_seed*.json"):
        if "ord210" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        cond = d["condition"]
        if cond not in BACKBONES:
            continue
        if d["cl_metrics"]["AA"] == 0 and d["cl_metrics"]["BWT"] == 0:
            continue  # degenerate/dead run
        seed = d["seed"]
        for rec in d.get("cka_records", []):
            for layer, v in rec["cka"].items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "cka",
                        "boundary": rec["task_boundary"],
                        "layer": layer,
                        "group": None,
                        "value": v,
                    }
                )
        for rec in d.get("displacement_records", []):
            for grp, v in rec.get("by_depth", {}).items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "displacement_depth",
                        "boundary": rec["task_boundary"],
                        "layer": None,
                        "group": grp,
                        "value": v,
                    }
                )
    return pd.DataFrame(rows)


def plot_cka_gradient(df: pd.DataFrame, out: Path) -> None:
    """Overlaid CKA-vs-depth, one line per backbone (boundary 0->1, mean over seeds)."""
    cka = _canonical(df[(df["metric"] == "cka") & (df["boundary"] == "0_to_1")]).copy()
    cka["depth"] = cka["layer"].map(_depth_of)
    cka = cka.dropna(subset=["depth"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = list(range(len(DEPTH_ORDER)))
    for cond, label in BACKBONES.items():
        sub = cka[cka["condition"] == cond]
        if sub.empty:
            continue
        m = sub.groupby("depth")["value"].mean()
        ys = [m.get(d, np.nan) for d in DEPTH_ORDER]
        ax.plot(xs, ys, marker="o", lw=2, label=label, color=COLORS[cond])
    ax.set_xticks(xs)
    ax.set_xticklabels(DEPTH_ORDER, rotation=15)
    ax.set_ylabel("Linear CKA  (1.0 = no drift)")
    ax.set_xlabel("Network depth")
    ax.set_title("Representational drift by depth, per backbone (FUNSD$\\rightarrow$CORD)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls=":", c="0.7", lw=1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Backbone", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_head_dominance(df: pd.DataFrame, out: Path) -> None:
    """Head vs max-non-head Fisher-weighted displacement, grouped bars, log scale."""
    dd = _canonical(df[df["metric"] == "displacement_depth"]).copy()
    # Sum displacement over both boundaries, mean over seeds, per (condition, depth).
    agg = dd.groupby(["condition", "seed", "group"])["value"].sum().reset_index()
    agg = agg.groupby(["condition", "group"])["value"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conds = [c for c in BACKBONES if c in set(agg["condition"])]
    x = np.arange(len(conds))
    w = 0.38
    head_vals, rest_vals = [], []
    for c in conds:
        sub = agg[agg["condition"] == c].set_index("group")["value"]
        head_vals.append(sub.get("head", np.nan))
        rest_vals.append(max(v for g, v in sub.items() if g != "head"))
    ax.bar(x - w / 2, head_vals, w, label="Classifier head", color="#D55E00")
    ax.bar(x + w / 2, rest_vals, w, label="Largest non-head bucket", color="#56B4E9")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONES[c] for c in conds])
    ax.set_ylabel("Fisher-weighted displacement (log)")
    ax.set_title("Forgetting concentrates at the head on every backbone")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(frameon=False)
    # Annotate the head:rest ratio above each pair.
    for i, (h, r) in enumerate(zip(head_vals, rest_vals, strict=False)):
        if r and r > 0:
            ax.text(i, h * 1.4, f"{h / r:.0f}$\\times$", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _load_cl_metrics() -> dict[str, dict[str, tuple[float, float]]]:
    """Mean/std of AA and BWT per backbone from canonical-order pilot JSONs."""
    runs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in glob.glob("results/pilot/*_seed*.json"):
        if "ord210" in f:
            continue
        with open(f) as fh:
            d = json.load(fh)
        c = d["condition"]
        if c not in BACKBONES:
            continue
        m = d["cl_metrics"]
        if m["AA"] == 0 and m["BWT"] == 0:
            continue
        runs[c]["AA"].append(m["AA"])
        runs[c]["BWT"].append(m["BWT"])
    return {
        c: {k: (float(np.mean(v)), float(np.std(v))) for k, v in mv.items()}
        for c, mv in runs.items()
    }


def plot_metrics(out: Path) -> None:
    """AA and BWT per backbone (paired bars, mean±std)."""
    met = _load_cl_metrics()
    conds = [c for c in BACKBONES if c in met]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(conds))
    labels = [BACKBONES[c] for c in conds]
    cols = [COLORS[c] for c in conds]
    aa = [met[c]["AA"][0] for c in conds]
    aae = [met[c]["AA"][1] for c in conds]
    bwt = [met[c]["BWT"][0] for c in conds]
    bwte = [met[c]["BWT"][1] for c in conds]
    axa.bar(x, aa, yerr=aae, color=cols, capsize=4)
    axa.set_xticks(x)
    axa.set_xticklabels(labels, rotation=15)
    axa.set_ylabel("Average Accuracy (AA)")
    axa.set_title("Final accuracy after the sequence")
    axa.grid(True, axis="y", alpha=0.3)
    axb.bar(x, bwt, yerr=bwte, color=cols, capsize=4)
    axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=15)
    axb.set_ylabel("Backward Transfer (BWT)")
    axb.set_title("Forgetting (more negative = worse)")
    axb.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Naive sequential FUNSD$\\rightarrow$CORD$\\rightarrow$SROIE, per backbone")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    df = load_canonical()
    if df.empty:
        raise SystemExit("No canonical-order pilot JSONs found in results/pilot/.")
    figdir = Path("results/pilot/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    plot_cka_gradient(df, figdir / "backbone_cka_gradient.pdf")
    plot_head_dominance(df, figdir / "backbone_head_dominance.pdf")
    plot_metrics(figdir / "backbone_metrics.pdf")
    print("Done. Cross-backbone figures in results/pilot/figures/.")


if __name__ == "__main__":
    main()
