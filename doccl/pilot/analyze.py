"""Pilot study analysis: aggregate results across conditions and seeds.

Inputs: results/pilot/{condition}_{seed}.json files (from run_pilot.py)

Outputs:
    - results/pilot/aggregated.parquet : long-format DataFrame for all metrics
    - results/pilot/figures/cka_heatmap.pdf : per-condition CKA heatmap
    - results/pilot/figures/fisher_bars.pdf : per-condition Fisher per-group
    - results/pilot/figures/forgetting_matrix.pdf : per-condition forgetting matrix
    - results/pilot/findings_summary.md : Markdown summary of patterns
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_pilot_results(pilot_dir: Path) -> list[dict]:
    """Load all pilot result JSONs."""
    results = []
    for fp in sorted(pilot_dir.glob("*_seed*.json")):
        with open(fp) as f:
            results.append(json.load(f))
    return results


def to_long_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert pilot results into long-format DataFrame.

    Columns: condition, seed, task_idx, layer (or group), metric, value
    """
    rows = []
    for r in results:
        cond = r["condition"]
        seed = r["seed"]

        # CKA records
        for cka_rec in r["cka_records"]:
            boundary = cka_rec["task_boundary"]
            for layer, cka_val in cka_rec["cka"].items():
                rows.append({
                    "condition": cond,
                    "seed": seed,
                    "metric": "cka",
                    "boundary": boundary,
                    "layer": layer,
                    "value": cka_val,
                })

        # Fisher records
        for fisher_rec in r["fisher_records"]:
            tidx = fisher_rec["task_idx"]
            for group, val in fisher_rec["fisher_per_group"].items():
                rows.append({
                    "condition": cond,
                    "seed": seed,
                    "metric": "fisher",
                    "task_idx": tidx,
                    "group": group,
                    "value": val,
                })

        # Accuracy records
        for acc_rec in r["accuracy_records"]:
            tidx = acc_rec["task_idx"]
            for tid, m in acc_rec["results"].items():
                rows.append({
                    "condition": cond,
                    "seed": seed,
                    "metric": "f1",
                    "task_idx": tidx,
                    "evaluated_task": int(tid),
                    "value": m["f1"],
                })

    return pd.DataFrame(rows)


def plot_cka_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot CKA heatmap per condition: rows=layers, cols=task boundaries."""
    cka_df = df[df["metric"] == "cka"].copy()
    if cka_df.empty:
        print("No CKA data — skipping heatmap")
        return

    # Average across seeds
    pivot = cka_df.groupby(["condition", "layer", "boundary"])["value"].mean().reset_index()
    conditions = sorted(pivot["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = pivot[pivot["condition"] == cond]
        mat = sub.pivot(index="layer", columns="boundary", values="value")
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "CKA"})
        ax.set_title(cond)
        ax.set_xlabel("Task boundary")
        ax.set_ylabel("Layer")

    fig.suptitle("Per-layer CKA between consecutive task checkpoints")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_fisher_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of Fisher per parameter group, per condition."""
    fisher_df = df[df["metric"] == "fisher"].copy()
    if fisher_df.empty:
        print("No Fisher data — skipping bars")
        return

    # Average across seeds, take final task only for clarity
    final_task = fisher_df["task_idx"].max()
    sub = fisher_df[fisher_df["task_idx"] == final_task]
    agg = sub.groupby(["condition", "group"])["value"].agg(["mean", "std"]).reset_index()

    conditions = sorted(agg["condition"].unique())
    groups = sorted(agg["group"].unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    width = 0.8 / len(conditions)
    for i, cond in enumerate(conditions):
        sub_c = agg[agg["condition"] == cond].set_index("group").reindex(groups)
        ax.bar(
            x + i * width - 0.4,
            sub_c["mean"].fillna(0),
            width,
            yerr=sub_c["std"].fillna(0),
            label=cond,
            capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("Mean Fisher information")
    ax.set_title(f"Fisher information per parameter group (after final task t={final_task})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_forgetting_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Plot forgetting matrix (R[i,j] = perf on task j after training task i) per condition."""
    f1_df = df[df["metric"] == "f1"].copy()
    if f1_df.empty:
        print("No F1 data — skipping matrix")
        return

    agg = f1_df.groupby(["condition", "task_idx", "evaluated_task"])["value"].mean().reset_index()
    conditions = sorted(agg["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = agg[agg["condition"] == cond]
        mat = sub.pivot(index="task_idx", columns="evaluated_task", values="value")
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax,
                    cbar_kws={"label": "F1"})
        ax.set_title(cond)
        ax.set_xlabel("Evaluated task")
        ax.set_ylabel("After training task")

    fig.suptitle("Forgetting matrix: F1 on task j after training task i")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def write_findings_summary(df: pd.DataFrame, results: list[dict], output_path: Path) -> None:
    """Generate a markdown summary of pilot findings."""
    lines = [
        "# Pilot Study Findings — Auto-generated Summary",
        "",
        "## CL Metrics per Condition (mean ± std across seeds)",
        "",
        "| Condition | AA | BWT | AF | FWT |",
        "|---|---|---|---|---|",
    ]

    by_cond: dict[str, list[dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r["cl_metrics"])

    for cond in sorted(by_cond.keys()):
        metrics_list = by_cond[cond]
        means = {k: np.mean([m[k] for m in metrics_list]) for k in ["AA", "BWT", "AF", "FWT"]}
        stds = {k: np.std([m[k] for m in metrics_list]) for k in ["AA", "BWT", "AF", "FWT"]}
        lines.append(
            f"| {cond} | "
            + " | ".join(f"{means[k]:.2f}±{stds[k]:.2f}" for k in ["AA", "BWT", "AF", "FWT"])
            + " |"
        )

    lines.extend([
        "",
        "## Hypothesis Test (informal)",
        "",
        "**H0:** Forgetting (BWT magnitude) uniform across conditions.  ",
        "**H1:** C4 (full multimodal) shows significantly different forgetting pattern than C1-C3.",
        "",
        "_Run statistical tests separately (e.g., Mann-Whitney U on BWT)._",
        "",
        "## Key Layer/Group Drift (CKA / Fisher)",
        "",
        "See `cka_heatmap.pdf` and `fisher_bars.pdf` in the figures/ directory.",
        "",
        "## Implications for Method Design",
        "",
        "_To be filled in based on findings — see Week 4 advisor meeting decision._",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", type=Path, default=Path("results/pilot"))
    parser.add_argument("--figures_dir", type=Path, default=Path("results/pilot/figures"))
    args = parser.parse_args()

    print(f"Loading pilot results from {args.pilot_dir}")
    results = load_pilot_results(args.pilot_dir)
    if not results:
        print("No pilot results found. Run pilot first: python -m doccl.pilot.run_pilot")
        return

    print(f"Loaded {len(results)} pilot runs")
    df = to_long_dataframe(results)
    df.to_parquet(args.pilot_dir / "aggregated.parquet")
    print(f"Saved long-format data: {args.pilot_dir / 'aggregated.parquet'} ({len(df)} rows)")

    plot_cka_heatmap(df, args.figures_dir / "cka_heatmap.pdf")
    plot_fisher_bars(df, args.figures_dir / "fisher_bars.pdf")
    plot_forgetting_matrix(df, args.figures_dir / "forgetting_matrix.pdf")
    write_findings_summary(df, results, args.pilot_dir / "findings_summary.md")

    print("\nDone. Review figures in", args.figures_dir)


if __name__ == "__main__":
    main()
