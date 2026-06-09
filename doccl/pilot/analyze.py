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


def _mannwhitney(a: list[float], b: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value; NaN if scipy missing or degenerate."""
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return float("nan")
    if len(a) < 1 or len(b) < 1 or (len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]):
        return float("nan")
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def condition_bwt_test(results: list[dict], reference: str = "c4_full", alpha: float = 0.05) -> dict | None:
    """H0/H1 across conditions: is the full-multimodal model's forgetting (|BWT|)
    distinguishable from each unimodal/ablated condition?

    Samples per condition = seeds (and alternate task orders, if those runs are
    present — each pilot JSON is one sample). Bonferroni-corrected at
    ``alpha / (#rival conditions)``.
    """
    by_cond: dict[str, list[float]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(abs(r["cl_metrics"]["BWT"]))
    if reference not in by_cond:
        return None
    rivals = sorted(c for c in by_cond if c != reference)
    bonf = alpha / max(len(rivals), 1)
    rows, reject_any = [], False
    for c in rivals:
        p = _mannwhitney(by_cond[reference], by_cond[c])
        sig = (p == p) and p < bonf  # p==p screens NaN
        reject_any = reject_any or sig
        rows.append((c, float(np.mean(by_cond[c])), p, sig))
    return {
        "reference": reference,
        "ref_mean": float(np.mean(by_cond[reference])),
        "alpha": alpha,
        "bonferroni": bonf,
        "rivals": rows,
        "reject_H0": reject_any,
        "n_per_group": len(by_cond[reference]),
    }


def component_hypothesis_test(df: pd.DataFrame, condition: str = "c4_full", alpha: float = 0.05) -> dict | None:
    """H0 (forgetting uniform across components) vs H1 (one component dominates),
    within ``condition``, on the per-group Fisher signal at the final task.

    Identifies the dominant (highest mean Fisher) group and Mann-Whitney-tests it
    against each rival group, pooling across seeds (and alternate orders, if
    present). Bonferroni-corrected at ``alpha / (#groups - 1)``.
    """
    fisher = df[(df["metric"] == "fisher") & (df["condition"] == condition)]
    if fisher.empty:
        return None
    final_task = fisher["task_idx"].max()
    sub = fisher[fisher["task_idx"] == final_task]
    groups = sorted(sub["group"].unique())
    samples = {g: sub[sub["group"] == g]["value"].tolist() for g in groups}
    means = {g: float(np.mean(v)) for g, v in samples.items() if v}
    if not means:
        return None
    dominant = max(means, key=means.get)
    bonf = alpha / max(len(groups) - 1, 1)
    rows, reject_any = [], False
    for g in groups:
        if g == dominant:
            continue
        p = _mannwhitney(samples[dominant], samples[g])
        sig = (p == p) and p < bonf
        reject_any = reject_any or sig
        rows.append((g, means[g], p, sig))
    return {
        "condition": condition,
        "dominant": dominant,
        "dominant_mean": means[dominant],
        "alpha": alpha,
        "bonferroni": bonf,
        "rivals": rows,
        "reject_H0": reject_any,
        "n_per_group": len(samples[dominant]),
    }


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

    # ─── Hypothesis tests (Mann-Whitney U, Bonferroni-corrected) ──────────────
    lines += ["", "## Hypothesis Tests (Mann-Whitney U, Bonferroni-corrected)", ""]

    cond_test = condition_bwt_test(results, reference="c4_full")
    lines += [
        "### Cross-condition: does full-multimodal forgetting differ from C1-C3?",
        "",
        "**H0:** |BWT| of C4 equals that of the ablated/unimodal conditions.  ",
        "**H1:** C4 forgets differently from at least one of C1-C3.",
        "",
    ]
    if cond_test is None:
        lines.append("_No `c4_full` runs found — cannot test._")
    else:
        lines += [
            f"- Reference **{cond_test['reference']}** mean |BWT| = {cond_test['ref_mean']:.2f} "
            f"(n={cond_test['n_per_group']} per group, α={cond_test['alpha']}, "
            f"Bonferroni α'={cond_test['bonferroni']:.4f}).",
            "",
            "| Rival condition | mean \\|BWT\\| | p (vs C4) | significant |",
            "|---|---|---|---|",
        ]
        for c, mean, p, sig in cond_test["rivals"]:
            pstr = "n/a" if p != p else f"{p:.4f}"
            lines.append(f"| {c} | {mean:.2f} | {pstr} | {'**yes**' if sig else 'no'} |")
        verdict = "**reject H0**" if cond_test["reject_H0"] else "fail to reject H0"
        lines += ["", f"Verdict: {verdict} at the Bonferroni-corrected level."]

    comp_test = component_hypothesis_test(df, condition="c4_full")
    lines += [
        "",
        "### Per-component (C4): is forgetting concentrated in one component?",
        "",
        "**H0:** per-group Fisher uniform across components.  ",
        "**H1:** one component (the dominant) differs significantly.",
        "",
    ]
    if comp_test is None:
        lines.append("_No per-group Fisher data for `c4_full` — cannot test._")
    else:
        lines += [
            f"- Dominant component: **{comp_test['dominant']}** "
            f"(mean Fisher = {comp_test['dominant_mean']:.3e}, n={comp_test['n_per_group']}, "
            f"Bonferroni α'={comp_test['bonferroni']:.4f}).",
            "",
            "| Rival group | mean Fisher | p (vs dominant) | significant |",
            "|---|---|---|---|",
        ]
        for g, mean, p, sig in comp_test["rivals"]:
            pstr = "n/a" if p != p else f"{p:.4f}"
            lines.append(f"| {g} | {mean:.3e} | {pstr} | {'**yes**' if sig else 'no'} |")
        verdict = "**reject H0**" if comp_test["reject_H0"] else "fail to reject H0"
        lines += [
            "",
            f"Verdict: {verdict}. ",
            "",
            "_Decision rule (CLAUDE.md): fusion-dominant → Candidate A; layout-position "
            "drift → Candidate B; scenario-dependent → Candidate C; no clear pattern "
            "(fail to reject H0) → characterization-only fallback._",
        ]

    lines += [
        "",
        "## Key Layer/Group Drift (CKA / Fisher)",
        "",
        "See `cka_heatmap.pdf` and `fisher_bars.pdf` in the figures/ directory.",
        "",
        "## Implications for Method Design",
        "",
        "_To be confirmed at the Week-4 advisor meeting (GATE A)._",
    ]

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
