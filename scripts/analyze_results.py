"""Aggregate CL runs into the thesis LaTeX result tables and figures.

Primary source is the **local** ``results/<run>/{metrics.json,matrix.npy}`` written
by ``scripts/train.py`` — so aggregation works on an offline Vast.ai box with no
W&B connectivity. W&B is an optional alternative source.

Produces (into ``--output_dir``, default ``results/``):
    - table_main.tex            : Table 6.1 — methods x scenarios, AA mean±std,
                                  best non-oracle bold, significance dagger on DocCL
    - table_ablation.tex        : Table 6.2 — component-targeting ablation (AA, BWT)
    - table_compute.tex         : Table 6.3 — time / trainable params / peak memory
    - figure_forgetting_curves.pdf : Fig 6.3 — avg seen-task accuracy over the sequence
    - all_runs.csv, pivot_<metric>.csv

``scripts/ingest_to_thesis.py`` then copies these into ``thesis/generated/`` and
``thesis/figures/`` (committed, CI-safe) and wires them into chapter 6.

Usage:
    python scripts/analyze_results.py                       # local results/ (default)
    python scripts/analyze_results.py --source wandb --project doccl-aaai2027
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None


# ─── Display maps (match thesis chapter 6 row/column labels) ────────────────────
METHOD_ORDER = [
    "naive", "ewc", "lwf", "er", "der_pp",
    "l2p", "dualprompt", "coda_prompt", "o_lora", "doccl",
]  # 'joint' is rendered separately as the oracle row
METHOD_DISPLAY = {
    "naive": "Naive (lower bound)",
    "joint": "Joint (upper bound)",
    "ewc": "EWC", "lwf": "LwF",
    "er": "ER", "der_pp": "DER++",
    "l2p": "L2P", "dualprompt": "DualPrompt", "coda_prompt": "CODA-Prompt",
    "o_lora": "O-LoRA",
    "doccl": "\\textbf{DocCL (ours)}",
}
SCENARIO_ORDER = ["cil_cord", "dil", "mixed"]
SCENARIO_DISPLAY = {"cil_cord": "CIL-CORD", "dil": "DIL", "mixed": "Mixed"}
COMPONENT_ORDER = ["text", "visual", "layout", "fusion", "uniform"]
COMPONENT_DISPLAY = {
    "text": "Text only", "visual": "Visual only", "layout": "Layout only",
    "fusion": "Fusion only", "uniform": "Uniform (all)",
}


# ─── Sources ────────────────────────────────────────────────────────────────────
def load_local_runs(results_dir: Path) -> pd.DataFrame:
    """Read every ``results/<run>/metrics.json`` into one row per run."""
    rows = []
    for mp in sorted(Path(results_dir).glob("*/metrics.json")):
        with open(mp) as f:
            d = json.load(f)
        rows.append(
            {
                "name": mp.parent.name,
                "state": "finished",
                "method": d.get("method", "?"),
                "scenario": d.get("scenario", "?"),
                "seed": d.get("seed", -1),
                "target_component": d.get("target_component"),
                "AA": d.get("AA"),
                "BWT": d.get("BWT"),
                "AF": d.get("AF"),
                "FWT": d.get("FWT"),
                "trainable_params": d.get("trainable_params"),
                "total_params": d.get("total_params"),
                "mean_time_per_task_s": d.get("mean_time_per_task_s"),
                "peak_gpu_mem_mb": d.get("peak_gpu_mem_mb"),
                "matrix": d.get("matrix"),
            }
        )
    return pd.DataFrame(rows)


def pull_runs(project: str, entity: str | None = None) -> pd.DataFrame:
    """Pull runs from a W&B project (optional alternative to local results)."""
    if wandb is None:
        raise ImportError("Install wandb: pip install wandb")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)
    rows = []
    for r in runs:
        cfg, summary = r.config, r.summary._json_dict
        rows.append(
            {
                "name": r.name,
                "state": r.state,
                "method": cfg.get("method", {}).get("name", "?"),
                "scenario": cfg.get("scenario", {}).get("name", "?"),
                "seed": cfg.get("seed", -1),
                "target_component": cfg.get("method", {}).get("target_component"),
                "AA": summary.get("final/AA"),
                "BWT": summary.get("final/BWT"),
                "AF": summary.get("final/AF"),
                "FWT": summary.get("final/FWT"),
                "trainable_params": summary.get("trainable_params"),
                "total_params": summary.get("total_params"),
                "mean_time_per_task_s": summary.get("mean_time_per_task_s"),
                "peak_gpu_mem_mb": summary.get("peak_gpu_mem_mb"),
                "matrix": None,
            }
        )
    return pd.DataFrame(rows)


# ─── Helpers ────────────────────────────────────────────────────────────────────
def _finished(df: pd.DataFrame, metric: str = "AA") -> pd.DataFrame:
    return df[df["state"] == "finished"].dropna(subset=[metric])


def _cell(mean: float, std: float, bold: bool = False, dagger: bool = False) -> str:
    val = f"{mean:.2f}\\;{{\\scriptsize $\\pm$ {std:.2f}}}"
    if dagger:
        val += "$^{\\dagger}$"
    return f"\\textbf{{{val}}}" if bold else val


def _paired_pvalue(a: list[float], b: list[float]) -> float:
    """Paired Wilcoxon p-value (falls back to paired t); NaN if not computable."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_rel, wilcoxon

        if len(set(np.subtract(a, b).tolist())) <= 1:
            return float("nan")
        try:
            return float(wilcoxon(a, b).pvalue)
        except ValueError:
            return float(ttest_rel(a, b).pvalue)
    except ImportError:
        return float("nan")


def aggregate(df: pd.DataFrame, metric: str = "AA") -> pd.DataFrame:
    """Mean ± std of metric across seeds, indexed by (scenario, method)."""
    df = _finished(df, metric)
    agg = df.groupby(["scenario", "method"])[metric].agg(["mean", "std", "count"]).reset_index()
    agg["mean_std"] = agg.apply(lambda r: f"{r['mean']:.2f}±{r['std']:.2f}", axis=1)
    return agg


# ─── Table 6.1 — main comparison ────────────────────────────────────────────────
def write_main_table(df: pd.DataFrame, output: Path, metric: str = "AA", proposed: str = "doccl") -> None:
    df = _finished(df, metric)
    # Only the un-ablated proposed run (no target_component) belongs in the main table.
    df = df[df["target_component"].isna() | (df["method"] != proposed) | (df["target_component"] == "uniform")]
    mean = df.pivot_table(values=metric, index="method", columns="scenario", aggfunc="mean")
    std = df.pivot_table(values=metric, index="method", columns="scenario", aggfunc="std")

    scenarios = [s for s in SCENARIO_ORDER if s in mean.columns]
    methods = [m for m in METHOD_ORDER if m in mean.index]

    # Best non-oracle per scenario (for bolding) + significance vs best baseline.
    best = {s: mean.loc[methods, s].max() if methods else np.nan for s in scenarios}
    daggers: dict[str, bool] = {}
    if proposed in mean.index:
        for s in scenarios:
            baselines = [m for m in methods if m != proposed]
            if not baselines:
                continue
            best_base = mean.loc[baselines, s].idxmax()
            a = df[(df["method"] == proposed) & (df["scenario"] == s)].sort_values("seed")[metric].tolist()
            b = df[(df["method"] == best_base) & (df["scenario"] == s)].sort_values("seed")[metric].tolist()
            p = _paired_pvalue(a, b)
            daggers[s] = (p == p) and p < 0.05

    col_spec = "l" + "c" * len(scenarios)
    lines = [
        "% Auto-generated by scripts/analyze_results.py — Table 6.1 (do not edit by hand).",
        f"% Metric: {metric} (mean $\\pm$ std over seeds). $\\dagger$: p<0.05 vs best baseline (paired).",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "\\textbf{Method} & " + " & ".join(f"\\textbf{{{SCENARIO_DISPLAY[s]}}}" for s in scenarios) + " \\\\",
        "\\midrule",
    ]

    def row(m: str) -> str:
        cells = []
        for s in scenarios:
            if m not in mean.index or s not in mean.columns or pd.isna(mean.loc[m, s]):
                cells.append("--")
                continue
            mu = float(mean.loc[m, s])
            sd = float(std.loc[m, s]) if not pd.isna(std.loc[m, s]) else 0.0
            is_best = abs(mu - best[s]) < 1e-9 and m != "joint"
            cells.append(_cell(mu, sd, bold=is_best, dagger=(m == proposed and daggers.get(s, False))))
        return f"{METHOD_DISPLAY.get(m, m)} & " + " & ".join(cells) + " \\\\"

    for m in methods:
        lines.append(row(m))
    if "joint" in mean.index:
        lines += ["\\midrule", row("joint")]
    lines += ["\\bottomrule", "\\end{tabular}"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


# ─── Table 6.2 — component-targeting ablation ───────────────────────────────────
def write_ablation_table(df: pd.DataFrame, output: Path, proposed: str = "doccl") -> None:
    df = df[(df["state"] == "finished") & (df["method"] == proposed) & df["target_component"].notna()]
    lines = [
        "% Auto-generated by scripts/analyze_results.py — Table 6.2 (do not edit by hand).",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Targeted component} & \\textbf{AA} & \\textbf{BWT} \\\\",
        "\\midrule",
    ]
    if df.empty:
        lines.append("\\multicolumn{3}{c}{\\itshape no ablation runs found} \\\\")
    else:
        comps = [c for c in COMPONENT_ORDER if c in set(df["target_component"])]
        for c in comps:
            sub = df[df["target_component"] == c]
            aa, aa_s = sub["AA"].mean(), sub["AA"].std(ddof=0)
            bw, bw_s = sub["BWT"].mean(), sub["BWT"].std(ddof=0)
            lines.append(
                f"{COMPONENT_DISPLAY.get(c, c)} & {_cell(aa, aa_s or 0.0)} & {_cell(bw, bw_s or 0.0)} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


# ─── Table 6.3 — computational overhead ─────────────────────────────────────────
def write_compute_table(df: pd.DataFrame, output: Path, methods: list[str] | None = None) -> None:
    methods = methods or ["naive", "ewc", "doccl"]
    df = df[(df["state"] == "finished")]
    df = df[df["target_component"].isna() | (df["target_component"] == "uniform")]
    lines = [
        "% Auto-generated by scripts/analyze_results.py — Table 6.3 (do not edit by hand).",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Time/task (s)} & \\textbf{Trainable params} & \\textbf{Peak mem (MB)} \\\\",
        "\\midrule",
    ]

    def human(n: float) -> str:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "--"
        return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}K"

    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            lines.append(f"{METHOD_DISPLAY.get(m, m)} & -- & -- & -- \\\\")
            continue
        t = sub["mean_time_per_task_s"].mean()
        params = sub["trainable_params"].mean()
        mem = sub["peak_gpu_mem_mb"].mean()
        t_str = "--" if pd.isna(t) else f"{t:.1f}"
        mem_str = "--" if pd.isna(mem) else f"{mem:.0f}"
        lines.append(f"{METHOD_DISPLAY.get(m, m)} & {t_str} & {human(params)} & {mem_str} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


# ─── Figure 6.3 — forgetting curves ─────────────────────────────────────────────
def plot_forgetting_curves(
    df: pd.DataFrame, output: Path, methods: list[str] | None = None, scenario: str | None = None
) -> None:
    import matplotlib.pyplot as plt

    methods = methods or ["naive", "ewc", "der_pp", "doccl"]
    sub = df[df["state"] == "finished"].copy()
    sub = sub[sub["target_component"].isna() | (sub["target_component"] == "uniform")]
    if scenario is None:  # pick the scenario with the most runs
        scenario = sub["scenario"].mode().iloc[0] if not sub.empty else None
    sub = sub[sub["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for m in methods:
        mats = [np.array(x, dtype=float) for x in sub[sub["method"] == m]["matrix"] if x is not None]
        if not mats:
            continue
        avg = np.nanmean(np.stack(mats), axis=0)  # (T, T) averaged over seeds
        # Average accuracy over tasks seen so far, after each training step i.
        curve = [float(np.nanmean(avg[i, : i + 1])) for i in range(avg.shape[0])]
        ax.plot(range(1, len(curve) + 1), curve, marker="o", label=METHOD_DISPLAY.get(m, m))
        plotted = True

    ax.set_xlabel("Tasks trained")
    ax.set_ylabel("Avg. entity-F1 over seen tasks")
    ax.set_title(f"Forgetting curves ({SCENARIO_DISPLAY.get(scenario, scenario)})")
    if plotted:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def write_pivot_csv(df: pd.DataFrame, output: Path, metric: str = "AA") -> None:
    df = _finished(df, metric)
    pivot = df.pivot_table(values=metric, index=["method", "scenario"], columns="seed")
    pivot["mean"], pivot["std"] = pivot.mean(axis=1), pivot.std(axis=1)
    pivot.to_csv(output)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "wandb"], default="local")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--project", default="doccl-aaai2027")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--proposed", default="doccl")
    parser.add_argument("--metrics", nargs="+", default=["AA", "BWT", "AF"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "wandb":
        print(f"Pulling runs from W&B {args.project}...")
        df = pull_runs(args.project, args.entity)
    else:
        print(f"Loading local runs from {args.results_dir}/*/metrics.json ...")
        df = load_local_runs(args.results_dir)
    print(f"Got {len(df)} runs")
    if df.empty:
        print("No runs found. Run the grid first (scripts/run_grid.sh).")
        return
    df.drop(columns=["matrix"], errors="ignore").to_csv(args.output_dir / "all_runs.csv", index=False)

    for metric in args.metrics:
        agg = aggregate(df, metric=metric)
        print(f"\n=== {metric} ===\n{agg.to_string(index=False)}")
        write_pivot_csv(df, args.output_dir / f"pivot_{metric}.csv", metric=metric)

    write_main_table(df, args.output_dir / "table_main.tex", metric="AA", proposed=args.proposed)
    write_ablation_table(df, args.output_dir / "table_ablation.tex", proposed=args.proposed)
    write_compute_table(df, args.output_dir / "table_compute.tex")
    plot_forgetting_curves(df, args.output_dir / "figure_forgetting_curves.pdf")


if __name__ == "__main__":
    main()
