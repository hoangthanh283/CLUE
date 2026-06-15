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
    - table_single_task_baselines.{tex,csv} : single-task naive b_i per dataset (the FWT
                                  baseline reference; see the FWT note below)
    - all_runs.csv, pivot_<metric>.csv

``scripts/ingest_to_thesis.py`` then copies these into ``thesis/generated/`` and
``thesis/figures/`` (committed, CI-safe) and wires them into chapter 6.

Forward transfer (FWT) — IMPORTANT:
    True FWT = mean_{i>0} (R[i-1, i] - b_i) needs the multi-task model's ZERO-SHOT F1
    on each task *before* it is trained (R[i-1, i]). The live CL loop only evaluates
    SEEN tasks, so R[i-1, i] is never measured and the stored matrices are strictly
    lower-triangular. True FWT is therefore NOT recoverable post-hoc and is reported as
    unavailable ("--") here rather than a misleading 0. We do emit the single-task
    baseline table (b_i) so the thesis has the honest upper-reference. See
    docs/FWT_NOTE.md for the exact train.py change that would enable true FWT later.

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

# ─── FWT baseline mapping (see docs/FWT_NOTE.md) ────────────────────────────────
# Per-task underlying *dataset* for each multi-task scenario, in task order. The
# single-task naive baseline b_i for task i is the from-scratch F1 on this dataset,
# read from the corresponding ``single_<dataset>`` run. Derived from the scenario
# builders in ``doccl/data/scenarios.py``:
#   - dil       = build_dil():    funsd → sroie → cord   (TaskInfo.metadata native_dataset)
#   - cil_cord  = build_cil_cord(): 5 CORD sessions       (all share the CORD baseline)
#   - mixed     = build_mixed():  funsd, funsd, sroie, cord, cord, funsd
# Within-dataset sessions (cil_cord, mixed) reuse a single dataset's single-task
# baseline — there is no per-session single-task run.
SCENARIO_TASK_DATASETS: dict[str, list[str]] = {
    "dil": ["funsd", "sroie", "cord"],
    "cil_cord": ["cord", "cord", "cord", "cord", "cord"],
    "mixed": ["funsd", "funsd", "sroie", "cord", "cord", "funsd"],
}
# Single-task baseline run scenario name for each dataset (build_single → SCENARIO_REGISTRY).
SINGLE_SCENARIO = {"funsd": "single_funsd", "cord": "single_cord", "sroie": "single_sroie"}


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


# ─── Single-task baselines & forward transfer (FWT) ─────────────────────────────
#
# Why this is a *baseline table* and NOT a FWT column
# ---------------------------------------------------
# True forward transfer is  FWT = mean_{i>0} (R[i-1, i] - b_i)  where R[i-1, i] is
# the multi-task model's ZERO-SHOT F1 on task i *before* it is trained (one step
# earlier in the sequence) and b_i is the single-task from-scratch baseline on task
# i's dataset (doccl/eval/metrics.py).
#
# The live CL loop (scripts/train.py, "Standard CL loop") only evaluates
# ``eval_loaders_seen`` — tasks 0..current — after each task. The future-task entry
# R[i-1, i] is therefore NEVER measured and the stored accuracy matrix is strictly
# lower-triangular (R[i-1, i] = NaN). The per-run tracker is also constructed without
# ``baseline_perf``, so ``forward_transfer()`` short-circuits to 0.0. As a result
# TRUE zero-shot FWT is **not recoverable post-hoc** from the saved matrices, and we
# refuse to fabricate it: FWT is reported as unavailable ("--") in the aggregates.
#
# What we CAN compute honestly is the single-task baseline table b_i (mean ± std over
# seeds) from the ``single_*`` runs — a "single-task upper-reference" the thesis can
# cite directly and which is exactly the b_i term true FWT would later subtract. See
# docs/FWT_NOTE.md for precisely what train.py change would unlock true FWT.
def compute_single_task_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """b_i per dataset = mean ± std of single-task naive F1 over seeds.

    Reads ``single_<dataset>`` runs (scenario == 'single_<dataset>'); each stores its
    from-scratch single-task F1 as ``AA`` (matrix = [[f1]]). Returns
    ``{dataset: {"mean": ..., "std": ..., "count": ...}}`` for whichever datasets have
    runs. Missing datasets are simply absent (degrade gracefully — never raises).
    """
    out: dict[str, dict[str, float]] = {}
    sub = df[df["state"] == "finished"]
    for dataset, scenario in SINGLE_SCENARIO.items():
        runs = sub[(sub["scenario"] == scenario)].dropna(subset=["AA"])
        if runs.empty:
            continue
        vals = runs["AA"].astype(float)
        out[dataset] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
            "count": int(len(vals)),
        }
    return out


def compute_fwt_per_run(matrix: list | None, scenario: str,
                        baselines: dict[str, dict[str, float]]) -> float:
    """True FWT for one run = mean_{i>0} (R[i-1, i] - b_i).

    Uses the zero-shot upper-triangular entries R[i-1, i] now recorded by train.py
    and the single-task baselines b_i (mean over seeds) mapped through
    SCENARIO_TASK_DATASETS. Returns NaN if the matrix lacks the zero-shot term
    (older lower-triangular runs) or a needed dataset baseline is missing — never
    fabricates a value.
    """
    if matrix is None:
        return float("nan")
    R = np.array(matrix, dtype=float)
    datasets = SCENARIO_TASK_DATASETS.get(scenario)
    if datasets is None or R.ndim != 2 or R.shape[0] < 2:
        return float("nan")
    diffs = []
    for i in range(1, R.shape[0]):
        zs = R[i - 1, i]  # zero-shot on task i before training it
        ds = datasets[i] if i < len(datasets) else None
        b = baselines.get(ds, {}).get("mean") if ds else None
        if not np.isnan(zs) and b is not None:
            diffs.append(zs - b)
    return float(np.mean(diffs)) if diffs else float("nan")


def add_fwt_column(df: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Populate df['FWT'] from saved matrices + single-task baselines (true FWT)."""
    df = df.copy()
    df["FWT"] = [
        compute_fwt_per_run(m, sc, baselines)
        for m, sc in zip(df["matrix"], df["scenario"])
    ]
    return df


def write_baseline_table(df: pd.DataFrame, output: Path) -> dict[str, dict[str, float]]:
    """Emit the single-task baseline table (b_i per dataset) as CSV + LaTeX.

    This is the FWT baseline reference (not FWT itself — see the module note above and
    docs/FWT_NOTE.md). Robust to missing single-task runs: writes an explicit
    placeholder row rather than crashing. Returns the computed baselines so callers can
    reuse them. Writes ``<output>`` (.tex) and a sibling ``.csv``.
    """
    baselines = compute_single_task_baselines(df)
    datasets = [d for d in ("funsd", "cord", "sroie") if d in baselines]

    # CSV (machine-readable, for thesis/generated/ ingestion).
    csv_path = output.with_suffix(".csv")
    csv_rows = [
        {"dataset": d, "single_task_f1_mean": baselines[d]["mean"],
         "single_task_f1_std": baselines[d]["std"], "n_seeds": baselines[d]["count"]}
        for d in datasets
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows, columns=["dataset", "single_task_f1_mean", "single_task_f1_std", "n_seeds"]).to_csv(
        csv_path, index=False
    )

    # LaTeX (single-task upper-reference the thesis can cite).
    lines = [
        "% Auto-generated by scripts/analyze_results.py (do not edit by hand).",
        "% Single-task naive baseline b_i (from-scratch entity-F1, mean $\\pm$ std over seeds).",
        "% This is the FWT baseline reference; true FWT is unavailable (see docs/FWT_NOTE.md).",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Single-task F1} \\\\",
        "\\midrule",
    ]
    if not datasets:
        lines.append("\\multicolumn{2}{c}{\\itshape no single-task baseline runs found} \\\\")
    else:
        for d in datasets:
            b = baselines[d]
            lines.append(f"{d.upper()} & {_cell(b['mean'], b['std'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    output.write_text("\n".join(lines))
    print(f"Wrote {output} and {csv_path}")
    return baselines


def report_fwt_status(df: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> None:
    """Print an explicit, honest FWT-unavailability notice (never a misleading 0).

    True zero-shot FWT cannot be computed from the stored lower-triangular matrices.
    We surface this loudly at aggregate time so the thesis never mistakes a 0/"--"
    placeholder for a measured 0, and show which single-task baselines b_i are ready
    to plug in once true FWT is enabled (see docs/FWT_NOTE.md).
    """
    print("\n=== FWT (Forward Transfer) ===")
    print(
        "FWT is UNAVAILABLE ('--'): the CL loop only evaluates seen tasks, so the\n"
        "future-task zero-shot term R[i-1, i] is never measured (matrices are lower-\n"
        "triangular). True FWT is NOT recoverable post-hoc and is NOT fabricated.\n"
        "See docs/FWT_NOTE.md for the train.py change that would enable it."
    )
    if baselines:
        ready = ", ".join(f"{d}={b['mean']:.2f}" for d, b in sorted(baselines.items()))
        print(f"Single-task baselines b_i ready for future FWT: {ready}")
    else:
        print("No single-task (single_*) baseline runs found yet.")


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
    parser.add_argument("--metrics", nargs="+", default=["AA", "BWT", "AF", "FWT"])
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
    # Compute the single-task baselines b_i first, then derive TRUE FWT per run from
    # the zero-shot upper-triangular term R[i-1, i] (now recorded by train.py) minus
    # b_i. Runs whose matrix lacks the zero-shot term (legacy lower-triangular) get
    # FWT=NaN and are simply excluded from the FWT aggregate — never fabricated.
    baselines = compute_single_task_baselines(df)
    df = add_fwt_column(df, baselines)

    df.drop(columns=["matrix"], errors="ignore").to_csv(args.output_dir / "all_runs.csv", index=False)

    for metric in args.metrics:
        if metric == "FWT" and df["FWT"].isna().all():
            print("\n=== FWT === unavailable (no run has the zero-shot term; "
                  "see docs/FWT_NOTE.md)")
            continue
        agg = aggregate(df, metric=metric)
        print(f"\n=== {metric} ===\n{agg.to_string(index=False)}")
        write_pivot_csv(df, args.output_dir / f"pivot_{metric}.csv", metric=metric)

    write_main_table(df, args.output_dir / "table_main.tex", metric="AA", proposed=args.proposed)
    write_ablation_table(df, args.output_dir / "table_ablation.tex", proposed=args.proposed)
    write_compute_table(df, args.output_dir / "table_compute.tex")
    plot_forgetting_curves(df, args.output_dir / "figure_forgetting_curves.pdf")

    # Single-task baseline reference (the b_i term of FWT). FWT itself is now computed
    # as a real column above when the zero-shot term is present (see add_fwt_column).
    write_baseline_table(df, args.output_dir / "table_single_task_baselines.tex")
    report_fwt_status(df, baselines)


if __name__ == "__main__":
    main()
