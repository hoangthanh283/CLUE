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
import re
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def min_achievable_p(n1: int, n2: int) -> float:
    """Smallest two-sided Mann–Whitney U p-value achievable with group sizes
    ``n1``, ``n2`` (under perfect separation).

    Equals ``2 / C(n1+n2, n2)``. If this floor exceeds the (Bonferroni-corrected)
    significance threshold, the test *cannot* reject H0 regardless of the data —
    the comparison is underpowered by construction and a non-rejection is a
    foregone conclusion of the sample size, not evidence (review C3).
    """
    if n1 < 1 or n2 < 1:
        return float("nan")
    return 2.0 / comb(n1 + n2, n2)


def _is_dead_run(r: dict) -> bool:
    """A pilot run is dead/degenerate if its CL metrics are all-zero.

    The legacy ``c1_bert`` prototype (June-11) and a handful of seeds collapsed to
    F1=0 on every task (AA=BWT=AF=0). These pollute the cross-condition tables and
    must be dropped, not aggregated. A genuine run always has AA>0 on at least the
    last (current) task. We keep the honest minimum: drop only when *all* of
    AA/BWT/AF are exactly zero.
    """
    m = r.get("cl_metrics", {})
    aa, bwt, af = m.get("AA", 0.0) or 0.0, m.get("BWT", 0.0) or 0.0, m.get("AF", 0.0) or 0.0
    return abs(aa) < 1e-9 and abs(bwt) < 1e-9 and abs(af) < 1e-9


def load_pilot_results(pilot_dir: Path) -> list[dict]:
    """Load all pilot result JSONs, dropping dead/degenerate (all-zero-metric) runs."""
    results = []
    for fp in sorted(pilot_dir.glob("*_seed*.json")):
        with open(fp) as f:
            r = json.load(f)
        if _is_dead_run(r):
            print(f"[pilot-analyze] dropping dead run (all-zero metrics): {fp.name}")
            continue
        results.append(r)
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
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "cka",
                        "boundary": boundary,
                        "layer": layer,
                        "value": cka_val,
                    }
                )

        # Fisher records (importance level to the current task)
        for fisher_rec in r["fisher_records"]:
            tidx = fisher_rec["task_idx"]
            for group, val in fisher_rec["fisher_per_group"].items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "fisher",
                        "task_idx": tidx,
                        "group": group,
                        "value": val,
                    }
                )

        # Displacement records (old-task-Fisher-weighted movement = forgetting)
        for disp_rec in r.get("displacement_records", []):
            boundary = disp_rec["task_boundary"]
            for group, val in disp_rec.get("by_group", {}).items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "displacement",
                        "boundary": boundary,
                        "group": group,
                        "value": val,
                    }
                )
            for bucket, val in disp_rec.get("by_depth", {}).items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "displacement_depth",
                        "boundary": boundary,
                        "group": bucket,
                        "value": val,
                    }
                )

        # Accuracy records
        for acc_rec in r["accuracy_records"]:
            tidx = acc_rec["task_idx"]
            for tid, m in acc_rec["results"].items():
                rows.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        "metric": "f1",
                        "task_idx": tidx,
                        "evaluated_task": int(tid),
                        "value": m["f1"],
                    }
                )

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
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "CKA"},
        )
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
        sns.heatmap(
            mat,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={"label": "F1"},
        )
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


def condition_bwt_test(
    results: list[dict], reference: str = "c4_full", alpha: float = 0.05
) -> dict | None:
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
    n_ref = len(by_cond[reference])
    rows, reject_any, any_underpowered = [], False, False
    for c in rivals:
        p = _mannwhitney(by_cond[reference], by_cond[c])
        floor = min_achievable_p(n_ref, len(by_cond[c]))
        underpowered = (floor == floor) and floor > bonf
        any_underpowered = any_underpowered or underpowered
        sig = (p == p) and p < bonf  # p==p screens NaN
        reject_any = reject_any or sig
        rows.append((c, float(np.mean(by_cond[c])), p, sig, floor, underpowered))
    return {
        "reference": reference,
        "ref_mean": float(np.mean(by_cond[reference])),
        "alpha": alpha,
        "bonferroni": bonf,
        "rivals": rows,
        "reject_H0": reject_any,
        "any_underpowered": any_underpowered,
        "n_per_group": n_ref,
    }


def component_profile_test(
    df: pd.DataFrame,
    reference: str = "c4_full",
    metric: str = "displacement_depth",
    alpha: float = 0.05,
) -> dict | None:
    """Distribution-targeting test (review C3): do conditions forget in a
    *different place*, not just by a different *amount*?

    For each condition we form the mean per-component forgetting **profile** (the
    vector of ``metric`` values across groups, L1-normalised to a distribution),
    then compare the reference profile to each rival's with the cosine distance
    ``1 - cos``. A permutation test over the pooled per-seed profiles gives a
    p-value on "are these two forgetting *distributions* different?", which the
    scalar |BWT| test cannot answer. Returns ``None`` if the metric is absent
    (e.g. older result files without displacement).
    """
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    groups = sorted(sub["group"].unique())

    def _profiles(cond: str) -> list[np.ndarray]:
        cc = sub[sub["condition"] == cond]
        out = []
        for seed, g in cc.groupby("seed"):
            vec = g.groupby("group")["value"].mean().reindex(groups).fillna(0.0).to_numpy()
            s = vec.sum()
            out.append(vec / s if s > 0 else vec)
        return out

    ref_profiles = _profiles(reference)
    if not ref_profiles:
        return None
    ref_mean = np.mean(ref_profiles, axis=0)

    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(1.0 - np.dot(a, b) / (na * nb))

    rivals = sorted(c for c in sub["condition"].unique() if c != reference)
    bonf = alpha / max(len(rivals), 1)
    rng = np.random.default_rng(0)
    rows, reject_any = [], False
    for c in rivals:
        riv_profiles = _profiles(c)
        if not riv_profiles:
            continue
        riv_mean = np.mean(riv_profiles, axis=0)
        observed = _cos_dist(ref_mean, riv_mean)
        # Permutation test: shuffle condition labels over the pooled profiles.
        pool = ref_profiles + riv_profiles
        n_ref = len(ref_profiles)
        count, n_perm = 0, 2000
        for _ in range(n_perm):
            rng.shuffle(pool)
            a = np.mean(pool[:n_ref], axis=0)
            b = np.mean(pool[n_ref:], axis=0)
            if _cos_dist(a, b) >= observed:
                count += 1
        p = (count + 1) / (n_perm + 1)
        sig = p < bonf
        reject_any = reject_any or sig
        rows.append((c, observed, p, sig))
    return {
        "reference": reference,
        "metric": metric,
        "groups": groups,
        "alpha": alpha,
        "bonferroni": bonf,
        "rivals": rows,
        "reject_H0": reject_any,
    }


def component_hypothesis_test(
    df: pd.DataFrame, condition: str = "c4_full", alpha: float = 0.05
) -> dict | None:
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


def _canonical_depth(layer: str) -> str:
    """Map a backbone-specific CKA layer name to a backbone-agnostic depth point.

    Both LayoutLMv3-base and BERT-base have 12 layers, so the probe points
    (embeddings, L0, L6, L11, head) line up across backbones — enabling the
    LayoutLMv3-vs-BERT depth-gradient contrast (review M1).
    """
    if "patch_embed" in layer:
        return "patch"
    if "classifier" in layer:
        return "head"
    m = re.search(r"encoder\.layer\.(\d+)", layer)
    if m:
        return f"L{int(m.group(1))}"
    if "embeddings" in layer:
        return "embeddings"
    return layer


def _bert_contrast_lines(df: pd.DataFrame) -> list[str]:
    """Markdown table contrasting the per-depth CKA gradient of the unimodal
    baseline (Cb/BERT) against the multimodal conditions at the first boundary.

    The key scientific question (review M1): does the multimodal encoder forget
    in a *different place* than a true unimodal text encoder, or does it just
    reproduce the known unimodal depth gradient?
    """
    cka = df[df["metric"] == "cka"].copy()
    out = [
        "",
        "### Multimodal-vs-unimodal contrast (CKA depth gradient, first boundary)",
        "",
    ]
    if cka.empty or "cb_bert" not in set(cka["condition"]):
        out.append("_No BERT (cb_bert) CKA data — run the `cb_bert` condition to populate._")
        return out
    first = sorted(cka["boundary"].unique())[0]
    sub = cka[cka["boundary"] == first].copy()
    sub["depth"] = sub["layer"].map(_canonical_depth)
    pivot = sub.groupby(["depth", "condition"])["value"].mean().reset_index()
    depth_order = ["embeddings", "patch", "L0", "L6", "L11", "head"]
    conds = sorted(pivot["condition"].unique())
    out += [
        f"Mean CKA (1.0 = no drift) at boundary `{first}`:",
        "",
        "| Depth | " + " | ".join(conds) + " |",
        "|" + "---|" * (len(conds) + 1),
    ]
    for d in depth_order:
        rowvals = []
        for c in conds:
            v = pivot[(pivot["depth"] == d) & (pivot["condition"] == c)]["value"]
            rowvals.append("--" if v.empty else f"{v.iloc[0]:.3f}")
        if all(x == "--" for x in rowvals):
            continue
        out.append(f"| {d} | " + " | ".join(rowvals) + " |")
    out += [
        "",
        "_Read the late-layer (L11) and head rows: if the multimodal conditions "
        "drift *more* (lower CKA) or *differently* than `cb_bert`, the depth/head "
        "forgetting is multimodal-specific rather than a re-confirmation of the "
        "unimodal result._",
    ]
    return out


def plot_displacement_bars(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of old-task-Fisher-weighted displacement (the forgetting
    localizer) per depth bucket, per condition, at the first task boundary."""
    disp = df[df["metric"] == "displacement_depth"].copy()
    if disp.empty:
        print("No displacement data — skipping displacement bars")
        return
    first = sorted(disp["boundary"].unique())[0]
    sub = disp[disp["boundary"] == first]
    agg = sub.groupby(["condition", "group"])["value"].agg(["mean", "std"]).reset_index()
    order = ["input", "early", "mid", "late", "head"]
    groups = [g for g in order if g in set(agg["group"])]
    conditions = sorted(agg["condition"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(len(conditions), 1)
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
    ax.set_xticklabels(groups)
    ax.set_ylabel("Fisher-weighted displacement (forgetting)")
    ax.set_title(f"Where forgetting lives by depth (boundary {first})")
    ax.legend()
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

    def _fmt(v: float) -> str:
        return "--" if v != v else f"{v:.2f}"  # v != v is True only for NaN

    for cond in sorted(by_cond.keys()):
        metrics_list = by_cond[cond]
        # nan-aware mean/std (an unavailable FWT must render as "--", never nan±nan or a
        # fabricated 0); ddof=1 sample std to match analyze_results.py result tables.
        means = {k: np.nanmean([m[k] for m in metrics_list]) for k in ["AA", "BWT", "AF", "FWT"]}
        stds = {
            k: (
                np.nanstd([m[k] for m in metrics_list], ddof=1)
                if len(metrics_list) > 1
                else float("nan")
            )
            for k in ["AA", "BWT", "AF", "FWT"]
        }
        lines.append(
            f"| {cond} | "
            + " | ".join(f"{_fmt(means[k])}±{_fmt(stds[k])}" for k in ["AA", "BWT", "AF", "FWT"])
            + " |"
        )

    # ─── Hypothesis tests (Mann-Whitney U, Bonferroni-corrected) ──────────────
    lines += ["", "## Hypothesis Tests (Mann-Whitney U, Bonferroni-corrected)", ""]

    cond_test = condition_bwt_test(results, reference="c4_full")
    lines += [
        "### Cross-condition (magnitude): does full-multimodal |BWT| differ from C1-C3/Cb?",
        "",
        "**H0:** |BWT| of C4 equals that of the ablated/unimodal conditions.  ",
        "**H1:** C4 forgets a different *amount* than at least one rival.",
        "",
        "_Note: this scalar-magnitude test is reported with its statistical-power "
        "floor. A non-rejection whose floor exceeds α is **inconclusive by "
        "construction** (underpowered), not evidence for H0 (review C3)._",
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
            "| Rival | mean \\|BWT\\| | p (vs C4) | min achievable p | significant | underpowered |",
            "|---|---|---|---|---|---|",
        ]
        for c, mean, p, sig, floor, under in cond_test["rivals"]:
            pstr = "n/a" if p != p else f"{p:.4f}"
            fstr = "n/a" if floor != floor else f"{floor:.4f}"
            lines.append(
                f"| {c} | {mean:.2f} | {pstr} | {fstr} | "
                f"{'**yes**' if sig else 'no'} | {'**yes**' if under else 'no'} |"
            )
        if cond_test["reject_H0"]:
            verdict = "**reject H0**"
        elif cond_test["any_underpowered"]:
            verdict = (
                "**inconclusive (underpowered)** — at least one comparison's "
                "minimum achievable p exceeds α'; add seeds before reading this as H0"
            )
        else:
            verdict = "fail to reject H0"
        lines += ["", f"Verdict: {verdict}."]

    # ─── Distribution-targeting test (where, not how much) ────────────────────
    prof_test = component_profile_test(df, reference="c4_full", metric="displacement_depth")
    lines += [
        "",
        "### Cross-condition (location): do conditions forget in a *different place*?",
        "",
        "**H0:** the per-depth forgetting *profile* of C4 equals each rival's.  ",
        "**H1:** the forgetting *distribution* across depth differs (permutation test "
        "on the displacement profile — the question |BWT| cannot answer).",
        "",
    ]
    if prof_test is None:
        lines.append("_No displacement data — re-run the pilot to populate this test._")
    else:
        lines += [
            f"- Reference **{prof_test['reference']}**, metric `{prof_test['metric']}`, "
            f"Bonferroni α'={prof_test['bonferroni']:.4f}.",
            "",
            "| Rival | cosine distance of profiles | p (perm) | significant |",
            "|---|---|---|---|",
        ]
        for c, dist, p, sig in prof_test["rivals"]:
            dstr = "n/a" if dist != dist else f"{dist:.3f}"
            pstr = "n/a" if p != p else f"{p:.4f}"
            lines.append(f"| {c} | {dstr} | {pstr} | {'**yes**' if sig else 'no'} |")
        verdict = "**reject H0**" if prof_test["reject_H0"] else "fail to reject H0"
        lines += ["", f"Verdict: {verdict} at the Bonferroni-corrected level."]

    # ─── BERT-vs-LayoutLMv3 depth gradient (multimodal-specific contrast) ─────
    lines += _bert_contrast_lines(df)

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
            "_Decision rule (revised; only measurable branches — review M4): "
            "head/late-layer-dominant forgetting → depth/head-targeted DocCL "
            "(classifier + late-layer protection); uniform across depth → uniform "
            "consolidation; no concentration → characterization-only. The earlier "
            "fusion-dominant / per-modality-visual branches are removed because a "
            "single-stream encoder cannot produce those signals._",
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
    plot_displacement_bars(df, args.figures_dir / "displacement_bars.pdf")
    plot_forgetting_matrix(df, args.figures_dir / "forgetting_matrix.pdf")
    write_findings_summary(df, results, args.pilot_dir / "findings_summary.md")

    print("\nDone. Review figures in", args.figures_dir)


if __name__ == "__main__":
    main()
