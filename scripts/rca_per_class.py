"""RCA-A1: per-class forgetting ledger across all saved runs.

Mines every ``results/<run>/per_class_f1.json`` (final-model per-entity-class F1 per seen
task) that has a sibling ``metrics.json``. Answers, per method:

1. Which classes survive the CL sequence and which die (gap to the joint oracle's
   per-class F1, 3 seeds).
2. The AAAI-review M5/Q6 question: is a method's near-oracle DIL AA a label-degeneracy
   artifact of the VALUE-dominant schema? Signature: cross-task support-weighted
   VALUE-F1 tracks AA while KEY-F1 collapses. (CORD emits no KEY spans at all, so KEY
   retention is carried entirely by FUNSD/SROIE — exactly where forgetting bites.)
3. Class-frequency correlation: Spearman(support, final F1) per method family.

CPU-only, read-only over results/, writes results/rca/a1_*.{csv,md}.
Run from CLUE/:  uv run python scripts/rca_per_class.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

AVG_ROWS = {"micro avg", "macro avg", "weighted avg"}
SKIP_DIR_PREFIXES = ("_invalid", "nightly_", "pilot", "rca")

FAMILY = {
    "naive": "none",
    "joint": "oracle",
    "ewc": "penalty",
    "lwf": "distill",
    "er": "replay_raw",
    "er_cflat": "replay_raw",
    "der_pp": "replay_logit",
    "latent_replay": "replay_latent",
    "colar": "replay_latent",
    "colar_bal": "replay_latent",
    "colar_knn": "replay_latent",
    "colar_meta": "replay_latent",
    "proxy_latent_replay": "replay_latent",
    "l2p": "prompt",
    "dualprompt": "prompt",
    "coda_prompt": "prompt",
    "hrp": "prompt",
    "o_lora": "adapter",
    "cl_lora": "adapter",
    "sd_lora": "adapter",
    "doccl": "hybrid",
}


def load_all_per_class(results_dir: str | Path = "results") -> pd.DataFrame:
    """Long-format ledger: one row per (run, task_idx, class)."""
    rows = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith(SKIP_DIR_PREFIXES):
            continue
        pcf, met = run_dir / "per_class_f1.json", run_dir / "metrics.json"
        if not (pcf.exists() and met.exists()):
            continue
        try:
            per_class = json.loads(pcf.read_text())
            metrics = json.loads(met.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("skipping %s: %s", run_dir.name, e)
            continue
        method = metrics.get("method", "?")
        for task_idx, classes in per_class.items():
            for cls, m in classes.items():
                rows.append(
                    {
                        "run": run_dir.name,
                        "method": method,
                        "family": FAMILY.get(method, "other"),
                        "scenario": metrics.get("scenario", "?"),
                        "seed": metrics.get("seed", -1),
                        "AA": metrics.get("AA"),
                        "BWT": metrics.get("BWT"),
                        "task_idx": int(task_idx),
                        "class": cls,
                        "is_avg": cls in AVG_ROWS,
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                        "f1": m.get("f1"),
                        "support": m.get("support"),
                    }
                )
    return pd.DataFrame(rows)


def joint_reference(df: pd.DataFrame, scenario: str = "dil") -> pd.DataFrame:
    """Per-(task_idx, class) joint-oracle F1 mean/std across seeds."""
    j = df[(df.scenario == scenario) & (df.method == "joint") & ~df.is_avg]
    return (
        j.groupby(["task_idx", "class"])["f1"]
        .agg(joint_mean="mean", joint_std="std", joint_n="count")
        .reset_index()
    )


def class_survival(df: pd.DataFrame, ref: pd.DataFrame, scenario: str = "dil") -> pd.DataFrame:
    """Gap to the joint oracle per (method, task_idx, class), averaged over seeds."""
    d = df[(df.scenario == scenario) & ~df.is_avg]
    agg = (
        d.groupby(["method", "family", "task_idx", "class"])["f1"]
        .agg(f1_mean="mean", f1_std="std", n_seeds="count")
        .reset_index()
    )
    out = agg.merge(ref, on=["task_idx", "class"], how="left")
    out["gap_to_joint"] = out["f1_mean"] - out["joint_mean"]
    return out.sort_values(["method", "task_idx", "class"])


def cross_task_class_f1(df: pd.DataFrame, scenario: str = "dil") -> pd.DataFrame:
    """Support-weighted per-class F1 across all seen tasks, per run — the M5/Q6 lens."""
    d = df[(df.scenario == scenario) & ~df.is_avg & (df.support > 0)].copy()
    d["wf1"] = d.f1 * d.support
    g = (
        d.groupby(["run", "method", "family", "seed", "AA", "class"])
        .agg(wf1=("wf1", "sum"), support=("support", "sum"))
        .reset_index()
    )
    g["class_f1"] = g.wf1 / g.support
    return g.drop(columns=["wf1"])


def m5q6_verdicts(cross: pd.DataFrame) -> pd.DataFrame:
    """Per method: does AA ride on VALUE while KEY collapses? (label-degeneracy check)."""
    pivot = cross.pivot_table(
        index=["method", "family", "seed", "AA"], columns="class", values="class_f1"
    ).reset_index()
    per_method = pivot.groupby(["method", "family"]).mean(numeric_only=True).reset_index()
    if {"KEY", "VALUE"} <= set(per_method.columns):
        per_method["key_minus_aa"] = per_method["KEY"] - per_method["AA"]
        per_method["value_minus_aa"] = per_method["VALUE"] - per_method["AA"]
        # degenerate: aggregate AA tracks VALUE (within 10 pts) while KEY sits >20 under it
        per_method["degeneracy_flag"] = (per_method["value_minus_aa"].abs() < 10) & (
            per_method["key_minus_aa"] < -20
        )
    return per_method.sort_values("AA", ascending=False)


def class_frequency_correlation(df: pd.DataFrame, scenario: str = "dil") -> pd.DataFrame:
    """Spearman(support, f1) over (task, class) cells, per method family."""
    d = df[(df.scenario == scenario) & ~df.is_avg & (df.support > 0)]
    rows = []
    for fam, g in d.groupby("family"):
        if len(g) >= 5:
            rows.append(
                {
                    "family": fam,
                    "spearman_support_f1": g["support"].corr(g["f1"], method="spearman"),
                    "n_cells": len(g),
                }
            )
    return pd.DataFrame(rows).sort_values("spearman_support_f1")


def write_summary(
    out_dir: Path, survival: pd.DataFrame, verdicts: pd.DataFrame, freq: pd.DataFrame
) -> None:
    lines = [
        "# RCA-A1: per-class forgetting ledger (auto-generated by scripts/rca_per_class.py)",
        "",
        "## M5/Q6 — label-degeneracy check (dil, cross-task support-weighted class F1)",
        "",
        "Degeneracy signature: AA ≈ VALUE-F1 while KEY-F1 ≪ AA (KEY exists only in",
        "FUNSD/SROIE — CORD emits no KEY spans, so aggregate AA can hide total KEY loss).",
        "",
        "```",
        verdicts.round(1).to_string(index=False),
        "```",
        "",
        "## Class survival vs joint oracle (dil, gap_to_joint = method − joint per class/task)",
        "",
        "Worst 20 cells (most-forgotten class×task per method):",
        "",
        "```",
        survival.nsmallest(20, "gap_to_joint").round(1).to_string(index=False),
        "```",
        "",
        "## Class-frequency correlation (Spearman support↔F1 per family)",
        "",
        "```",
        freq.round(3).to_string(index=False),
        "```",
        "",
    ]
    (out_dir / "a1_summary.md").write_text("\n".join(lines))


def main() -> None:
    out_dir = Path("results/rca")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_all_per_class()
    log.info("ledger: %d rows from %d runs", len(df), df.run.nunique())
    df.to_csv(out_dir / "a1_per_class_ledger.csv", index=False)

    ref = joint_reference(df)
    survival = class_survival(df, ref)
    survival.to_csv(out_dir / "a1_class_survival.csv", index=False)

    cross = cross_task_class_f1(df)
    verdicts = m5q6_verdicts(cross)
    verdicts.to_csv(out_dir / "a1_m5q6_verdicts.csv", index=False)
    freq = class_frequency_correlation(df)

    write_summary(out_dir, survival, verdicts, freq)
    log.info("M5/Q6 verdicts:\n%s", verdicts.round(1).to_string(index=False))
    log.info("saved -> %s/a1_*.csv + a1_summary.md", out_dir)


if __name__ == "__main__":
    main()
