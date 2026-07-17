"""Deep forgetting RCA tables from Tier-B instrumented JSONs.

This is the paper-facing, CPU-only consolidation pass. It does not train anything; it
repackages ``results/rca/dil_*_seed42_rca.json`` into anatomy tables:

- which classes die, by task/boundary/mask;
- whether old-task output marginals snap to the just-trained task;
- whether modality masks change the forgetting amount after the validity guard;
- where Fisher-weighted movement lives.

Run from CLUE/:  uv run python scripts/rca_deep_dive.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MASK_VALID_F1 = 20.0
EXTINCT_F1 = 5.0
AVG_ROWS = {"micro avg", "macro avg", "weighted avg"}
TRUNK_BUCKETS = ("input", "early", "mid", "late")
MODALITY_EMBEDS = ("text_word_embed", "layout_2d_pos_embed", "image_patch_embed")


def _dist(v: np.ndarray) -> np.ndarray:
    total = float(v.sum())
    return v / total if total else np.full_like(v, np.nan, dtype=float)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else float("nan")


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def load_runs(rca_dir: str | Path = "results/rca") -> list[dict]:
    paths = sorted(Path(rca_dir).glob("dil_*_seed42_rca.json"))
    if not paths:
        raise SystemExit(f"no Tier-B JSONs in {rca_dir}")
    return [json.loads(p.read_text()) for p in paths]


def class_extinction(run: dict) -> pd.DataFrame:
    """Per-class F1 trajectory with at-learning baseline and extinction flag."""
    rows = []
    boundaries = run["boundaries"]
    for boundary_idx, boundary in enumerate(boundaries):
        for task_key, by_mask in boundary["eval"].items():
            task_idx = int(task_key)
            task_name = boundaries[task_idx]["task_name"]
            for mask, metrics in by_mask.items():
                task_at_learning = boundaries[task_idx]["eval"][task_key][mask]["f1"]
                learned = boundaries[task_idx]["eval"][task_key][mask]["per_class"]
                for cls, cur in metrics["per_class"].items():
                    base = learned.get(cls, {})
                    at_learning = base.get("f1", float("nan"))
                    f1 = cur.get("f1", float("nan"))
                    support = cur.get("support", 0.0)
                    rows.append(
                        {
                            "method": run["method"],
                            "boundary": boundary_idx,
                            "task_idx": task_idx,
                            "task_name": task_name,
                            "mask": mask,
                            "class": cls,
                            "old_task": task_idx < boundary_idx,
                            "final_boundary": boundary_idx == len(boundaries) - 1,
                            "at_learning_f1": at_learning,
                            "f1": f1,
                            "drop_from_at_learning": at_learning - f1,
                            "retention": f1 / at_learning if at_learning else np.nan,
                            "support": support,
                            "task_mask_valid": task_at_learning >= MASK_VALID_F1,
                            "class_acquired": at_learning >= MASK_VALID_F1,
                            "extinct": support > 0 and f1 < EXTINCT_F1,
                        }
                    )
    return pd.DataFrame(rows)


def task_mask_drops(run: dict) -> pd.DataFrame:
    """Task-level modality drops, using the same at-learning validity guard as Tier C."""
    rows = []
    boundaries = run["boundaries"]
    final = boundaries[-1]["eval"]
    for task_idx, boundary in enumerate(boundaries):
        for mask in run["masks"]:
            at_learning = boundary["eval"][str(task_idx)][mask]["f1"]
            final_f1 = final[str(task_idx)][mask]["f1"]
            rows.append(
                {
                    "method": run["method"],
                    "task_idx": task_idx,
                    "task_name": boundary["task_name"],
                    "mask": mask,
                    "old_task": task_idx < len(boundaries) - 1,
                    "at_learning_f1": at_learning,
                    "final_f1": final_f1,
                    "drop": at_learning - final_f1,
                    "retention": final_f1 / at_learning if at_learning else np.nan,
                    "mask_valid": at_learning >= MASK_VALID_F1,
                }
            )
    return pd.DataFrame(rows)


def marginal_snap(run: dict) -> pd.DataFrame:
    """Output-marginal snap metrics on old tasks under the FULL mask."""
    labels = run["labels"]
    rows = []
    for boundary_idx, boundary in enumerate(run["boundaries"]):
        if boundary_idx == 0:
            continue
        trained_conf = np.asarray(
            boundary["eval"][str(boundary_idx)]["full"]["confusion"], dtype=float
        )
        trained_gold = _dist(trained_conf.sum(axis=1))
        for task_idx in range(boundary_idx):
            conf = np.asarray(boundary["eval"][str(task_idx)]["full"]["confusion"], dtype=float)
            pred = _dist(conf.sum(axis=0))
            own_gold = _dist(conf.sum(axis=1))
            rows.append(
                {
                    "method": run["method"],
                    "boundary": boundary_idx,
                    "task_idx": task_idx,
                    "cos_pred_vs_trained_gold": _cos(pred, trained_gold),
                    "cos_pred_vs_own_gold": _cos(pred, own_gold),
                    "tv_pred_vs_trained_gold": 0.5 * float(np.abs(pred - trained_gold).sum()),
                    "tv_pred_vs_own_gold": 0.5 * float(np.abs(pred - own_gold).sum()),
                    "kl_pred_vs_trained_gold": _kl(pred, trained_gold),
                    "kl_pred_vs_own_gold": _kl(pred, own_gold),
                    "pred_top1": labels[int(np.nanargmax(pred))],
                    "pred_top1_mass": float(np.nanmax(pred)),
                }
            )
    return pd.DataFrame(rows)


def locus_summary(run: dict) -> pd.DataFrame:
    """Head/trunk and modality-embedding displacement summary."""
    rows = []
    for boundary in run["boundaries"]:
        depth = boundary.get("displacement_by_depth")
        if not depth:
            continue
        group = boundary.get("displacement_by_group", {})
        trunk_total = sum(depth.get(k, 0.0) for k in TRUNK_BUCKETS)
        modality = {k: group.get(k, np.nan) for k in MODALITY_EMBEDS}
        max_modality = max(
            modality, key=lambda k: -np.inf if np.isnan(modality[k]) else modality[k]
        )
        rows.append(
            {
                "method": run["method"],
                "boundary": boundary["after_task"],
                "head_displacement": depth.get("head", np.nan),
                "trunk_displacement": trunk_total,
                "head_trunk_ratio": (
                    depth.get("head", np.nan) / trunk_total if trunk_total else np.nan
                ),
                "late_trunk_share": (
                    depth.get("late", np.nan) / trunk_total if trunk_total else np.nan
                ),
                "max_modality_embed": max_modality,
                "max_modality_embed_disp": modality[max_modality],
                "text_word_embed_disp": modality["text_word_embed"],
                "layout_2d_pos_embed_disp": modality["layout_2d_pos_embed"],
                "image_patch_embed_disp": modality["image_patch_embed"],
                "layernorm_disp": group.get("layernorm", np.nan),
            }
        )
    return pd.DataFrame(rows)


def multimodal_summary(drops: pd.DataFrame) -> pd.DataFrame:
    old = drops[drops.old_task & drops.mask_valid]
    return (
        old.groupby(["method", "mask"], as_index=False)
        .agg(
            n_tasks=("task_idx", "count"),
            mean_at_learning_f1=("at_learning_f1", "mean"),
            mean_final_f1=("final_f1", "mean"),
            mean_drop=("drop", "mean"),
            mean_retention=("retention", "mean"),
        )
        .sort_values(["method", "mask"])
    )


def write_summary(
    out: Path, classes: pd.DataFrame, snaps: pd.DataFrame, loci: pd.DataFrame
) -> None:
    final_old = classes[
        classes.final_boundary
        & classes.old_task
        & (classes["mask"] == "full")
        & classes.task_mask_valid
        & ~classes["class"].isin(AVG_ROWS)
        & classes.extinct
    ]
    snap = snaps.copy()
    snap["snap_margin"] = snap.cos_pred_vs_trained_gold - snap.cos_pred_vs_own_gold
    head = loci.groupby("method", as_index=False).agg(
        mean_head_trunk_ratio=("head_trunk_ratio", "mean"),
        mean_late_trunk_share=("late_trunk_share", "mean"),
    )
    lines = [
        "# RCA-D: deep forgetting anatomy",
        "",
        "Generated by `scripts/rca_deep_dive.py` from Tier-B instrumented JSONs.",
        "",
        "## Final old-task extinctions (FULL mask, valid task mask)",
        "",
        "```",
        final_old[["method", "task_name", "class", "at_learning_f1", "f1", "support"]]
        .sort_values(["method", "task_name", "class"])
        .round(1)
        .to_string(index=False),
        "```",
        "",
        "## Strongest marginal snaps",
        "",
        "```",
        snap.sort_values("snap_margin", ascending=False)[
            [
                "method",
                "boundary",
                "task_idx",
                "cos_pred_vs_trained_gold",
                "cos_pred_vs_own_gold",
                "pred_top1",
                "pred_top1_mass",
            ]
        ]
        .head(15)
        .round(3)
        .to_string(index=False),
        "```",
        "",
        "## Locus summary",
        "",
        "```",
        head.round(3).to_string(index=False),
        "```",
        "",
    ]
    (out / "d_summary.md").write_text("\n".join(lines))


def main(rca_dir: str | Path = "results/rca") -> None:
    out = Path(rca_dir)
    runs = load_runs(out)
    log.info("loaded %d runs: %s", len(runs), [r["method"] for r in runs])

    classes = pd.concat([class_extinction(r) for r in runs], ignore_index=True)
    drops = pd.concat([task_mask_drops(r) for r in runs], ignore_index=True)
    snaps = pd.concat([marginal_snap(r) for r in runs], ignore_index=True)
    loci = pd.concat([locus_summary(r) for r in runs], ignore_index=True)

    classes.to_csv(out / "d_class_extinction.csv", index=False)
    drops.to_csv(out / "d_task_mask_drops.csv", index=False)
    multimodal_summary(drops).to_csv(out / "d_multimodal_summary.csv", index=False)
    snaps.to_csv(out / "d_marginal_snap_extended.csv", index=False)
    loci.to_csv(out / "d_locus_summary.csv", index=False)
    write_summary(out, classes, snaps, loci)
    log.info("saved -> %s/d_*.csv + d_summary.md", out)


if __name__ == "__main__":
    main()
