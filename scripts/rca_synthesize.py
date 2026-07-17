"""RCA Tier C synthesis: adjudication tables/figures for docs/RCA_HYPOTHESES_2026-07.md.

Reads the Tier B instrumented-run JSONs (``results/rca/dil_<method>_seed42_rca.json``,
produced by scripts/rca_baselines.py) plus Tier A's ``a2_family_signature.csv`` and emits:

  c_modality_deltas.csv            per (method, task, mask): at-learning vs final F1 drop,
                                   with the pre-registered mask-validity guard (H1/H3).
  c_confusion_flow.csv             per (method, boundary, old task, entity class): where the
                                   gold mass went — retained / O / VALUE / top-2 columns
                                   (H2/H4 disambiguation).
  c_displacement_vs_signature.csv  per (method, boundary): depth-profile shares + modality
                                   embed displacement, joined vs a2 immediate_share (H1/H3).
  c_fig_modality_drop.png          mask-drop bars per method.
  c_fig_displacement_vs_signature.png  late-trunk share vs immediate_share scatter.

CPU-only, read-only over results/. Run from CLUE/:  uv run python scripts/rca_synthesize.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MASKS = ["full", "text_only", "text_layout", "image_layout"]
MIN_AT_LEARNING_F1 = 20.0  # pre-registered mask-validity guard (RCA_HYPOTHESES_2026-07.md)
TRUNK_BUCKETS = ["input", "early", "mid", "late"]
MODALITY_EMBEDS = ["text_word_embed", "layout_2d_pos_embed", "image_patch_embed"]


def _entity(tag: str) -> str | None:
    """BIO tag -> entity class ('B-KEY' -> 'KEY'); None for 'O'."""
    return tag.split("-", 1)[1] if "-" in tag else None


def modality_deltas(data: dict) -> pd.DataFrame:
    """Drop(t, m) = at-learning F1 − final F1, per old task per mask (H1/H3 eval-space)."""
    boundaries = data["boundaries"]
    final_eval = boundaries[-1]["eval"]
    rows = []
    for t, boundary in enumerate(boundaries):
        for mask in data["masks"]:
            at_learning = boundary["eval"][str(t)][mask]["f1"]
            final = final_eval[str(t)][mask]["f1"]
            rows.append(
                {
                    "method": data["method"],
                    "task_idx": t,
                    "task_name": boundary["task_name"],
                    "mask": mask,
                    "at_learning_f1": at_learning,
                    "final_f1": final,
                    "drop": at_learning - final,
                    "is_old_task": t < len(boundaries) - 1,
                    "mask_valid": at_learning >= MIN_AT_LEARNING_F1,
                }
            )
    return pd.DataFrame(rows)


def confusion_flow(data: dict) -> pd.DataFrame:
    """Where each entity class's gold mass goes at every boundary (H2/H4).

    For gold rows of class E under FULL: retained = mass inside E's own B/I block
    (B/I flips count as retained); the rest decomposes into O, VALUE-tag columns,
    and the top-2 absorbing columns. Shares of off-diagonal mass, div0-guarded.
    """
    labels: list[str] = data["labels"]
    o_col = labels.index("O")
    cls_cols = {
        e: [i for i, tag in enumerate(labels) if _entity(tag) == e]
        for e in sorted({_entity(t) for t in labels} - {None})
    }
    rows = []
    for b, boundary in enumerate(data["boundaries"]):
        if b == 0:
            continue
        for t in range(b):  # old tasks only
            conf = np.asarray(boundary["eval"][str(t)]["full"]["confusion"], dtype=float)
            for entity, own_cols in cls_cols.items():
                gold = conf[own_cols]
                total = gold.sum()
                if total == 0:
                    continue
                retained = gold[:, own_cols].sum()
                off = total - retained
                col_mass = gold.sum(axis=0)
                col_mass[own_cols] = 0.0
                top2 = np.argsort(col_mass)[::-1][:2]
                to_value = 0.0 if entity == "VALUE" else col_mass[cls_cols["VALUE"]].sum()
                rows.append(
                    {
                        "method": data["method"],
                        "boundary": b,
                        "task_idx": t,
                        "entity": entity,
                        "gold_tokens": total,
                        "retained_share": retained / total,
                        "off_diag_mass": off,
                        "to_O_share": col_mass[o_col] / off if off else 0.0,
                        "to_VALUE_share": to_value / off if off else 0.0,
                        "top1_col": labels[top2[0]],
                        "top1_share": col_mass[top2[0]] / off if off else 0.0,
                        "top2_col": labels[top2[1]],
                        "top2_share": col_mass[top2[1]] / off if off else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def marginal_snap(data: dict) -> pd.DataFrame:
    """H4-amended test: does the head's output marginal on OLD tasks snap to the
    just-trained task's gold label marginal?

    Per (boundary b >= 1, old task t < b), FULL mask: cosine of the predicted-label
    marginal on task t vs (a) the just-trained task b's gold marginal and (b) task t's
    own gold marginal, plus O-row accuracy (invisible to seqeval per-class F1).
    """
    labels: list[str] = data["labels"]
    o_col = labels.index("O")
    rows = []
    for b, boundary in enumerate(data["boundaries"]):
        if b == 0:
            continue
        trained = np.asarray(boundary["eval"][str(b)]["full"]["confusion"], dtype=float)
        trained_gold = trained.sum(axis=1) / trained.sum()
        for t in range(b):
            conf = np.asarray(boundary["eval"][str(t)]["full"]["confusion"], dtype=float)
            pred = conf.sum(axis=0) / conf.sum()
            own_gold = conf.sum(axis=1) / conf.sum()
            o_row = conf[o_col]
            cos = lambda a, c: float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c)))  # noqa: E731
            rows.append(
                {
                    "method": data["method"],
                    "boundary": b,
                    "task_idx": t,
                    "cos_pred_vs_trained_gold": cos(pred, trained_gold),
                    "cos_pred_vs_own_gold": cos(pred, own_gold),
                    "o_row_acc": o_row[o_col] / o_row.sum() if o_row.sum() else np.nan,
                    "pred_top1": labels[int(np.argmax(pred))],
                }
            )
    return pd.DataFrame(rows)


def displacement_profile(data: dict) -> pd.DataFrame:
    """Depth-profile shares + modality-embed displacement per boundary (H1/H3).

    Trunk shares are over {input, early, mid, late} only (head reported raw — priors
    already say head dominates). A method whose trunk is frozen (colar past boundary 0)
    has no trunk keys -> shares NaN, flagged trunk_frozen.
    """
    rows = []
    for boundary in data["boundaries"]:
        depth = boundary.get("displacement_by_depth")
        if depth is None:  # boundary 0 has no displacement
            continue
        group = boundary.get("displacement_by_group", {})
        trunk = {k: depth[k] for k in TRUNK_BUCKETS if k in depth}
        trunk_total = sum(trunk.values())
        row = {
            "method": data["method"],
            "boundary": boundary["after_task"],
            "trunk_frozen": trunk_total == 0,
            "head_displacement": depth.get("head", np.nan),
        }
        for k in TRUNK_BUCKETS:
            row[f"{k}_share"] = trunk.get(k, np.nan) / trunk_total if trunk_total else np.nan
        for k in MODALITY_EMBEDS:
            row[f"{k}_disp"] = group.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def join_signature(profile: pd.DataFrame, a2_path: Path) -> pd.DataFrame:
    """Left-join a2 family signature (immediate_share) onto the displacement profile."""
    a2 = pd.read_csv(a2_path)[["method", "family", "immediate_share", "total_drop"]]
    return profile.merge(a2, on="method", how="left")


def fig_modality_drop(deltas: pd.DataFrame, out: Path) -> None:
    old = deltas[deltas.is_old_task & deltas.mask_valid]
    pivot = old.pivot_table(index="method", columns="mask", values="drop", aggfunc="mean")
    pivot = pivot.reindex(columns=[m for m in MASKS if m in pivot.columns])
    ax = pivot.plot.bar(figsize=(8, 4), rot=0)
    ax.set_ylabel("mean F1 drop on old tasks (at-learning − final)")
    ax.set_title("Modality-ablated forgetting per method (valid masks only)")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    plt.close(ax.figure)


def fig_displacement_vs_signature(joined: pd.DataFrame, out: Path) -> None:
    d = (
        joined[~joined.trunk_frozen]
        .groupby("method", as_index=False)
        .agg(late_share=("late_share", "mean"), immediate_share=("immediate_share", "first"))
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d.immediate_share, d.late_share)
    for _, r in d.iterrows():
        ax.annotate(r.method, (r.immediate_share, r.late_share), fontsize=8)
    ax.set_xlabel("immediate_share (a2 family signature)")
    ax.set_ylabel("late-trunk displacement share")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(rca_dir: str | Path = "results/rca") -> None:
    rca_dir = Path(rca_dir)
    paths = sorted(rca_dir.glob("dil_*_seed42_rca.json"))
    if not paths:
        raise SystemExit(f"no Tier B JSONs in {rca_dir} — chain not finished?")
    runs = [json.loads(p.read_text()) for p in paths]
    log.info("loaded %d Tier B runs: %s", len(runs), [r["method"] for r in runs])

    deltas = pd.concat([modality_deltas(r) for r in runs], ignore_index=True)
    flow = pd.concat([confusion_flow(r) for r in runs], ignore_index=True)
    snap = pd.concat([marginal_snap(r) for r in runs], ignore_index=True)
    profile = pd.concat([displacement_profile(r) for r in runs], ignore_index=True)
    joined = join_signature(profile, rca_dir / "a2_family_signature.csv")

    deltas.to_csv(rca_dir / "c_modality_deltas.csv", index=False)
    flow.to_csv(rca_dir / "c_confusion_flow.csv", index=False)
    snap.to_csv(rca_dir / "c_marginal_snap.csv", index=False)
    joined.to_csv(rca_dir / "c_displacement_vs_signature.csv", index=False)
    fig_modality_drop(deltas, rca_dir / "c_fig_modality_drop.png")
    fig_displacement_vs_signature(joined, rca_dir / "c_fig_displacement_vs_signature.png")
    log.info("wrote c_*.csv and c_fig_*.png to %s", rca_dir)


if __name__ == "__main__":
    main()
