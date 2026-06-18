"""Continual learning evaluation metrics.

Reference: Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning",
NeurIPS 2017, arXiv:1706.08840

Definitions (matrix R[i][j] = perf on task j after training task i):
    AA  (Average Accuracy)      = (1/T) Σ_i R[T-1, i]
    BWT (Backward Transfer)     = (1/(T-1)) Σ_{i<T-1} (R[T-1, i] - R[i, i])
    AF  (Average Forgetting)    = -BWT
    FWT (Forward Transfer)      = (1/(T-1)) Σ_{i>0} (R[i-1, i] - b_i)
        where b_i is random/baseline init performance on task i
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_precision
from seqeval.metrics import recall_score as seq_recall


def compute_token_f1(
    preds: list[int],
    labels: list[int],
    label_map: dict[int, str],
) -> dict[str, float]:
    """Span-based F1 using seqeval (BIO scheme)."""
    if not preds:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred_str = [label_map.get(p, "O") for p in preds]
    gold_str = [label_map.get(l, "O") for l in labels]
    return {
        "f1": seq_f1([gold_str], [pred_str], zero_division=0) * 100,
        "precision": seq_precision([gold_str], [pred_str], zero_division=0) * 100,
        "recall": seq_recall([gold_str], [pred_str], zero_division=0) * 100,
    }


def compute_per_class_f1(
    preds: list[int],
    labels: list[int],
    label_map: dict[int, str],
) -> dict[str, dict[str, float]]:
    """Per-entity-type precision/recall/F1/support via seqeval.

    Returns ``{entity_type: {"precision", "recall", "f1", "support"}}`` (values
    in %; support is the raw gold-span count). Used to diagnose whether an
    aggregate score is carried by one dominant class — e.g. the VALUE-dominant
    DIL schema where KEY appears only in FUNSD/SROIE (review M5).
    """
    if not preds:
        return {}
    pred_str = [label_map.get(p, "O") for p in preds]
    gold_str = [label_map.get(l, "O") for l in labels]
    report = seq_report([gold_str], [pred_str], output_dict=True, zero_division=0)
    out: dict[str, dict[str, float]] = {}
    for cls, vals in report.items():
        if not isinstance(vals, dict):
            continue
        out[cls] = {
            "precision": vals.get("precision", 0.0) * 100,
            "recall": vals.get("recall", 0.0) * 100,
            "f1": vals.get("f1-score", 0.0) * 100,
            "support": float(vals.get("support", 0)),
        }
    return out


def label_frequencies(
    labels: list[int],
    label_map: dict[int, str],
) -> dict[str, dict[str, int]]:
    """Count BIO-tag and entity-type frequencies in a flat label stream.

    ``-100`` (ignored) positions are skipped. Returns
    ``{"bio": {tag: count}, "entity": {entity_type: count}}`` where the entity
    count tallies span *starts* (``B-`` tags), the right unit for "how many KEY
    spans exist in this task" — the evidence the DIL interpretation needs (M5).
    """
    bio: Counter[str] = Counter()
    entity: Counter[str] = Counter()
    for lid in labels:
        if lid == -100:
            continue
        name = label_map.get(lid, "O")
        bio[name] += 1
        if name.startswith("B-"):
            entity[name[2:]] += 1
    return {"bio": dict(bio), "entity": dict(entity)}


class CLMetricsTracker:
    """Tracks the accuracy matrix R ∈ R^{TxT} and computes AA/BWT/FWT.

    R[i][j] = performance on task j after training task i (NaN if not measured).
    """

    def __init__(self, num_tasks: int, baseline_perf: list[float] | None = None):
        self.num_tasks = num_tasks
        self.matrix = np.full((num_tasks, num_tasks), np.nan)
        self.baseline = baseline_perf

    def update(self, task_idx_after: int, results: dict[int, dict[str, float]]) -> None:
        """Args:
        task_idx_after: index of the task just trained (0-indexed)
        results: {task_id: {"f1": value, ...}}
        """
        for tid, metrics in results.items():
            self.matrix[task_idx_after, tid] = metrics["f1"]

    def average_accuracy(self) -> float:
        """AA = mean over tasks of final-step performance."""
        return float(np.nanmean(self.matrix[self.num_tasks - 1]))

    def backward_transfer(self) -> float:
        """BWT = mean over tasks i<T-1 of (R[T-1, i] - R[i, i])."""
        T = self.num_tasks
        diffs = []
        for i in range(T - 1):
            if not np.isnan(self.matrix[T - 1, i]) and not np.isnan(self.matrix[i, i]):
                diffs.append(self.matrix[T - 1, i] - self.matrix[i, i])
        # NaN (not 0.0) when no valid pair exists — e.g. the Joint oracle fills only
        # the last matrix row, so the diagonal R[i,i] for i<T-1 is NaN and BWT is
        # undefined. 0.0 is a real "no forgetting" value and must not be conflated
        # with "undefined"; downstream aggregation/rendering treats NaN honestly.
        return float(np.mean(diffs)) if diffs else float("nan")

    def forgetting(self) -> float:
        """AF = -BWT (positive = magnitude of forgetting)."""
        return -self.backward_transfer()

    def forward_transfer(self) -> float:
        """FWT = mean over tasks i>0 of (R[i-1, i] - b_i). Requires baseline.

        Returns NaN (not 0.0) when the per-run baseline b_i was not seeded — 0.0
        is a real FWT value (no transfer) and must not be confused with "unknown".
        The zero-shot term R[i-1, i] is recorded by train.py and persisted in the
        matrix, so true FWT is computed at aggregation time (analyze_results.py)
        where the single-task baselines b_i are available.
        """
        if self.baseline is None:
            return float("nan")
        diffs = []
        for i in range(1, self.num_tasks):
            if not np.isnan(self.matrix[i - 1, i]):
                diffs.append(self.matrix[i - 1, i] - self.baseline[i])
        return float(np.mean(diffs)) if diffs else float("nan")

    def per_task_forgetting(self) -> dict[int, float]:
        """Forgetting per individual task (for diagnostic plots)."""
        T = self.num_tasks
        out = {}
        for i in range(T - 1):
            if not np.isnan(self.matrix[T - 1, i]) and not np.isnan(self.matrix[i, i]):
                out[i] = self.matrix[i, i] - self.matrix[T - 1, i]
        return out

    def summary(self) -> dict[str, float]:
        return {
            "AA": self.average_accuracy(),
            "BWT": self.backward_transfer(),
            "AF": self.forgetting(),
            "FWT": self.forward_transfer(),
        }

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "num_tasks": self.num_tasks,
            **self.summary(),
        }
