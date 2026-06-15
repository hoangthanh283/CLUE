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

import numpy as np
from seqeval.metrics import (
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
)


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
        return float(np.mean(diffs)) if diffs else 0.0

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
