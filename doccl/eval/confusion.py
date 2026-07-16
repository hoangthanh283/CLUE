"""Token-level confusion counts — the error-taxonomy primitive the RCA needs.

A confusion matrix distinguishes forgetting root causes that aggregate F1 conflates:
entity→O mass extinction (head-bias drift toward the majority class) vs entity→entity
confusion (representation entanglement) vs B-/I- boundary flips (span fragmentation).
"""

from __future__ import annotations

import numpy as np

__all__ = ["confusion_counts"]


def confusion_counts(preds: list[int], labels: list[int], n_labels: int) -> np.ndarray:
    """(n_labels, n_labels) int64 counts, ``cm[gold, pred] += 1``; gold==-100 excluded."""
    p = np.asarray(preds, dtype=np.int64)
    g = np.asarray(labels, dtype=np.int64)
    keep = g != -100
    p, g = p[keep], g[keep]
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(cm, (g, p), 1)
    return cm
