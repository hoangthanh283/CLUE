"""Unit tests for doccl.eval.confusion — exact counts, -100 exclusion."""

from __future__ import annotations

import numpy as np

from doccl.eval.confusion import confusion_counts


def test_exact_counts_gold_rows_pred_cols():
    cm = confusion_counts(preds=[0, 1, 1], labels=[0, 1, 0], n_labels=2)
    assert cm.tolist() == [[1, 1], [0, 1]]  # gold 0: pred {0,1}; gold 1: pred 1


def test_ignore_index_excluded():
    cm = confusion_counts(preds=[0, 1, 0], labels=[-100, 1, -100], n_labels=2)
    assert cm.sum() == 1
    assert cm[1, 1] == 1


def test_empty_input():
    cm = confusion_counts(preds=[], labels=[], n_labels=3)
    assert cm.shape == (3, 3) and cm.sum() == 0


def test_row_sums_match_gold_support():
    labels = [0, 0, 1, 2, 2, 2, -100]
    preds = [0, 1, 1, 0, 2, 2, 0]
    cm = confusion_counts(preds, labels, n_labels=3)
    assert cm.sum(axis=1).tolist() == [2, 1, 3]
    assert isinstance(cm, np.ndarray) and cm.dtype == np.int64
