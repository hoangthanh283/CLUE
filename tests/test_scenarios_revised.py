"""Offline tests for the revised scenario suite (Part B).

Covers the pure logic that needs no dataset download: the receipt unified-schema
remapping and the CORD class-partitioning (longer sequences + ordering). Full
scenario builds (which instantiate datasets) are exercised on the grid machine.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from doccl.data.receipt_remapping import (
    RECEIPT_LABEL_TO_ID,
    RECEIPT_UNIFIED_LABELS,
    Receipt_LabelRemapper,
    _wildreceipt_unified_name,
)
from doccl.data.scenarios import _CORD_CANONICAL_SESSIONS, _cord_class_sessions

# ───────────────────────────── receipt unified schema ──────────────────────────


def test_receipt_schema_is_13_bio_tags():
    assert len(RECEIPT_UNIFIED_LABELS) == 13  # O + 6 classes × {B,I}
    assert RECEIPT_UNIFIED_LABELS[0] == "O"


@pytest.mark.parametrize(
    "native,expected",
    [
        ("O", "O"),
        ("B-Store_name_value", "B-STORE"),
        ("I-Store_addr_key", "I-STORE"),
        ("B-Tel_value", "B-STORE"),
        ("B-Date_value", "B-DATE"),
        ("I-Time_value", "I-DATE"),
        ("B-Prod_item_value", "B-ITEM"),
        ("B-Prod_price_value", "B-PRICE"),
        ("B-Prod_quantity_value", "B-PRICE"),
        ("B-Subtotal_value", "B-TOTAL"),
        ("B-Tax_value", "B-TOTAL"),
        ("B-Total_value", "B-TOTAL"),
    ],
)
def test_wildreceipt_group_resolution(native, expected):
    assert _wildreceipt_unified_name(native) == expected


class _FakeDataset(Dataset):
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def test_receipt_remapper_sroie():
    native = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY", 3: "B-TOTAL", 4: "I-TOTAL"}
    ds = _FakeDataset([{"labels": torch.tensor([0, 1, 2, 3, 4, -100])}])
    out = Receipt_LabelRemapper(ds, "sroie", native)[0]["labels"]
    expected = [
        RECEIPT_LABEL_TO_ID["O"],
        RECEIPT_LABEL_TO_ID["B-STORE"],
        RECEIPT_LABEL_TO_ID["I-STORE"],
        RECEIPT_LABEL_TO_ID["B-TOTAL"],
        RECEIPT_LABEL_TO_ID["I-TOTAL"],
        -100,
    ]
    assert out.tolist() == expected


def test_receipt_remapper_cord():
    native = {0: "O", 1: "B-menu", 2: "B-void_menu", 3: "B-total"}
    ds = _FakeDataset([{"labels": torch.tensor([0, 1, 2, 3])}])
    out = Receipt_LabelRemapper(ds, "cord", native)[0]["labels"]
    assert out.tolist() == [
        RECEIPT_LABEL_TO_ID["O"],
        RECEIPT_LABEL_TO_ID["B-ITEM"],
        RECEIPT_LABEL_TO_ID["B-OTHER"],
        RECEIPT_LABEL_TO_ID["B-TOTAL"],
    ]


def test_receipt_remapper_unknown_dataset_raises():
    with pytest.raises(ValueError, match="No receipt mapping"):
        Receipt_LabelRemapper(_FakeDataset([]), "funsd", {})


# ───────────────────────────── CORD partition logic ────────────────────────────


def test_cord_canonical_is_5x6_over_30_classes():
    sessions = _cord_class_sessions(5, None)
    assert len(sessions) == 5
    assert all(len(s) == 6 for s in sessions)
    assert sum(len(s) for s in sessions) == 30
    assert sessions == [list(s) for s in _CORD_CANONICAL_SESSIONS]


def test_cord_long_partition_10x3():
    sessions = _cord_class_sessions(10, None)
    assert len(sessions) == 10
    assert all(len(s) == 3 for s in sessions)
    # Re-chunking preserves the full class set and canonical order.
    flat = [c for s in sessions for c in s]
    canonical_flat = [c for s in _CORD_CANONICAL_SESSIONS for c in s]
    assert flat == canonical_flat


def test_cord_order_permutation_reverses_sessions():
    base = _cord_class_sessions(5, None)
    rev = _cord_class_sessions(5, [4, 3, 2, 1, 0])
    assert rev == base[::-1]


def test_cord_invalid_num_sessions_raises():
    with pytest.raises(ValueError, match="divide 30 evenly"):
        _cord_class_sessions(7, None)


def test_cord_invalid_order_raises():
    with pytest.raises(ValueError, match="permutation"):
        _cord_class_sessions(5, [0, 1, 2, 3, 3])
