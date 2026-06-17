"""Unified receipt schema for the receipt-domain DIL scenario (``dil_receipts``).

SROIE, CORD-super, and WildReceipt are all receipts, so they align onto a richer,
far less lossy unified space than the form↔receipt 4-class DIL: 6 entity classes
+ O = 13 fixed BIO tags. The domain shift across the three tasks is *which fields
each receipt type exposes* (a realistic pure domain-IL with a constant head).

Coverage by dataset (a class absent from a dataset simply never appears in that
task — the same imbalance noted for the heterogeneous ``dil``, documented here):

    STORE  : SROIE company/address, WildReceipt Store_* + Tel
    DATE   : SROIE date, WildReceipt Date/Time
    ITEM   : CORD menu/sub, WildReceipt Prod_item
    PRICE  : WildReceipt Prod_quantity/Prod_price
    TOTAL  : SROIE total, CORD sub_total/total, WildReceipt Subtotal/Tax/Tips/Total
    OTHER  : CORD void_menu

This mirrors ``dil_remapping`` (name-level mapping → id translation) but with a
receipt-specific schema; WildReceipt's 24 ``<group>_{key,value}`` classes are
resolved by a group rule rather than an exhaustive table.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset

# Unified receipt schema (6 entity classes + O = 13 BIO tags). Fixed head.
RECEIPT_UNIFIED_LABELS = [
    "O",
    "B-STORE",
    "I-STORE",
    "B-DATE",
    "I-DATE",
    "B-ITEM",
    "I-ITEM",
    "B-PRICE",
    "I-PRICE",
    "B-TOTAL",
    "I-TOTAL",
    "B-OTHER",
    "I-OTHER",
]

RECEIPT_LABEL_TO_ID = {l: i for i, l in enumerate(RECEIPT_UNIFIED_LABELS)}


# Explicit native→unified NAME maps for the small-vocabulary datasets.
_STATIC_NAME_MAPPING: dict[str, dict[str, str]] = {
    # SROIE: company/address identify the store; date→DATE; total→TOTAL.
    "sroie": {
        "O": "O",
        "B-COMPANY": "B-STORE",
        "I-COMPANY": "I-STORE",
        "B-ADDRESS": "B-STORE",
        "I-ADDRESS": "I-STORE",
        "B-DATE": "B-DATE",
        "I-DATE": "I-DATE",
        "B-TOTAL": "B-TOTAL",
        "I-TOTAL": "I-TOTAL",
    },
    # CORD super-classes: menu/sub are line items; sub_total/total are totals;
    # void_menu is auxiliary. (CORD-super carries no store/date super-class.)
    "cord": {
        "O": "O",
        "B-menu": "B-ITEM",
        "I-menu": "I-ITEM",
        "B-sub": "B-ITEM",
        "I-sub": "I-ITEM",
        "B-sub_total": "B-TOTAL",
        "I-sub_total": "I-TOTAL",
        "B-total": "B-TOTAL",
        "I-total": "I-TOTAL",
        "B-void_menu": "B-OTHER",
        "I-void_menu": "I-OTHER",
    },
}

# WildReceipt 24 entity classes are ``<group>_{key,value}`` — map by group.
_WILDRECEIPT_GROUP_TO_UNIFIED = {
    "Store_name": "STORE",
    "Store_addr": "STORE",
    "Tel": "STORE",
    "Date": "DATE",
    "Time": "DATE",
    "Prod_item": "ITEM",
    "Prod_quantity": "PRICE",
    "Prod_price": "PRICE",
    "Subtotal": "TOTAL",
    "Tax": "TOTAL",
    "Tips": "TOTAL",
    "Total": "TOTAL",
}


def _wildreceipt_unified_name(native_name: str) -> str:
    """Resolve a WildReceipt BIO label (e.g. ``B-Store_name_value``) to unified."""
    if native_name == "O" or "-" not in native_name:
        return "O"
    prefix, cls = native_name.split("-", 1)  # "B", "Store_name_value"
    group = cls.rsplit("_", 1)[0]  # strip the trailing _value / _key
    unified = _WILDRECEIPT_GROUP_TO_UNIFIED.get(group)
    return f"{prefix}-{unified}" if unified else "O"


# native NAME → unified NAME resolver per dataset.
_RESOLVERS: dict[str, Callable[[str], str]] = {
    "sroie": lambda n: _STATIC_NAME_MAPPING["sroie"].get(n, "O"),
    "cord": lambda n: _STATIC_NAME_MAPPING["cord"].get(n, "O"),
    "wildreceipt": _wildreceipt_unified_name,
}


class Receipt_LabelRemapper(Dataset):
    """Wrap a receipt dataset and remap native labels to the unified receipt schema.

    Args:
        underlying: a Dataset returning dicts with a ``labels`` (L,) int tensor.
        dataset_name: one of {"sroie", "cord", "wildreceipt"}.
        native_id_to_label: the underlying dataset's ``id_to_label`` map.
    """

    def __init__(
        self,
        underlying: Dataset,
        dataset_name: str,
        native_id_to_label: dict[int, str],
    ):
        if dataset_name not in _RESOLVERS:
            raise ValueError(
                f"No receipt mapping for {dataset_name!r}. Available: {list(_RESOLVERS)}"
            )
        self.underlying = underlying
        self.dataset_name = dataset_name
        resolve = _RESOLVERS[dataset_name]
        self._id_translation: dict[int, int] = {
            native_id: RECEIPT_LABEL_TO_ID[resolve(native_name)]
            for native_id, native_name in native_id_to_label.items()
        }

    def __len__(self) -> int:
        return len(self.underlying)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.underlying[idx]
        labels = item["labels"]
        remapped = labels.clone()
        for native, unified in self._id_translation.items():
            remapped[labels == native] = unified
        remapped[labels == -100] = -100  # preserve ignore index
        return {**item, "labels": remapped}
