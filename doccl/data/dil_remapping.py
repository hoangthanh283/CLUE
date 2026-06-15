"""Dataset utilities: label remapping for cross-dataset unified schemas (DIL).

The DIL scenario requires FUNSD/SROIE/CORD to share a unified label space
(HEADER / KEY / VALUE / OTHER). Each underlying dataset has its own native
labels which we remap to this shared space.

This module provides:
    - DIL_LabelRemapper: a Dataset wrapper that translates ner_tags on the fly
    - DIL_UNIFIED_LABELS: the canonical 9-tag BIO label list
    - DIL_MAPPING: dict[dataset_name -> dict[native_label_id -> unified_label_id]]
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

# Unified DIL schema (4 entity classes + O = 9 BIO tags)
DIL_UNIFIED_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-KEY",
    "I-KEY",
    "B-VALUE",
    "I-VALUE",
    "B-OTHER",  # reserved for explicit "other entity" tags from CORD
    "I-OTHER",
]

DIL_LABEL_TO_ID = {l: i for i, l in enumerate(DIL_UNIFIED_LABELS)}


# Mapping from native label NAMES to unified label NAMES.
# We map at the name level (not id level) so it survives label-set changes
# in the underlying datasets.
DIL_NAME_MAPPING: dict[str, dict[str, str]] = {
    # FUNSD: 4 entity types — straightforward
    "funsd": {
        "O": "O",
        "B-HEADER": "B-HEADER",
        "I-HEADER": "I-HEADER",
        "B-QUESTION": "B-KEY",
        "I-QUESTION": "I-KEY",
        "B-ANSWER": "B-VALUE",
        "I-ANSWER": "I-VALUE",
    },
    # SROIE: 4 fields, all are values from a form-filling perspective.
    # Treat company/address as KEY-like (entity name fields), date/total as VALUE.
    # This mapping is somewhat arbitrary — document the choice in
    # docs/dil_schema_mapping.md.
    "sroie": {
        "O": "O",
        "B-COMPANY": "B-KEY",
        "I-COMPANY": "I-KEY",
        "B-ADDRESS": "B-VALUE",
        "I-ADDRESS": "I-VALUE",
        "B-DATE": "B-VALUE",
        "I-DATE": "I-VALUE",
        "B-TOTAL": "B-VALUE",
        "I-TOTAL": "I-VALUE",
    },
    # CORD super-classes: menu/sub_total/total/void_menu/sub
    # Map menu/total/sub_total to VALUE (numeric/itemized data)
    # void_menu and sub are auxiliary — map to OTHER
    "cord": {
        "O": "O",
        "B-menu": "B-VALUE",
        "I-menu": "I-VALUE",
        "B-sub_total": "B-VALUE",
        "I-sub_total": "I-VALUE",
        "B-total": "B-VALUE",
        "I-total": "I-VALUE",
        "B-void_menu": "B-OTHER",
        "I-void_menu": "I-OTHER",
        "B-sub": "B-OTHER",
        "I-sub": "I-OTHER",
    },
    # XFUND shares FUNSD's exact entity schema (HEADER/QUESTION/ANSWER), so it uses
    # the identical mapping. Used by the cross-lingual DIL scenario where the domain
    # shift is *language*, not schema — the unified label space is constant across
    # tasks, isolating representation drift.
    "xfund": {
        "O": "O",
        "B-HEADER": "B-HEADER",
        "I-HEADER": "I-HEADER",
        "B-QUESTION": "B-KEY",
        "I-QUESTION": "I-KEY",
        "B-ANSWER": "B-VALUE",
        "I-ANSWER": "I-VALUE",
    },
}


class DIL_LabelRemapper(Dataset):
    """Wraps a dataset and remaps native labels to the unified DIL schema.

    Args:
        underlying: a Dataset returning dicts with `labels` (B, L) tensor of int ids
        dataset_name: one of {"funsd", "sroie", "cord"}
        native_id_to_label: dict[int, str] from underlying dataset
    """

    def __init__(
        self,
        underlying: Dataset,
        dataset_name: str,
        native_id_to_label: dict[int, str],
    ):
        if dataset_name not in DIL_NAME_MAPPING:
            raise ValueError(
                f"No DIL mapping for {dataset_name!r}. "
                f"Available: {list(DIL_NAME_MAPPING.keys())}"
            )
        self.underlying = underlying
        self.dataset_name = dataset_name

        # Build native_id -> unified_id translation table
        name_map = DIL_NAME_MAPPING[dataset_name]
        self._id_translation: dict[int, int] = {}
        for native_id, native_name in native_id_to_label.items():
            unified_name = name_map.get(native_name, "O")
            self._id_translation[native_id] = DIL_LABEL_TO_ID[unified_name]

    def __len__(self) -> int:
        return len(self.underlying)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.underlying[idx]
        labels = item["labels"]
        # Remap labels element-wise; preserve -100 (ignore index)
        remapped = labels.clone()
        for native, unified in self._id_translation.items():
            remapped[labels == native] = unified
        # Don't touch -100 positions
        remapped[labels == -100] = -100
        item = {**item, "labels": remapped}
        return item
