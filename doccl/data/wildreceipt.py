"""WildReceipt dataset loader (large-scale receipts, 26 KIE fields).

Source: ``kaydee/wildreceipt`` on HuggingFace — WildReceipt in token-classification
form WITH images. ~1.7k receipts, far larger and schema-richer than SROIE, used here
for a class-incremental scenario (``build_cil_wildreceipt``): the 24 real entity
classes are split into incremental sessions with a growing classifier head.

The upstream ``ner_tags`` are FLAT span labels (not BIO): a ClassLabel of 26 names
``['Ignore', 'Store_name_value', 'Store_name_key', ..., 'Total_key', 'Others']``,
paired ``*_key`` / ``*_value`` fields. ``Ignore`` and ``Others`` are background → O.
We run-encode the flat tags into a BIO scheme (first token of a span → B-, rest → I-),
exactly as the SROIE loader does for its flat ``S-<FIELD>`` mirror.

Reference:
    Sun et al., "Spatial Dual-Modality Graph Reasoning for Key Information Extraction"
    (WildReceipt), arXiv:2103.14470.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor

# Upstream background classes that map to O (no entity).
_BACKGROUND = {"Ignore", "Others"}

# Process-wide cache keyed by split.
_RAW_DS_CACHE: dict[str, Any] = {}


def _build_label_names(native_names: list[str]) -> list[str]:
    """Build the BIO LABEL_NAMES from the upstream flat class names.

    Order is deterministic: O first, then B-/I- for each non-background class in the
    upstream order, so label ids are stable across runs/seeds.
    """
    classes = [n for n in native_names if n not in _BACKGROUND]
    names = ["O"]
    for c in classes:
        names.append(f"B-{c}")
        names.append(f"I-{c}")
    return names


class WildReceiptDataset(Dataset):
    """WildReceipt KIE dataset wrapper (24 entity classes → 49 BIO tags).

    LABEL_NAMES / NUM_LABELS are derived from the upstream ClassLabel at load time
    (class-attribute defaults are filled on first construction so scenarios can read
    them without instantiating).
    """

    # Filled from the upstream schema on first load; the 24 non-background classes give
    # 1 + 2*24 = 49 BIO tags. Declared here so static references don't fail pre-load.
    LABEL_NAMES: list[str] = []
    NUM_LABELS: int = 0

    def __init__(
        self,
        split: str = "train",
        processor: LayoutLMv3Processor | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
        hf_name: str = "kaydee/wildreceipt",
    ):
        """Args:
        split: "train" or "test".
        processor: LayoutLMv3Processor (created from base if None).
        max_length: max token length.
        label_filter: keep only examples with an in-set entity, masking the rest
                      to O (class-incremental split building).
        hf_name: HuggingFace dataset id.
        """
        self.split = split
        self.max_length = max_length
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        if split not in _RAW_DS_CACHE:
            _RAW_DS_CACHE[split] = load_dataset(hf_name, split=split)
        self._ds = _RAW_DS_CACHE[split]

        self._native_names = self._ds.features["ner_tags"].feature.names
        names = _build_label_names(self._native_names)
        # Set on the class so scenario builders can read LABEL_NAMES/NUM_LABELS.
        type(self).LABEL_NAMES = names
        type(self).NUM_LABELS = len(names)
        self.label_to_id = {l: i for i, l in enumerate(names)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        self.data = self._parse_examples(self._ds)
        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _parse_examples(self, ds) -> list[dict]:
        """Build lightweight rows (no decoded image): row idx + tokens + boxes
        + BIO ner_tags run-encoded from the upstream flat span labels.

        WildReceipt bboxes are already in 0–1000 space (verified), so no normalisation.
        """
        data = []
        for row in range(len(ds)):
            ex = ds[row]
            native = ex["ner_tags"]
            bio = self._flat_to_bio([self._native_names[t] for t in native])
            data.append(
                {
                    "row": row,
                    "tokens": ex["words"],
                    "bboxes": ex["bboxes"],
                    "ner_tags": bio,
                }
            )
        return data

    def _flat_to_bio(self, flat_names: list[str]) -> list[int]:
        """Run-encode flat per-token class names into BIO label ids.

        Consecutive tokens with the same non-background class form one span: the first
        token → B-<class>, the rest → I-<class>. Background classes → O.
        """
        out: list[int] = []
        prev = None
        for name in flat_names:
            if name in _BACKGROUND:
                out.append(self.label_to_id["O"])
                prev = None
                continue
            tag = f"B-{name}" if name != prev else f"I-{name}"
            out.append(self.label_to_id[tag])
            prev = name
        return out

    def _filter_by_labels(self, data: list, label_filter: list[str]) -> list:
        """CIL split: keep examples with ≥1 in-session entity; mask the rest to O."""
        o_id = self.label_to_id["O"]
        target_entity_ids = {self.label_to_id[l] for l in label_filter if l in self.label_to_id} - {
            o_id
        }

        filtered = []
        for ex in data:
            if not (set(ex["ner_tags"]) & target_entity_ids):
                continue
            masked = [t if t in target_entity_ids else o_id for t in ex["ner_tags"]]
            filtered.append({**ex, "ner_tags": masked})
        return filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.data[idx]
        image = self._ds[ex["row"]]["image"]  # decode on demand
        encoding = self.processor(
            image,
            ex["tokens"],
            boxes=ex["bboxes"],
            word_labels=ex["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
