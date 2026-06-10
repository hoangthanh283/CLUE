"""CORD (Consolidated Receipt Dataset) loader.

Source: naver-clova-ix/cord-v2 on HuggingFace.
1000 receipts (800 train / 100 val / 100 test), 30 fine-grained entity classes
organized in 5 super-classes.

Reference:
    Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing",
    DI@NeurIPS 2019.

This loader flattens the hierarchical JSON structure into a flat sequence-labeling
format compatible with LayoutLMv3 token classification. We use 30 fine-grained
classes (61 BIO tags) for class-incremental scenarios, with optional super-class
mapping (11 BIO tags) for domain-incremental scenarios.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


# 30 fine-grained class names — derived from CORD-v2 schema
CORD_FINE_LABELS = [
    "menu.cnt",
    "menu.discountprice",
    "menu.itemsubtotal",
    "menu.nm",
    "menu.num",
    "menu.price",
    "menu.sub.cnt",
    "menu.sub.nm",
    "menu.sub.price",
    "menu.sub.unitprice",
    "menu.unitprice",
    "menu.vatyn",
    "sub_total.discount_price",
    "sub_total.etc",
    "sub_total.othersvc_price",
    "sub_total.service_price",
    "sub_total.subtotal_price",
    "sub_total.tax_price",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.emoneyprice",
    "total.menuqty_cnt",
    "total.menutype_cnt",
    "total.total_etc",
    "total.total_price",
    "void_menu.nm",
    "void_menu.price",
    "sub.nm",
    "sub.cnt",
]

# Super-class mapping (for DIL scenario)
SUPERCLASS_MAP = {
    "menu": ["menu.cnt", "menu.discountprice", "menu.itemsubtotal", "menu.nm",
             "menu.num", "menu.price", "menu.unitprice", "menu.vatyn",
             "menu.sub.cnt", "menu.sub.nm", "menu.sub.price", "menu.sub.unitprice"],
    "sub_total": ["sub_total.discount_price", "sub_total.etc", "sub_total.othersvc_price",
                  "sub_total.service_price", "sub_total.subtotal_price", "sub_total.tax_price"],
    "total": ["total.cashprice", "total.changeprice", "total.creditcardprice",
              "total.emoneyprice", "total.menuqty_cnt", "total.menutype_cnt",
              "total.total_etc", "total.total_price"],
    "void_menu": ["void_menu.nm", "void_menu.price"],
    "sub": ["sub.nm", "sub.cnt"],
}


def _build_bio_labels(fine_class_names: list[str]) -> list[str]:
    """Build BIO label list from class names. Always starts with 'O'."""
    labels = ["O"]
    for cls in fine_class_names:
        labels.append(f"B-{cls}")
        labels.append(f"I-{cls}")
    return labels


class CORDDataset(Dataset):
    """CORD-v2 receipt KIE dataset.

    Granularity:
        - "fine" (default): 30 classes → 61 BIO tags
        - "super": 5 super-classes → 11 BIO tags (used for DIL scenario)
    """

    LABEL_NAMES_FINE = _build_bio_labels(CORD_FINE_LABELS)  # 61 tags
    LABEL_NAMES_SUPER = _build_bio_labels(list(SUPERCLASS_MAP.keys()))  # 11 tags

    def __init__(
        self,
        split: str = "train",
        granularity: str = "fine",
        processor: LayoutLMv3Processor | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
    ):
        if granularity not in ("fine", "super"):
            raise ValueError(f"granularity must be 'fine' or 'super', got {granularity}")
        if split == "validation":
            split = "validation"  # CORD uses "validation" not "val"

        self.split = split
        self.granularity = granularity
        self.max_length = max_length
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        self.label_names = (
            self.LABEL_NAMES_FINE if granularity == "fine" else self.LABEL_NAMES_SUPER
        )
        self.label_to_id = {l: i for i, l in enumerate(self.label_names)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        # Load raw CORD-v2 (contains image + JSON ground truth)
        ds = load_dataset("naver-clova-ix/cord-v2", split=split, trust_remote_code=True)
        self.data = self._parse_examples(list(ds))

        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _parse_examples(self, raw: list[dict]) -> list[dict]:
        """Parse CORD-v2 raw format into flat (tokens, boxes, labels) per example.

        CORD-v2 ground truth is JSON-structured; we need to flatten to BIO sequence.
        Each example has:
            - "image": PIL Image
            - "ground_truth": JSON string with "valid_line" list of regions
        """
        parsed = []
        for ex in raw:
            image = ex["image"]
            gt = json.loads(ex["ground_truth"])
            tokens, boxes, labels = self._flatten_gt(gt, image.size)
            if not tokens:
                continue
            parsed.append({
                "image": image,
                "tokens": tokens,
                "bboxes": boxes,
                "ner_tags": labels,
            })
        return parsed

    def _flatten_gt(
        self, gt: dict, image_size: tuple[int, int]
    ) -> tuple[list[str], list[list[int]], list[int]]:
        """Convert CORD JSON ground-truth into flat BIO sequence.

        Returns (tokens, boxes [normalized to 0-1000], label_ids).
        """
        W, H = image_size
        tokens: list[str] = []
        boxes: list[list[int]] = []
        labels: list[int] = []

        for line in gt.get("valid_line", []):
            category = line.get("category", "other")
            # Map to coarse category if super granularity
            label_class = self._map_category(category)
            words = line.get("words", [])
            for w_idx, w in enumerate(words):
                text = w.get("text", "").strip()
                if not text:
                    continue
                quad = w.get("quad", {})
                # CORD quad: x1,y1 x2,y2 x3,y3 x4,y4 (clockwise from top-left)
                # Convert to axis-aligned bbox
                xs = [quad.get(f"x{i}", 0) for i in range(1, 5)]
                ys = [quad.get(f"y{i}", 0) for i in range(1, 5)]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                # Normalize to [0, 1000] and clamp: CORD quads occasionally carry
                # slightly out-of-frame coords (negative or > W/H from annotation
                # noise), which become negative / >1000 ids and trip LayoutLMv3's
                # 2D position-embedding bounds (valid [0, 1023]) -> CUDA device-side
                # assert. FUNSD/SROIE are pre-clamped upstream; CORD is not.
                box = [
                    min(1000, max(0, int(1000 * x0 / W))),
                    min(1000, max(0, int(1000 * y0 / H))),
                    min(1000, max(0, int(1000 * x1 / W))),
                    min(1000, max(0, int(1000 * y1 / H))),
                ]
                # BIO tag
                if label_class is None:
                    label_id = self.label_to_id["O"]
                else:
                    prefix = "B" if w_idx == 0 else "I"
                    tag = f"{prefix}-{label_class}"
                    label_id = self.label_to_id.get(tag, self.label_to_id["O"])

                tokens.append(text)
                boxes.append(box)
                labels.append(label_id)

        return tokens, boxes, labels

    def _map_category(self, category: str) -> str | None:
        """Map CORD category string to our label space.

        Returns None if category should be 'O' (unknown/other).
        """
        if not category or category == "other":
            return None

        if self.granularity == "fine":
            return category if category in CORD_FINE_LABELS else None
        else:
            for super_cls, fine_list in SUPERCLASS_MAP.items():
                if category in fine_list:
                    return super_cls
            return None

    def _filter_by_labels(self, data: list, label_filter: list[str]) -> list:
        keep_ids = {self.label_to_id[l] for l in label_filter if l in self.label_to_id}
        keep_ids.add(self.label_to_id["O"])
        target_entity_ids = keep_ids - {self.label_to_id["O"]}

        filtered = []
        for ex in data:
            tag_set = set(ex["ner_tags"])
            if tag_set.issubset(keep_ids) and tag_set & target_entity_ids:
                filtered.append(ex)
        return filtered

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.data[idx]
        encoding = self.processor(
            ex["image"],
            ex["tokens"],
            boxes=ex["bboxes"],
            word_labels=ex["ner_tags"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
