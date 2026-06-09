"""SROIE (Scanned Receipt OCR and Information Extraction) loader.

Reference:
    Huang et al., "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction",
    ICDAR 2019.

SROIE has 4 KIE fields (company, date, address, total) → 9 BIO tags.
973 receipts (626 train / 347 test).

Note: SROIE is not on HuggingFace officially. We use the community mirror
`mlpc-lab/sroie` or expect locally-prepared data in `data/sroie/`.

Format expected per example:
    {
        "image": PIL.Image,
        "tokens": list[str],
        "bboxes": list[[x0, y0, x1, y1]] (normalized to 0-1000),
        "ner_tags": list[int],
    }
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


class SROIEDataset(Dataset):
    """SROIE 4-field KIE dataset.

    Labels: O, B-COMPANY, I-COMPANY, B-DATE, I-DATE, B-ADDRESS, I-ADDRESS,
            B-TOTAL, I-TOTAL (9 tags).
    """

    LABEL_NAMES = [
        "O",
        "B-COMPANY",
        "I-COMPANY",
        "B-DATE",
        "I-DATE",
        "B-ADDRESS",
        "I-ADDRESS",
        "B-TOTAL",
        "I-TOTAL",
    ]

    NUM_LABELS = 9

    def __init__(
        self,
        split: str = "train",
        data_root: str | Path = "data/sroie",
        processor: LayoutLMv3Processor | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
    ):
        self.split = split
        self.max_length = max_length
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        self.label_to_id = {l: i for i, l in enumerate(self.LABEL_NAMES)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        # Load from local preprocessed JSON (created by scripts/prepare_sroie.py)
        data_root = Path(data_root)
        json_path = data_root / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"SROIE {split} data not found at {json_path}. "
                "Run `python scripts/prepare_sroie.py` first to download and preprocess."
            )

        with open(json_path) as f:
            self.data = json.load(f)
        self._image_root = data_root / split / "images"

        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

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
        image = Image.open(self._image_root / ex["image_filename"]).convert("RGB")
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
