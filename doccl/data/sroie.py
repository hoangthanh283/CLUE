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
        source: str = "local",
        hf_name: str = "darentang/sroie",
    ):
        """Args:
            split: "train" or "test".
            data_root: local dir holding ``{split}.json`` (used when source="local").
            source: "local" (default; from ``scripts/prepare_sroie.py``) or "hf"
                (a community HuggingFace mirror — convenient for one-command cloud
                runs, but PROVENANCE MUST BE CONFIRMED: mirrors vary in field names
                and BIO-tag ordering. Default mirror ``darentang/sroie`` is assumed
                to align with ``LABEL_NAMES``; verify before citing in the thesis).
            hf_name: HuggingFace dataset id used when source="hf".
        """
        self.split = split
        self.max_length = max_length
        self.source = source
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        self.label_to_id = {l: i for i, l in enumerate(self.LABEL_NAMES)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}
        self._image_root: Path | None = None

        if source == "hf":
            self.data = self._load_hf(hf_name, split)
        elif source == "local":
            data_root = Path(data_root)
            json_path = data_root / f"{split}.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"SROIE {split} data not found at {json_path}. "
                    "Run `python scripts/prepare_sroie.py` first, or pass source='hf'."
                )
            with open(json_path) as f:
                self.data = json.load(f)
            self._image_root = data_root / split / "images"
        else:
            raise ValueError(f"Unknown SROIE source {source!r} (expected 'local' or 'hf').")

        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _load_hf(self, hf_name: str, split: str) -> list[dict]:
        """Load + normalise a HuggingFace SROIE mirror into our example schema.

        Normalises common field aliases (tokens/words, bboxes/boxes/bbox,
        ner_tags/labels/tags) and keeps the PIL image in-memory. Asserts the tag
        space matches the expected 9-BIO scheme so a mismatched mirror fails loud.
        """
        from datasets import load_dataset

        hf_split = {"train": "train", "test": "test"}.get(split, split)
        ds = load_dataset(hf_name, split=hf_split, trust_remote_code=True)

        def pick(ex: dict, *aliases: str):
            for a in aliases:
                if a in ex:
                    return ex[a]
            raise KeyError(f"None of {aliases} present in HF SROIE example (keys={list(ex)}).")

        out: list[dict] = []
        for ex in ds:
            tags = pick(ex, "ner_tags", "labels", "tags")
            if tags and isinstance(tags[0], str):  # string labels → map to ids
                tags = [self.label_to_id.get(t, 0) for t in tags]
            if tags and max(tags) >= self.NUM_LABELS:
                raise ValueError(
                    f"HF mirror {hf_name!r} has tag id {max(tags)} >= {self.NUM_LABELS}; "
                    "its BIO scheme does not match LABEL_NAMES — confirm provenance."
                )
            out.append(
                {
                    "image": pick(ex, "image", "img"),
                    "tokens": pick(ex, "tokens", "words"),
                    "bboxes": pick(ex, "bboxes", "boxes", "bbox"),
                    "ner_tags": tags,
                }
            )
        return out

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
        if "image" in ex:  # HF source: PIL image already in-memory
            image = ex["image"].convert("RGB")
        else:  # local source: open from disk
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
