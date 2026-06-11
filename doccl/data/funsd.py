"""FUNSD dataset loader.

Source: nielsr/funsd-layoutlmv3 on HuggingFace (preprocessed for LayoutLMv3).
149 train / 50 test forms, 4 entity types (HEADER, QUESTION, ANSWER, OTHER)
in BIO scheme = 7 BIO tags.

Reference:
    Jaume et al., "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents",
    ICDAR-OST 2019, arXiv:1905.13538
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor


class FUNSDDataset(Dataset):
    """FUNSD dataset wrapper.

    Labels (BIO scheme):
        0: O
        1: B-HEADER     2: I-HEADER
        3: B-QUESTION   4: I-QUESTION
        5: B-ANSWER     6: I-ANSWER
    """

    LABEL_NAMES = [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]

    NUM_LABELS = 7

    def __init__(
        self,
        split: str = "train",
        processor: LayoutLMv3Processor | None = None,
        max_length: int = 512,
        label_filter: list[str] | None = None,
    ):
        """Args:
            split: "train" or "test"
            processor: LayoutLMv3Processor (created from microsoft/layoutlmv3-base if None)
            max_length: max token length (LayoutLMv3 supports up to 512)
            label_filter: if provided, only keep examples containing labels in this set.
                          Used for class-incremental scenario building.
        """
        self.split = split
        self.max_length = max_length
        self.processor = processor or LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

        ds = load_dataset("nielsr/funsd-layoutlmv3", split=split, trust_remote_code=True)
        self.data = list(ds)

        self.label_to_id = {l: i for i, l in enumerate(self.LABEL_NAMES)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        if label_filter is not None:
            self.data = self._filter_by_labels(self.data, label_filter)

    def _filter_by_labels(self, data: list, label_filter: list[str]) -> list:
        """Class-incremental split: keep any example with >=1 in-session entity, and
        mask out-of-session entity tokens to 'O'.

        A FUNSD form contains HEADER, QUESTION and ANSWER entities together, so a
        strict-subset filter keeps almost no documents. Standard CIL token-classification
        practice is to retain documents with at least one target-class entity and relabel
        every out-of-session entity token as background ('O') for this session.
        """
        o_id = self.label_to_id["O"]
        target_entity_ids = {
            self.label_to_id[l] for l in label_filter if l in self.label_to_id
        } - {o_id}

        filtered = []
        for ex in data:
            if not (set(ex["ner_tags"]) & target_entity_ids):
                continue
            masked_tags = [t if t in target_entity_ids else o_id for t in ex["ner_tags"]]
            filtered.append({**ex, "ner_tags": masked_tags})
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
