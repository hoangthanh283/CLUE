"""BERT WordPiece adapter for the pilot's unimodal text baseline.

The pilot's LayoutLMv3 path tokenises each document with the RoBERTa-based
``LayoutLMv3Processor``. A genuine BERT baseline (review C1/M1) must use BERT's
own WordPiece vocabulary on the *same* documents — not a re-use of the RoBERTa
ids — so this adapter re-tokenises the raw ``tokens`` (words) + ``ner_tags`` that
every doccl dataset stores in ``self.data`` and re-aligns the BIO labels to
subwords (label on the first subword of each word, ``-100`` on continuations and
specials), exactly as the LayoutLMv3 processor's ``word_labels`` does.

Labels are emitted in the **same local id space** the LayoutLMv3 ``__getitem__``
emits, so ``run_pilot`` treats BERT and LayoutLMv3 identically and the C4-vs-BERT
contrast is apples-to-apples.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class BertKIEAdapter(Dataset):
    """Wrap a doccl KIE dataset and yield BERT token-classification tensors.

    Reads the underlying dataset's ``self.data[idx]`` (uniform across FUNSD /
    CORD / SROIE: ``tokens``, ``bboxes``, ``ner_tags``) and produces
    ``{input_ids, attention_mask, labels}`` for ``BertForTokenClassification``.

    Args:
        underlying: a doccl dataset exposing ``.data`` (list of dicts with
            ``tokens`` and ``ner_tags``).
        tokenizer: a *fast* BERT tokenizer (provides ``word_ids``).
        max_length: padded/truncated sequence length.
    """

    def __init__(self, underlying: Dataset, tokenizer: Any, max_length: int = 512):
        if not hasattr(underlying, "data"):
            raise TypeError(
                "BertKIEAdapter expects a doccl dataset exposing `.data`; got "
                f"{type(underlying).__name__}"
            )
        self.underlying = underlying
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Expose the underlying label maps so run_pilot can size/expand the head.
        self.id_to_label = getattr(underlying, "id_to_label", {})
        self.label_to_id = getattr(underlying, "label_to_id", {})

    def __len__(self) -> int:
        return len(self.underlying.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.underlying.data[idx]
        words = list(ex["tokens"])
        word_labels = list(ex["ner_tags"])

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        labels: list[int] = []
        prev_wid: int | None = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)  # special token ([CLS]/[SEP]/[PAD])
            elif wid != prev_wid:
                labels.append(word_labels[wid])  # first subword of a word
                prev_wid = wid
            else:
                labels.append(-100)  # continuation subword
        out = {k: v.squeeze(0) for k, v in enc.items()}
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out
