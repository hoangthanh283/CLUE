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

    def __init__(
        self,
        underlying: Dataset,
        tokenizer: Any,
        max_length: int = 512,
        label_remap: dict[int, int] | None = None,
    ):
        # Unwrap label-remapping scenario wrappers (DIL_LabelRemapper / CIL_LabelRemapper):
        # they hold the per-document words+ner_tags on their wrapped `.underlying.data`,
        # not on themselves. We need the RAW words/ner_tags here to re-tokenise with BERT
        # WordPiece, and a `label_remap` (native_id -> unified_id) to translate the tags
        # into the SAME label space the model head is sized to. Without this, the BERT
        # baseline was trained on native ids that don't match the (remapped) head and
        # collapsed to all-O (0 entity-F1) — the cause of the degenerate Cb runs.
        base = underlying
        while not hasattr(base, "data") and hasattr(base, "underlying"):
            base = base.underlying
        if not hasattr(base, "data"):
            raise TypeError(
                "BertKIEAdapter expects a doccl dataset exposing `.data` (directly or via "
                f"a remapper's `.underlying`); got {type(underlying).__name__}"
            )
        self.underlying = base
        self.tokenizer = tokenizer
        self.max_length = max_length
        # native_id -> target(unified)_id translation for ner_tags. Identity when None.
        self.label_remap = label_remap or {}
        # Expose the BASE dataset's label maps (run_pilot overrides the head sizing
        # from scenario.tasks[0].label_set, so these are only informational here).
        self.id_to_label = getattr(base, "id_to_label", {})
        self.label_to_id = getattr(base, "label_to_id", {})

    def __len__(self) -> int:
        return len(self.underlying.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.underlying.data[idx]
        words = list(ex["tokens"])
        # Translate native ner_tag ids into the model's (unified) label space; tags with
        # no mapping fall back to their native id (identity remap leaves them unchanged).
        word_labels = [self.label_remap.get(t, t) for t in ex["ner_tags"]]

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
