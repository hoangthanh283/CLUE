"""LiLT wrapper — decoupled text+layout backbone (no vision), multilingual.

LiLT (Wang et al., ACL 2022) pairs a language model's text stream with a
language-independent layout stream. We use the multilingual XLM-R variant by
default (``nielsr/lilt-xlm-roberta-base``); the English ``SCUT-DLVCLab/
lilt-roberta-en-base`` is config-swappable. Text+layout only — no image — so the
vision-free defaults in ``TokenClassificationWrapper`` apply unchanged.

``_pos_offset = 2`` because the (XLM-)RoBERTa position table reserves padding
offsets, matching the LayoutLMv3 prompt-truncation convention.
"""
from __future__ import annotations

from transformers import LiltForTokenClassification

from doccl.models.base_wrapper import TokenClassificationWrapper


class LiLTWrapper(TokenClassificationWrapper):
    """Wrapper around HuggingFace ``LiltForTokenClassification``."""

    _HF_CLS = LiltForTokenClassification
    _inner_attr = "lilt"
    _pos_offset = 2
    _has_image = False

    def __init__(
        self,
        model_name: str = "nielsr/lilt-xlm-roberta-base",
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__(model_name, num_labels=num_labels, freeze_backbone=freeze_backbone)
