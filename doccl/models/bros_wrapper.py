"""BROS wrapper — text+layout backbone (no vision), order-robust.

BROS (Hong et al., AAAI 2022) encodes text + 2D spatial coordinates with no
image stream, using a BERT-style WordPiece vocabulary. The vision-free defaults
in ``TokenClassificationWrapper`` apply unchanged.

``_pos_offset = 0`` because BROS uses BERT-style absolute positions (0..L-1) with
no padding offset. ``BrosModel.forward`` accepts a 4-dim ``bbox`` and expands it
to the 8-dim corner form internally, so the encoder emits the same (B, L, 4)
boxes as every other backbone.
"""
from __future__ import annotations

from transformers import BrosForTokenClassification

from doccl.models.base_wrapper import TokenClassificationWrapper


class BROSWrapper(TokenClassificationWrapper):
    """Wrapper around HuggingFace ``BrosForTokenClassification``."""

    _HF_CLS = BrosForTokenClassification
    _inner_attr = "bros"
    _pos_offset = 0
    _has_image = False

    def __init__(
        self,
        model_name: str = "jinho8345/bros-base-uncased",
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__(model_name, num_labels=num_labels, freeze_backbone=freeze_backbone)
