"""Per-backbone KIE tokenization encoders.

``input_ids`` are vocab-specific (LayoutLMv3 = RoBERTa BPE, LiLT = XLM-R
SentencePiece, BROS = BERT WordPiece), so the *same* raw document (words, boxes,
word-level BIO labels) must be tokenized into each backbone's own subword space
with boxes + labels re-aligned. An encoder is the per-backbone strategy for that.

- ``LayoutLMv3Encoder`` (default) reuses the HF ``LayoutLMv3Processor`` verbatim,
  preserving the validated primary pipeline byte-for-byte (image + bbox + RoBERTa
  ids). It is the only image-bearing encoder here.
- ``LiLTEncoder`` / ``BROSEncoder`` are vision-free: they tokenize with the
  backbone's own fast tokenizer (vocab guaranteed to match the model) and
  propagate word-level boxes + BIO labels to subwords — the ``bert_adapter``
  label-alignment pattern, extended to boxes (box on every subword; label on the
  first subword of each word, ``-100`` on continuations and specials).

``build_encoder(model_cfg)`` selects the encoder from ``model_cfg.family``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from transformers import AutoTokenizer, LayoutLMv3Processor

_DEFAULT_LAYOUTLMV3 = "microsoft/layoutlmv3-base"


@runtime_checkable
class KIEEncoder(Protocol):
    """Turns one raw document into a backbone-ready tensor dict."""

    has_image: bool

    def encode(
        self,
        image: Any,
        words: list[str],
        boxes: list[list[int]],
        word_labels: list[int],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]: ...


class LayoutLMv3Encoder:
    """Default multimodal encoder — reproduces the current loader call exactly."""

    has_image = True

    def __init__(self, model_name: str = _DEFAULT_LAYOUTLMV3):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)

    def encode(self, image, words, boxes, word_labels, max_length=512):
        encoding = self.processor(
            image,
            list(words),
            boxes=list(boxes),
            word_labels=list(word_labels),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


class _SubwordKIEEncoder:
    """Vision-free encoder: subword tokenization with box + BIO-label propagation.

    Uses the backbone's own *fast* tokenizer (``word_ids`` available) so ``input_ids``
    land in the correct vocabulary. Boxes are placed on every subword of a word
    (the layout stream needs a position for each token); labels follow the
    first-subword convention (``-100`` on continuations / specials), matching the
    LayoutLMv3 processor's ``word_labels`` behaviour.
    """

    has_image = False
    # Boxes arrive word-level in [0, 1000] (LayoutLM convention). LiLT keeps that
    # integer convention; BROS expects coordinates normalised to [0, 1] floats.
    _bbox_normalize = False

    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError(
                f"{tokenizer_name} did not yield a fast tokenizer; word_ids() is required "
                "for subword box/label alignment."
            )

    def encode(self, image, words, boxes, word_labels, max_length=512):  # noqa: ARG002 — no image
        enc = self.tokenizer(
            list(words),
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        bbox: list[list[float]] = []
        labels: list[int] = []
        prev_wid: int | None = None
        for wid in word_ids:
            if wid is None:  # special / padding token
                bbox.append([0, 0, 0, 0])
                labels.append(-100)
            else:
                bbox.append(list(boxes[wid]))  # box on every subword
                labels.append(word_labels[wid] if wid != prev_wid else -100)
                prev_wid = wid
        out = {k: v.squeeze(0) for k, v in enc.items()}
        if self._bbox_normalize:
            out["bbox"] = torch.tensor(bbox, dtype=torch.float) / 1000.0  # → [0, 1]
        else:
            out["bbox"] = torch.tensor(bbox, dtype=torch.long)
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out


class BERTEncoder(_SubwordKIEEncoder):
    """BERT-WordPiece text-only encoder for the unimodal baseline.

    Tokenises like the other subword encoders (so labels align to first subwords)
    and still emits a ``bbox`` for collate uniformity, but the boxes are inert:
    ``BERTWrapper.forward`` drops them, so this is a genuine text-only stream. Kept
    as its own class (vs reusing ``_SubwordKIEEncoder`` directly) so the encoder
    factory reads cleanly per family.
    """


class LiLTEncoder(_SubwordKIEEncoder):
    """XLM-R (or RoBERTa) box-aware subword encoder for LiLT."""


class BROSEncoder(_SubwordKIEEncoder):
    """BERT-WordPiece box-aware subword encoder for BROS.

    BROS consumes 4-dim boxes normalised to [0, 1] (it expands 4→8 corners and
    applies ``bbox_scale`` internally). Verify on the grid machine — a wrong
    convention shows up immediately as a near-zero BROS F1.
    """

    _bbox_normalize = True


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a field from an OmegaConf DictConfig or a plain dict."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ENCODER_BY_FAMILY = {
    "layoutlmv3": lambda cfg: LayoutLMv3Encoder(_cfg_get(cfg, "name", _DEFAULT_LAYOUTLMV3)),
    "lilt": lambda cfg: LiLTEncoder(_cfg_get(cfg, "tokenizer_name") or _cfg_get(cfg, "name")),
    "bros": lambda cfg: BROSEncoder(_cfg_get(cfg, "tokenizer_name") or _cfg_get(cfg, "name")),
    "bert": lambda cfg: BERTEncoder(_cfg_get(cfg, "tokenizer_name") or _cfg_get(cfg, "name")),
}


def build_encoder(model_cfg: Any) -> KIEEncoder:
    """Build the KIE encoder for ``model_cfg.family`` (defaults to LayoutLMv3)."""
    family = _cfg_get(model_cfg, "family", "layoutlmv3")
    if family not in _ENCODER_BY_FAMILY:
        raise ValueError(
            f"No encoder for model family {family!r}. Known: {list(_ENCODER_BY_FAMILY)}"
        )
    return _ENCODER_BY_FAMILY[family](model_cfg)


# ─── process-level default encoder ───────────────────────────────────────────────
# A run builds exactly one scenario for one backbone, so the active encoder is set
# once (by ``get_scenario``/``train.py``) and read by every dataset loader whose
# ``encoder`` arg is left None. Defaults lazily to LayoutLMv3 so existing callers
# (tests, pilot, analysis) are unaffected.
_DEFAULT_ENCODER: KIEEncoder | None = None


def set_default_encoder(encoder: KIEEncoder | None) -> None:
    """Set the process-wide default encoder used when a loader gets ``encoder=None``."""
    global _DEFAULT_ENCODER
    _DEFAULT_ENCODER = encoder


def get_default_encoder() -> KIEEncoder:
    """Return the process-wide default encoder (lazily LayoutLMv3)."""
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = LayoutLMv3Encoder()
    return _DEFAULT_ENCODER
