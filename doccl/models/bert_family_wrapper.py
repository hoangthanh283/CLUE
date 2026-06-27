"""BERT-base wrapper as a first-class CL backbone (``family: bert``).

The external **text-only** comparator for the main grid. Where ``bros_wrapper`` /
``lilt_wrapper`` are text+layout (they consume ``bbox``), this is genuinely
unimodal: ``BertForTokenClassification`` takes only ``input_ids`` /
``attention_mask`` / ``labels``. It therefore overrides the text+layout ``forward``
/ ``forward_with_prompts`` / ``encode_query`` of ``TokenClassificationWrapper`` to
drop the layout stream, while inheriting the class-incremental head growth,
parameter grouping, and freeze/checkpoint controls unchanged.

This supersedes the pilot-only ``bert_wrapper.BertTokenClassificationWrapper`` for
grid use: same backbone, but slotted into ``MODEL_REGISTRY`` + ``build_encoder`` so
``method=naive model=bert_base`` runs any scenario through the standard CL loop.

``_pos_offset = 0`` — BERT uses absolute positions 0..L-1 with no pad offset.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import BertForTokenClassification

from doccl.models.base_wrapper import TokenClassificationWrapper


class BERTWrapper(TokenClassificationWrapper):
    """Text-only ``BertForTokenClassification`` wrapped for the CL loop."""

    _HF_CLS = BertForTokenClassification
    _inner_attr = "bert"
    _pos_offset = 0
    _has_image = False

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__(model_name, num_labels=num_labels, freeze_backbone=freeze_backbone)

    # ─── forward (unimodal: no bbox / pixel_values / image) ──────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,  # noqa: ARG002 — ignored (text only)
        pixel_values: torch.Tensor | None = None,  # noqa: ARG002 — ignored
        image: torch.Tensor | None = None,  # noqa: ARG002 — ignored
        modality_mask: Any = None,  # noqa: ARG002 — ignored
        **kwargs: Any,  # tolerate extra keys from the shared loop
    ) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # ─── CLS query for prompt methods (text-only) ────────────────────────────────
    @torch.no_grad()
    def encode_query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """q(x) = CLS embedding from the frozen BERT encoder. (B, D)."""
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask") if k in batch}
        return self._inner(**inputs).last_hidden_state[:, 0]

    # ─── per-token encoder features (text-only: BertModel rejects bbox) ───────────
    def token_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-token BERT hidden states feeding the classifier. (B, L, D).

        Text-only override of the base ``token_features``: the raw ``BertModel`` does
        not accept ``bbox``, so we pass only ``input_ids``/``attention_mask`` (mirrors
        ``encode_query``). ``classifier(token_features)`` reproduces the token logits.
        """
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask") if k in batch}
        seq_len = batch["input_ids"].shape[1]
        return self._inner(**inputs).last_hidden_state[:, :seq_len]

    # ─── prompt injection (text-only: prepend to token embeddings, no bbox) ───────
    def forward_with_prompts(
        self,
        input_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,  # noqa: ARG002 — ignored (text only)
        pixel_values: torch.Tensor | None = None,  # noqa: ARG002 — ignored
        image: torch.Tensor | None = None,  # noqa: ARG002 — ignored
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prepend ``prompt_embeds`` to the BERT token embeddings; return token logits
        over the prompt-truncated text positions, shape (B, L_used, n_labels).
        """
        B, L = input_ids.shape
        P = prompt_embeds.shape[1]
        inner = self._inner

        max_total = self.model.config.max_position_embeddings - self._pos_offset
        L_used = min(L, max_total - P)
        if L_used < L:
            input_ids = input_ids[:, :L_used]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :L_used]

        word_embeds = inner.embeddings.word_embeddings(input_ids)  # (B, L_used, D)
        inputs_embeds = torch.cat([prompt_embeds.to(word_embeds.dtype), word_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(B, L_used, dtype=torch.long, device=input_ids.device)
        prompt_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device)
        new_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = inner(inputs_embeds=inputs_embeds, attention_mask=new_mask)
        text_out = outputs[0][:, P : P + L_used]  # drop the P prompt slots
        logits = self.model.classifier(self.model.dropout(text_out))
        return logits
