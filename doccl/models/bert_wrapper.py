"""BERT-base wrapper: the true unimodal text baseline for the pilot study.

The earlier "text-only" condition (C1) was a *masked* LayoutLMv3, not a unimodal
encoder; the AC review (C1, M1) asks for both a real text-only LayoutLMv3 *and* a
genuine external text baseline so the characterization can show that the
multimodal encoder forgets *differently* from a unimodal one. This wrapper is
that external baseline.

It mirrors the slice of the ``LayoutLMv3Wrapper`` interface the pilot uses
(``forward``, ``expand_classifier``, ``param_groups``, ``param_groups_by_depth``,
``cka_layers``, freeze/checkpoint controls) so the same pilot loop drives both
backbones. It consumes ``input_ids`` + ``attention_mask`` + ``labels`` produced
by ``doccl.data.bert_adapter`` (BERT WordPiece tokenisation of the same
documents) and ignores any ``bbox`` / ``pixel_values`` / ``modality_mask`` passed
by the shared loop.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForTokenClassification

from doccl.models import param_grouping

logger = logging.getLogger(__name__)


class BertTokenClassificationWrapper(nn.Module):
    """Wrapper around HuggingFace ``BertForTokenClassification``.

    Exposes the subset of the LayoutLMv3 wrapper API the pilot relies on so the
    two backbones are interchangeable in ``run_pilot``.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.hidden_size = self.model.config.hidden_size  # 768 for base
        self.num_layers = self.model.config.num_hidden_layers  # 12 for base

        # Label maps (mutated by expand_classifier during CIL)
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        if freeze_backbone:
            self.freeze_backbone()

    # ─── Forward (ignores non-text inputs from the shared pilot loop) ──────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,  # noqa: ARG002 — ignored (unimodal)
        pixel_values: torch.Tensor | None = None,  # noqa: ARG002 — ignored
        modality_mask: Any = None,  # noqa: ARG002 — ignored (text only)
        **kwargs: Any,  # tolerate extra keys from the shared loop
    ) -> Any:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    # ─── Class-incremental: expand classifier (plain Linear head) ─────────────
    def expand_classifier(self, new_labels: list[str]) -> None:
        """Widen the token-classification head, preserving old logits."""
        old_labels = list(self.id_to_label.values())
        all_labels = old_labels + [l for l in new_labels if l not in self.label_to_id]
        new_n = len(all_labels)

        out_linear = self.model.classifier  # BertForTokenClassification head is Linear
        new_linear = nn.Linear(out_linear.in_features, new_n).to(out_linear.weight.device)
        with torch.no_grad():
            if old_labels:
                new_linear.weight[: len(old_labels)] = out_linear.weight
                new_linear.bias[: len(old_labels)] = out_linear.bias
            nn.init.normal_(new_linear.weight[len(old_labels):], std=0.02)
            nn.init.zeros_(new_linear.bias[len(old_labels):])

        self.model.classifier = new_linear
        self.model.config.num_labels = new_n
        self.model.num_labels = new_n
        self.id_to_label = {i: l for i, l in enumerate(all_labels)}
        self.label_to_id = {l: i for i, l in enumerate(all_labels)}

    # ─── CKA probe layers (depth points matched to LayoutLMv3) ─────────────────
    @property
    def cka_layers(self) -> list[str]:
        last = self.num_layers - 1
        mid = self.num_layers // 2
        return [
            "model.bert.embeddings",
            f"model.bert.encoder.layer.0",
            f"model.bert.encoder.layer.{mid}",
            f"model.bert.encoder.layer.{last}",
            "model.classifier",
        ]

    # ─── Parameter groups (shared vocabulary; layout/patch groups stay empty) ──
    @property
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return param_grouping.param_groups(self.model)

    @property
    def param_groups_by_depth(self) -> dict[str, list[nn.Parameter]]:
        return param_grouping.param_groups_by_depth(self.model, self.num_layers)

    # ─── Freeze / checkpoint controls ─────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.model.bert.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.bert.parameters():
            p.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.model.gradient_checkpointing_enable()

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
