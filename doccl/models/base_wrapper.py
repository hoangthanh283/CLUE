"""Shared base for secondary text+layout token-classification backbones.

Captures the backbone-agnostic slice of the ``LayoutLMv3Wrapper`` interface that
the CL loop, the methods, and the metrics rely on (``expand_classifier``,
``forward``, ``forward_with_prompts``, ``encode_query``, ``param_groups`` /
``param_groups_by_depth``, ``get_layout_signature``, freeze / checkpoint controls,
param counts) so a new backbone is a drop-in once it provides three things:

    ``_HF_CLS``     the HuggingFace ``*ForTokenClassification`` class to load
    ``_inner_attr`` the encoder submodule name (e.g. ``"lilt"``, ``"bros"``)
    ``_pos_offset`` absolute-position pad offset (2 for RoBERTa/XLM-R, 0 for BERT)

The defaults here are for **vision-free** text+layout encoders (LiLT, BROS): the
``forward`` ignores ``pixel_values`` / ``image`` / ``modality_mask`` (the tolerant
``bert_wrapper`` pattern), and prompt injection prepends prompt vectors to the text
stream with dummy bboxes and no image. An image-bearing backbone (e.g. LayoutXLM)
can subclass and set ``_has_image = True`` + override ``forward`` /
``forward_with_prompts`` to thread the visual stream.

``LayoutLMv3Wrapper`` deliberately does **not** inherit this — it predates the base
and its 54 validated runs must stay reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from doccl.models import param_grouping

logger = logging.getLogger(__name__)


class TokenClassificationWrapper(nn.Module):
    """Backbone-agnostic wrapper base (vision-free defaults)."""

    # ─── subclass contract ──────────────────────────────────────────────────────
    _HF_CLS: type | None = None  # HF *ForTokenClassification class
    _inner_attr: str = ""  # encoder submodule name on the HF model
    _pos_offset: int = 2  # absolute-position pad offset for prompt truncation
    _has_image: bool = False  # vision-free by default

    def __init__(
        self,
        model_name: str,
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if self._HF_CLS is None or not self._inner_attr:
            raise NotImplementedError(f"{type(self).__name__} must set _HF_CLS and _inner_attr")
        self.model_name = model_name
        self.model = self._HF_CLS.from_pretrained(model_name, num_labels=num_labels)
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Force a plain Linear head so class-incremental expansion never hits a
        # saturating MLP head (same reasoning as LayoutLMv3Wrapper._force_linear_head).
        self._force_linear_head()

        # Label maps (mutated by expand_classifier during CIL)
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        # Extension state populated by methods (L2P/LAPP prompts, O-LoRA/H-LoRA banks)
        self._injected_prompts: dict[int, torch.Tensor] = {}
        self._lora_banks: dict[str, nn.Module] = {}

        if freeze_backbone:
            self.freeze_backbone()

    # ─── inner encoder accessor (delegates through PeftModel for O-LoRA) ─────────
    @property
    def _inner(self) -> nn.Module:
        return getattr(self.model, self._inner_attr)

    # ─── head: force plain Linear ───────────────────────────────────────────────
    def _force_linear_head(self) -> None:
        head = self.model.classifier
        if isinstance(head, nn.Linear):
            return
        out = getattr(head, "out_proj", None)
        n = out.out_features if out is not None else self.model.config.num_labels
        device = out.weight.device if out is not None else next(self.model.parameters()).device
        linear = nn.Linear(self.hidden_size, n).to(device)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)
        self.model.classifier = linear

    # ─── class-incremental: expand classifier ───────────────────────────────────
    def expand_classifier(self, new_labels: list[str]) -> None:
        """Widen the final output Linear, preserving old logits + label maps."""
        old_labels = list(self.id_to_label.values())
        all_labels = old_labels + [lbl for lbl in new_labels if lbl not in self.label_to_id]
        new_n = len(all_labels)

        # Under PEFT (O-LoRA / CL-LoRA) the classifier is auto-registered in
        # ``modules_to_save`` and REPLACED with a ``ModulesToSaveWrapper`` — which has
        # neither ``.out_proj`` nor ``.in_features``, so reading dims off it raises an
        # opaque AttributeError mid-CIL. Unwrap to the real head it holds (the
        # active-adapter copy is what the forward uses; fall back to
        # ``original_module``), expand that Linear, then write the widened copy back
        # into every internal reference so PEFT's forward sees the new width. This
        # mirrors ``LayoutLMv3Wrapper.expand_classifier`` so LoRA + class-IL runs on
        # every backbone, not just LayoutLMv3.
        head = self.model.classifier
        saved = getattr(head, "modules_to_save", None)
        if saved is not None:  # ModulesToSaveWrapper
            active = getattr(head, "active_adapter", None)
            if isinstance(active, (list, tuple)):
                active = active[0] if active else None
            keys = list(saved.keys())
            if active is not None and active in keys:
                head = saved[active]
            elif keys:
                head = saved[keys[0]]
            else:
                head = getattr(self.model.classifier, "original_module", head)

        out_linear = head.out_proj if hasattr(head, "out_proj") else head
        new_linear = nn.Linear(out_linear.in_features, new_n).to(out_linear.weight.device)
        with torch.no_grad():
            if old_labels:
                new_linear.weight[: len(old_labels)] = out_linear.weight
                new_linear.bias[: len(old_labels)] = out_linear.bias
            nn.init.normal_(new_linear.weight[len(old_labels) :], std=0.02)
            nn.init.zeros_(new_linear.bias[len(old_labels) :])

        if hasattr(head, "out_proj"):
            head.out_proj = new_linear
        elif saved is not None:
            # Re-point the PEFT wrapper's copies at the widened Linear so the forward
            # (which reads modules_to_save[active] / original_module) uses the new width.
            for adapter in list(saved.keys()):
                saved[adapter] = new_linear
            if getattr(self.model.classifier, "original_module", None) is not None:
                self.model.classifier.original_module = new_linear
        else:
            self.model.classifier = new_linear
        self.model.config.num_labels = new_n
        self.model.num_labels = new_n
        self.id_to_label = {i: lbl for i, lbl in enumerate(all_labels)}
        self.label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}

    # ─── layout signature (shared, backbone-agnostic — operates on bboxes) ───────
    @staticmethod
    def get_layout_signature(boxes: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
        """Histogram of box centers on a grid_size × grid_size grid. (B, grid_size²)."""
        B, N, _ = boxes.shape
        valid = (boxes.sum(dim=-1) > 0).float()
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        bin_x = (cx / 1000 * grid_size).clamp(0, grid_size - 1).long()
        bin_y = (cy / 1000 * grid_size).clamp(0, grid_size - 1).long()
        bin_idx = bin_y * grid_size + bin_x
        hist = torch.zeros(B, grid_size * grid_size, device=boxes.device)
        for b in range(B):
            mask = valid[b].bool()
            if mask.any():
                idxs = bin_idx[b][mask]
                hist[b].scatter_add_(0, idxs, torch.ones_like(idxs, dtype=torch.float))
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        return hist

    # ─── forward (vision-free; tolerant of extra streams) ───────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,  # noqa: ARG002 — ignored (no vision)
        image: torch.Tensor | None = None,  # noqa: ARG002 — ignored (no vision)
        modality_mask: Any = None,  # noqa: ARG002 — ignored (no pilot masking)
        **kwargs: Any,  # tolerate extra keys from the shared loop
    ) -> Any:
        return self.model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, labels=labels
        )

    # ─── CLS query for prompt-based methods (replaces reaching into internals) ───
    @torch.no_grad()
    def encode_query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """q(x) = CLS embedding from the frozen encoder. (B, D)."""
        keys = ["input_ids", "bbox", "attention_mask"]
        if self._has_image:
            keys += ["pixel_values", "image"]
        inputs = {k: batch[k] for k in keys if k in batch}
        return self._inner(**inputs).last_hidden_state[:, 0]

    # ─── per-token encoder features (the input the classifier head consumes) ─────
    def token_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-token encoder hidden states feeding the classifier. (B, L, D).

        Backbone-agnostic mirror of ``LayoutLMv3Wrapper.token_features`` for the
        vision-free secondaries (LiLT/BROS/BERT): the inner encoder's
        ``last_hidden_state`` over the text positions ``[0, input_ids.shape[1])``.
        There are no trailing image patches here, so the slice is a no-op safeguard
        that keeps the returned length aligned to the token labels.
        ``classifier(token_features)`` reproduces the model's token logits. Used by
        the LCA and HGT methods to estimate per-class feature Gaussians / gradient
        subspaces and to run the head on sampled features.
        """
        keys = ["input_ids", "bbox", "attention_mask"]
        if self._has_image:
            keys += ["pixel_values", "image"]
        inputs = {k: batch[k] for k in keys if k in batch}
        seq_len = batch["input_ids"].shape[1]
        return self._inner(**inputs).last_hidden_state[:, :seq_len]

    # ─── prompt injection (vision-free; prepend to text stream) ──────────────────
    def forward_with_prompts(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,  # noqa: ARG002 — ignored (no vision)
        image: torch.Tensor | None = None,  # noqa: ARG002 — ignored (no vision)
        **kwargs: Any,
    ) -> torch.Tensor:
        """Prepend ``prompt_embeds`` to the text token embeddings; return token logits
        over the *used* (prompt-truncated) text positions, shape (B, L_used, n_labels).
        """
        B, L = input_ids.shape
        P = prompt_embeds.shape[1]
        inner = self._inner

        max_total = self.model.config.max_position_embeddings - self._pos_offset
        L_used = min(L, max_total - P)
        if L_used < L:
            input_ids = input_ids[:, :L_used]
            bbox = bbox[:, :L_used]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :L_used]

        word_embeds = inner.embeddings.word_embeddings(input_ids)  # (B, L_used, D)
        inputs_embeds = torch.cat(
            [prompt_embeds.to(word_embeds.dtype), word_embeds], dim=1
        )  # (B, P + L_used, D)

        dummy_bbox = torch.zeros(B, P, bbox.shape[-1], dtype=bbox.dtype, device=bbox.device)
        new_bbox = torch.cat([dummy_bbox, bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(B, L_used, dtype=torch.long, device=input_ids.device)
        prompt_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device)
        new_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = inner(inputs_embeds=inputs_embeds, bbox=new_bbox, attention_mask=new_mask)
        text_out = outputs[0][:, P : P + L_used]  # drop the P prompt slots
        logits = self.model.classifier(self.model.dropout(text_out))
        return logits

    # ─── CKA probe layers (depth points matched to LayoutLMv3 / BERT) ────────────
    @property
    def cka_layers(self) -> list[str]:
        last = self.num_layers - 1
        mid = self.num_layers // 2
        a = self._inner_attr
        return [
            f"model.{a}.embeddings",
            f"model.{a}.encoder.layer.0",
            f"model.{a}.encoder.layer.{mid}",
            f"model.{a}.encoder.layer.{last}",
            "model.classifier",
        ]

    # ─── parameter groups (shared vocabulary; absent groups stay empty) ──────────
    @property
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return param_grouping.param_groups(self.model)

    @property
    def param_groups_by_depth(self) -> dict[str, list[nn.Parameter]]:
        return param_grouping.param_groups_by_depth(self.model, self.num_layers)

    # ─── freeze / checkpoint controls ────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self._inner.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self._inner.parameters():
            p.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # older transformers without the kwargs argument
            self.model.gradient_checkpointing_enable()
        except ValueError as e:
            # Some backbones (e.g. BROS) don't implement checkpointing. It is only a
            # memory optimization, so warn and continue rather than crash the run —
            # these models are small enough to fit without it.
            logger.warning(
                "%s: gradient checkpointing unavailable (%s); continuing without it.",
                type(self).__name__,
                e,
            )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
