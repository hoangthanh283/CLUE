"""LayoutLMv3 wrapper with extension points for CL methods and pilot study.

Extension points:
    - expand_classifier(new_labels): for CIL
    - get_layout_signature(boxes): for layout-aware methods (LAPP)
    - forward(..., modality_mask=...): for pilot conditions C2/C3
    - forward_with_prompts(...): for prompt-based methods (L2P/DualPrompt/CODA/routed)
    - param_groups: dict of named parameter groups for Fisher analysis
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from doccl.types import ModalityMask

logger = logging.getLogger(__name__)


class LayoutLMv3Wrapper(nn.Module):
    """Wrapper around HuggingFace LayoutLMv3.

    Exposes hooks needed by CL methods (prompt injection, LoRA banks, classifier
    expansion) and the pilot study (modality masking, parameter groups).
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        num_labels: int = 7,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name

        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.hidden_size = self.model.config.hidden_size  # 768 for base
        self.num_layers = self.model.config.num_hidden_layers  # 12 for base

        # Label maps (mutated by expand_classifier during CIL)
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        # Extension state populated by methods
        self._injected_prompts: dict[int, torch.Tensor] = {}
        self._lora_banks: dict[str, nn.Module] = {}

        if freeze_backbone:
            self.freeze_backbone()

    # ─── Class-incremental: expand classifier ──────────────────────────────────
    def expand_classifier(self, new_labels: list[str]) -> None:
        """Extend classification head to add new labels. Old logits preserved."""
        old_labels = list(self.id_to_label.values())
        all_labels = old_labels + [l for l in new_labels if l not in self.label_to_id]
        new_n = len(all_labels)

        old_clf = self.model.classifier
        new_clf = nn.Linear(self.hidden_size, new_n).to(old_clf.weight.device)

        with torch.no_grad():
            if len(old_labels) > 0:
                new_clf.weight[: len(old_labels)] = old_clf.weight
                new_clf.bias[: len(old_labels)] = old_clf.bias
            nn.init.normal_(new_clf.weight[len(old_labels):], std=0.02)
            nn.init.zeros_(new_clf.bias[len(old_labels):])

        self.model.classifier = new_clf
        self.model.config.num_labels = new_n
        # HF caches config.num_labels as an attribute at __init__ and uses the
        # cached copy in the loss reshape — update it too or loss crashes after
        # the first expansion (shape '[-1, old_n]' is invalid ...).
        self.model.num_labels = new_n
        self.id_to_label = {i: l for i, l in enumerate(all_labels)}
        self.label_to_id = {l: i for i, l in enumerate(all_labels)}

    # ─── Layout signature for LAPP and pilot analysis ──────────────────────────
    @staticmethod
    def get_layout_signature(boxes: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
        """Compute layout signature φ(boxes) as histogram of box centers on grid.

        Args:
            boxes: (B, N, 4) tensor [x0, y0, x1, y1] normalized to [0, 1000]
            grid_size: divide [0,1000]² into grid_size × grid_size bins

        Returns:
            (B, grid_size**2) normalized histogram
        """
        B, N, _ = boxes.shape
        valid = (boxes.sum(dim=-1) > 0).float()  # (B, N) — exclude padding

        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2

        bin_x = (cx / 1000 * grid_size).clamp(0, grid_size - 1).long()
        bin_y = (cy / 1000 * grid_size).clamp(0, grid_size - 1).long()
        bin_idx = bin_y * grid_size + bin_x  # (B, N)

        hist = torch.zeros(B, grid_size * grid_size, device=boxes.device)
        for b in range(B):
            mask = valid[b].bool()
            if mask.any():
                idxs = bin_idx[b][mask]
                hist[b].scatter_add_(0, idxs, torch.ones_like(idxs, dtype=torch.float))
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        return hist

    # ─── Forward with optional modality masking (for pilot C2/C3) ──────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        modality_mask: ModalityMask = ModalityMask.FULL,
    ) -> Any:
        """Forward pass. Modality masking zeros out specific input streams.

        modality_mask=FULL          → standard LayoutLMv3 (text + image + layout)
        modality_mask=TEXT_LAYOUT   → image masked (C3 in pilot)
        modality_mask=IMAGE_LAYOUT  → text masked (C2 in pilot)
        modality_mask=TEXT_ONLY     → image AND layout masked (degenerate, sanity)
        """
        masked_input_ids, masked_pixel_values, masked_bbox = self._apply_mask(
            input_ids, pixel_values, bbox, modality_mask
        )

        return self.model(
            input_ids=masked_input_ids,
            bbox=masked_bbox,
            pixel_values=masked_pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _apply_mask(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor,
        mask: ModalityMask,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero out specific modalities for pilot conditions.

        Implementation note: we replace text with [PAD] tokens and zero-out
        pixel_values/bbox rather than modifying the model. This preserves
        architecture identity (same layers, same shapes) — critical for the
        pilot's apples-to-apples claim.
        """
        if mask == ModalityMask.FULL:
            return input_ids, pixel_values, bbox

        new_input_ids = input_ids
        new_pixel_values = pixel_values
        new_bbox = bbox

        if mask in (ModalityMask.IMAGE_LAYOUT, ModalityMask.TEXT_ONLY):
            # Mask text: replace with PAD (id=1 in roberta tokenizer used by LayoutLMv3)
            pad_id = self.processor.tokenizer.pad_token_id
            new_input_ids = torch.full_like(input_ids, pad_id)

        if mask in (ModalityMask.TEXT_LAYOUT, ModalityMask.TEXT_ONLY):
            # Mask image: zero pixel values
            new_pixel_values = torch.zeros_like(pixel_values)

        if mask == ModalityMask.TEXT_ONLY:
            # Also mask layout
            new_bbox = torch.zeros_like(bbox)

        return new_input_ids, new_pixel_values, new_bbox

    # ─── Prompt injection (L2P / DualPrompt / CODA-Prompt / routed prompts) ─────
    def forward_with_prompts(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with ``prompt_embeds`` prepended to the text token embeddings.

        Shared injection path for all prompt-based methods. We embed ``input_ids``
        via the word-embedding table, prepend the ``(B, P, D)`` prompt vectors,
        prepend dummy bboxes (``[0,0,0,0]``) and attention-mask ones for the prompt
        slots, run the frozen LayoutLMv3 (text stream first, then visual patches),
        then **slice the prompt slots off** the text outputs before the token
        classifier.

        Because the datasets pad to ``max_length`` (512) and LayoutLMv3's absolute
        position table caps usable length at ``max_position_embeddings - 2`` (512
        for base), we truncate the text to ``max_total - P`` so ``P + L`` fits. The
        dropped tail is almost always padding. Returns per-token logits over the
        *used* text positions, shape ``(B, L_used, num_labels)`` — callers must
        align labels to ``logits.shape[1]`` (the prompt base class does this).
        """
        B, L = input_ids.shape
        P = prompt_embeds.shape[1]
        lm = self.model.layoutlmv3

        max_total = self.model.config.max_position_embeddings - 2  # RoBERTa pad offset
        L_used = min(L, max_total - P)
        if L_used < L:
            input_ids = input_ids[:, :L_used]
            bbox = bbox[:, :L_used]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :L_used]

        word_embeds = lm.embeddings.word_embeddings(input_ids)  # (B, L_used, D)
        inputs_embeds = torch.cat(
            [prompt_embeds.to(word_embeds.dtype), word_embeds], dim=1
        )  # (B, P + L_used, D)

        dummy_bbox = torch.zeros(B, P, 4, dtype=bbox.dtype, device=bbox.device)
        new_bbox = torch.cat([dummy_bbox, bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(B, L_used, dtype=torch.long, device=input_ids.device)
        prompt_mask = torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device)
        new_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = lm(
            inputs_embeds=inputs_embeds,
            bbox=new_bbox,
            pixel_values=pixel_values,
            attention_mask=new_mask,
        )
        # Text tokens come first; drop the P prompt slots → (B, L_used, D)
        text_out = outputs[0][:, P : P + L_used]
        logits = self.model.classifier(self.model.dropout(text_out))
        return logits

    # ─── Parameter groups for Fisher analysis (pilot study) ────────────────────
    @property
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named parameter groups used for per-component Fisher analysis.

        Groups align with the architectural components hypothesized to forget
        differently (text-attention, visual-attention, fusion, classifier).
        """
        groups: dict[str, list[nn.Parameter]] = {
            "text_word_embed": [],
            "layout_2d_pos_embed": [],
            "image_patch_embed": [],
            "text_attn": [],
            "visual_attn": [],
            "fusion": [],  # cross-modal attn projections
            "ffn": [],  # feedforward layers
            "classifier": [],
        }
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "word_embeddings" in name:
                groups["text_word_embed"].append(p)
            elif "x_position_embeddings" in name or "y_position_embeddings" in name or "h_position_embeddings" in name or "w_position_embeddings" in name:
                groups["layout_2d_pos_embed"].append(p)
            elif "patch_embed" in name:
                groups["image_patch_embed"].append(p)
            elif "classifier" in name:
                groups["classifier"].append(p)
            elif "attention" in name and ("query" in name or "key" in name or "value" in name):
                # All attention Q/K/V — distinguish text vs visual layers if possible
                # In LayoutLMv3 single-stream, text and image attend together — mark as "text_attn"
                # for now. Refine after inspecting actual module names.
                groups["text_attn"].append(p)
            elif "intermediate" in name or "output.dense" in name:
                groups["ffn"].append(p)
            else:
                # Catch-all for anything not matched
                groups.setdefault("other", []).append(p)

        # Drop empty groups
        return {k: v for k, v in groups.items() if v}

    # ─── Freeze controls ───────────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        """Freeze everything except classifier head. Used by prompt-only methods."""
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for memory: recompute activations in the backward pass
        instead of storing them. Essential for full fine-tuning on limited-VRAM
        GPUs. Uses non-reentrant checkpointing so it also works when the backbone
        is frozen (prompt methods). ``self.model`` may be a PeftModel (O-LoRA /
        DocCL-A); both delegate this call to the base model.

        transformers >= 4.50 removed the checkpointing path from the LayoutLMv3
        encoder (``supports_gradient_checkpointing`` is False and the layer loop
        has no ``_gradient_checkpointing_func`` branch), so when the native call
        is unavailable we wrap each encoder layer's ``forward`` with
        ``torch.utils.checkpoint`` ourselves. Non-reentrant checkpointing accepts
        kwargs (``rel_pos``/``rel_2d_pos``) and tolerates frozen inputs.
        """
        if getattr(self.model, "supports_gradient_checkpointing", False):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:  # older transformers without the kwargs argument
                self.model.gradient_checkpointing_enable()
            return

        from torch.utils.checkpoint import checkpoint

        encoder = self.model.layoutlmv3.encoder  # PeftModel delegates getattr
        for layer in encoder.layer:
            if getattr(layer, "_doccl_ckpt_wrapped", False):
                continue
            orig_forward = layer.forward

            def wrapped_forward(*args, _fwd=orig_forward, _mod=layer, **kwargs):  # noqa: ANN002,ANN003,ANN202
                if _mod.training and torch.is_grad_enabled():
                    return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
                return _fwd(*args, **kwargs)

            layer.forward = wrapped_forward
            layer._doccl_ckpt_wrapped = True
        logger.info("Enabled per-layer gradient checkpointing on %d encoder layers", len(encoder.layer))

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
