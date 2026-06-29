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
import os
from typing import Any

import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from doccl.models import param_grouping
from doccl.types import ModalityMask

logger = logging.getLogger(__name__)

# Opt-in bf16 autocast (default OFF -> exact fp32, unchanged for laptop/CPU/tests).
# Set DOCCL_AMP=1 on a bf16-capable GPU (e.g. RTX 4090) for ~1.5-2x faster training.
# bf16 needs no GradScaler (full fp32 exponent range), so backward() stays unchanged.
_AMP_ENABLED = os.environ.get("DOCCL_AMP", "0") == "1"


from contextlib import contextmanager  # noqa: E402


@contextmanager
def _autocast_ctx(enabled: bool):
    """Yield True under a bf16 CUDA autocast when enabled, else False (no-op)."""
    if enabled:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield True
    else:
        yield False


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

        # LayoutLMv3 swaps in a 2-layer MLP head (dense -> tanh -> out_proj) once
        # num_labels >= 10. In class-incremental learning that head collapses to
        # predicting all-O after the first expand_classifier: the task-0-trained
        # ``dense`` saturates the tanh, so gradients to freshly-added class rows
        # vanish and the new classes are never learned (verified: a freshly-built
        # 25-way MLP head trains to F1~37, but the SAME head grown 13->25 collapses
        # to F1=0, predicting only O). Force a single Linear head — the architecture
        # the working <10-label runs already used — so CIL expansion just widens a
        # plain Linear with no saturating nonlinearity.
        self._force_linear_head()

        # Label maps (mutated by expand_classifier during CIL)
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        # Extension state populated by methods
        self._injected_prompts: dict[int, torch.Tensor] = {}
        self._lora_banks: dict[str, nn.Module] = {}

        if freeze_backbone:
            self.freeze_backbone()

    def _force_linear_head(self) -> None:
        """Replace LayoutLMv3's classifier with a single ``nn.Linear`` head.

        Ensures a non-saturating head for class-incremental expansion regardless of
        the initial label count (HF would otherwise use a 2-layer MLP head for
        ``num_labels >= 10``). Reuses the existing final projection's weights when the
        current head already is/contains a Linear of the right shape, else inits fresh.
        """
        head = self.model.classifier
        if isinstance(head, nn.Linear):
            return  # already a plain Linear head
        out = getattr(head, "out_proj", None)
        n = out.out_features if out is not None else self.model.config.num_labels
        device = out.weight.device if out is not None else next(self.model.parameters()).device
        linear = nn.Linear(self.hidden_size, n).to(device)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)
        self.model.classifier = linear

    def expand_classifier(self, new_labels: list[str]) -> None:
        """Extend the classification head to add new labels. Old logits preserved.

        LayoutLMv3 chooses its head type by label count (``modeling_layoutlmv3``):
        ``num_labels < 10`` → a plain ``nn.Linear``; ``num_labels >= 10`` →
        ``LayoutLMv3ClassificationHead`` (an MLP whose final projection is
        ``out_proj``). We grow only the FINAL output Linear in place — for the MLP
        head that is ``classifier.out_proj`` (the dense/intermediate layers are kept
        untouched), for the plain head it is ``classifier`` itself — so this works for
        either head type and across the 10-label boundary (the head stays whichever
        type it was initialised as; a wider Linear head is valid).
        """
        old_labels = list(self.id_to_label.values())
        all_labels = old_labels + [l for l in new_labels if l not in self.label_to_id]
        new_n = len(all_labels)

        # Under PEFT (O-LoRA / CL-LoRA) the classifier is auto-registered in
        # ``modules_to_save`` and REPLACED with a ``ModulesToSaveWrapper`` — which has
        # neither ``.out_proj`` nor ``.in_features``, so reading dims off it raises
        # ``AttributeError: 'ModulesToSaveWrapper' object has no attribute 'in_features'``.
        # Unwrap to the real head it holds: the active-adapter copy is what the forward
        # uses (fall back to ``original_module``). We expand that Linear and write the
        # widened copy back into EVERY internal reference the wrapper keeps, so PEFT's
        # forward sees the new width immediately (this subsumes the post-hoc re-sync in
        # OLoRA.before_task, which previously never ran because expansion crashed first).
        head = self.model.classifier
        saved = getattr(head, "modules_to_save", None)
        if saved is not None:  # ModulesToSaveWrapper
            # ``modules_to_save`` is an nn.ModuleDict (no ``.get``); index by key after a
            # membership check. active_adapter may be a str or (some PEFT versions) a list.
            active = getattr(head, "active_adapter", None)
            if isinstance(active, (list, tuple)):
                active = active[0] if active else None
            keys = list(saved.keys())
            if active is not None and active in keys:
                inner = saved[active]
            elif keys:
                inner = saved[keys[0]]
            else:
                inner = getattr(head, "original_module", head)
            head = inner

        # Locate the final output Linear regardless of head type.
        out_linear = head.out_proj if hasattr(head, "out_proj") else head

        new_linear = nn.Linear(out_linear.in_features, new_n).to(out_linear.weight.device)
        with torch.no_grad():
            if len(old_labels) > 0:
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
        # HF caches config.num_labels as an attribute at __init__ and uses the
        # cached copy in the loss reshape — update it too or loss crashes after
        # the first expansion (shape '[-1, old_n]' is invalid ...).
        self.model.num_labels = new_n
        self.id_to_label = {i: l for i, l in enumerate(all_labels)}
        self.label_to_id = {l: i for i, l in enumerate(all_labels)}

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
        modality_mask=TEXT_ONLY     → image AND layout masked; TEXT KEPT
                                      (C1: a real text-only LayoutLMv3)
        """
        masked_input_ids, masked_pixel_values, masked_bbox = self._apply_mask(
            input_ids, pixel_values, bbox, modality_mask
        )

        with self._autocast() as amp_on:
            out = self.model(
                input_ids=masked_input_ids,
                bbox=masked_bbox,
                pixel_values=masked_pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )
        # Cast logits/loss back to fp32 at the boundary so downstream method code
        # (DER++ MSE vs fp32 cached logits, LwF KD, EWC penalty, argmax) is unchanged
        # and never hits a bf16-vs-fp32 dtype mismatch. The speedup is in the bf16
        # matmuls inside the transformer; this final cast is negligible.
        if amp_on:
            if getattr(out, "logits", None) is not None:
                out.logits = out.logits.float()
            if getattr(out, "loss", None) is not None:
                out.loss = out.loss.float()
        return out

    def _autocast(self):
        """bf16 autocast context when DOCCL_AMP=1 on CUDA; else a no-op (exact fp32).

        Centralised here so every caller (all CL methods' train loops, eval, and
        forward_with_prompts) gets AMP without per-method changes. Forward-only;
        backward runs outside, and bf16 needs no loss scaling. The context yields True
        when AMP is active so callers cast outputs back to fp32 at the boundary.
        """
        return _autocast_ctx(_AMP_ENABLED and torch.cuda.is_available())

    def _apply_mask(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor,
        mask: ModalityMask,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zero out the *inactive* input streams for a pilot condition.

        We replace inactive text with [PAD] tokens and zero-out the inactive
        ``pixel_values``/``bbox`` rather than modifying the model, preserving
        architecture identity (same layers, same shapes) — critical for the
        pilot's apples-to-apples claim.

        Stream activity per condition (✓ = kept, ✗ = zeroed):

            mask          text  image  layout
            FULL           ✓     ✓      ✓
            TEXT_LAYOUT    ✓     ✗      ✓   (C3)
            IMAGE_LAYOUT   ✗     ✓      ✓   (C2)
            TEXT_ONLY      ✓     ✗      ✗   (C1 — text is KEPT)
        """
        if mask == ModalityMask.FULL:
            return input_ids, pixel_values, bbox

        new_input_ids = input_ids
        new_pixel_values = pixel_values
        new_bbox = bbox

        # Mask text only when the condition removes the text stream (C2). C1
        # (TEXT_ONLY) and C3 (TEXT_LAYOUT) keep real ``input_ids``.
        if mask == ModalityMask.IMAGE_LAYOUT:
            pad_id = self.processor.tokenizer.pad_token_id
            new_input_ids = torch.full_like(input_ids, pad_id)

        # Zero the image when the condition removes the visual stream (C1, C3).
        if mask in (ModalityMask.TEXT_LAYOUT, ModalityMask.TEXT_ONLY):
            new_pixel_values = torch.zeros_like(pixel_values)

        # Zero the layout only for the text-only condition (C1).
        if mask == ModalityMask.TEXT_ONLY:
            new_bbox = torch.zeros_like(bbox)

        return new_input_ids, new_pixel_values, new_bbox

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

        with self._autocast() as amp_on:
            outputs = lm(
                inputs_embeds=inputs_embeds,
                bbox=new_bbox,
                pixel_values=pixel_values,
                attention_mask=new_mask,
            )
            # Text tokens come first; drop the P prompt slots → (B, L_used, D)
            text_out = outputs[0][:, P : P + L_used]
            logits = self.model.classifier(self.model.dropout(text_out))
        # fp32 at the boundary so prompt-method CE/aux losses are unchanged.
        return logits.float() if amp_on else logits

    @torch.no_grad()
    def encode_query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """q(x) = CLS embedding from the frozen encoder. (B, D).

        Lets prompt-based methods obtain the query without reaching into the
        backbone-specific submodule, so the same ``_query`` works across backbones.
        """
        inputs = {
            k: batch[k]
            for k in ("input_ids", "bbox", "pixel_values", "attention_mask")
            if k in batch
        }
        return self.model.layoutlmv3(**inputs).last_hidden_state[:, 0]

    @torch.no_grad()
    def token_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-token encoder features — the input the classifier head consumes. (B, L, D).

        These are the ``sequence_output`` LayoutLMv3 feeds to ``classifier`` (the text-token
        hidden states span ``[0, input_ids.shape[1])`` of ``last_hidden_state``; the trailing
        image patches are excluded so the returned length matches the token labels). Used by
        the LCA method to estimate per-class feature Gaussians and to run the classifier on
        sampled features. ``classifier(token_features)`` reproduces the model's token logits.
        """
        inputs = {
            k: batch[k]
            for k in ("input_ids", "bbox", "pixel_values", "attention_mask")
            if k in batch
        }
        seq_len = batch["input_ids"].shape[1]
        return self.model.layoutlmv3(**inputs).last_hidden_state[:, :seq_len]

    @property
    def cka_layers(self) -> list[str]:
        """Module names probed for per-layer CKA, spanning input→depth→head.

        Shared depth points (embeddings, early/mid/late encoder, classifier) line
        up with the BERT baseline's so the LayoutLMv3-vs-unimodal depth-gradient
        contrast (review M1) is apples-to-apples; ``patch_embed`` is
        LayoutLMv3-only.
        """
        last = self.num_layers - 1
        mid = self.num_layers // 2
        return [
            "model.layoutlmv3.embeddings",
            "model.layoutlmv3.patch_embed",
            f"model.layoutlmv3.encoder.layer.0",
            f"model.layoutlmv3.encoder.layer.{mid}",
            f"model.layoutlmv3.encoder.layer.{last}",
            "model.classifier",
        ]

    @property
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named, fully-populated component groups (see ``param_grouping``)."""
        return param_grouping.param_groups(self.model)

    @property
    def param_groups_by_depth(self) -> dict[str, list[nn.Parameter]]:
        """Encoder parameters bucketed by depth (input/early/mid/late/head)."""
        return param_grouping.param_groups_by_depth(self.model, self.num_layers)

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

            def wrapped_forward(
                *args, _fwd=orig_forward, _mod=layer, **kwargs
            ):  # noqa: ANN002,ANN003,ANN202
                if _mod.training and torch.is_grad_enabled():
                    return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
                return _fwd(*args, **kwargs)

            layer.forward = wrapped_forward
            layer._doccl_ckpt_wrapped = True
        logger.info(
            "Enabled per-layer gradient checkpointing on %d encoder layers", len(encoder.layer)
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
