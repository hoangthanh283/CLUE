"""ViT image-classification backbone for the image-CL scope test.

Satisfies the slice of the wrapper contract the classical methods use
(``forward`` → ``.loss``/``.logits``, ``expand_classifier``, ``param_groups``,
``enable_gradient_checkpointing``, ``encode_query``/``token_features``) with a
CLS-token linear head. ``task_type = "image"`` switches the eval path from seqeval
span-F1 to top-1 accuracy. Prompt injection is not supported here.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import ViTForImageClassification

from doccl.models.base_wrapper import TokenClassificationWrapper


class ViTWrapper(TokenClassificationWrapper):
    _HF_CLS = ViTForImageClassification
    _inner_attr = "vit"
    _has_image = True
    task_type = "image"

    def __init__(
        self, model_name: str = "google/vit-base-patch16-224-in21k", num_labels: int = 10, **kw
    ):
        super().__init__(model_name=model_name, num_labels=num_labels, **kw)

    def forward(  # type: ignore[override]
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,  # noqa: ARG002 — tolerate loop extras (attention_mask etc.)
    ) -> Any:
        return self.model(pixel_values=pixel_values, labels=labels)

    @torch.no_grad()
    def encode_query(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._inner(pixel_values=batch["pixel_values"]).last_hidden_state[:, 0]

    def token_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """CLS feature feeding the head, kept 3-D (B, 1, D) for the token-style consumers."""
        return self._inner(pixel_values=batch["pixel_values"]).last_hidden_state[:, :1]

    def forward_with_prompts(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ARG002
        raise NotImplementedError("prompt methods are not wired for ViTWrapper")
