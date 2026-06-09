"""Learn to Prompt (L2P).

Reference: Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.

A frozen pre-trained backbone with a learnable prompt pool. For each input:
    1. Compute query q(x) = CLS embedding from frozen backbone
    2. Select top-K prompts from pool by cosine similarity with keys
    3. Prepend selected prompts to the input token sequence
    4. Forward through frozen backbone + trainable classifier

Loss = CE + lambda_key * key_pull (pull selected keys toward query).

The training loop, query extraction, prompt injection and label truncation are
shared via ``PromptBasedMethod``; L2P only defines the prompt selection rule.
The simplified input-prepend injection (rather than per-layer prefix tuning) is
the documented AAAI-Lite variant — see ``LayoutLMv3Wrapper.forward_with_prompts``.
"""
from __future__ import annotations

import torch

from doccl.methods.prompt_base import PromptBasedMethod, PromptPool

__all__ = ["L2P", "PromptPool"]


class L2P(PromptBasedMethod):
    """Learn to Prompt with frozen LayoutLMv3 backbone + trainable prompt pool."""

    name = "l2p"

    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        n_prompts = config.get("n_prompts", 10)
        prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 5)
        self.lambda_key = config.get("lambda_key", 0.5)
        self._register_prompt_module(
            "prompt_pool", PromptPool(n_prompts, prompt_length, hidden_dim)
        )

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected, key_pull = self.prompt_pool.select(query, self.top_k)  # (B, K, L_p, D)
        B, K, Lp, D = selected.shape
        prompt_embeds = selected.reshape(B, K * Lp, D)
        return prompt_embeds, self.lambda_key * key_pull
