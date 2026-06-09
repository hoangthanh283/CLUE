"""CODA-Prompt.

Reference: Smith et al., "CODA-Prompt: COntinual Decomposed Attention-based
Prompting for Rehearsal-Free Continual Learning", CVPR 2023.

Instead of hard top-K selection (L2P) or a discrete expert (DualPrompt), CODA-Prompt
forms an input-conditioned *weighted combination* of M prompt components — fully
differentiable end-to-end. For query q and component m with key K_m and
attention vector A_m:

    w_m   = cos( q ⊙ A_m , K_m )            (feature-selective attention weight)
    P(q)  = Σ_m  w_m · P_m                   (weighted sum of components)

An orthogonality penalty on the keys and attention vectors discourages
inter-component (hence inter-task) interference. Prepended via
``LayoutLMv3Wrapper.forward_with_prompts`` (AAAI-Lite input-prepend variant).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from doccl.methods.prompt_base import PromptBasedMethod


class _CodaModule(nn.Module):
    """Decomposed attention prompt: components P, keys K, attention vectors A."""

    def __init__(self, n_components: int = 10, prompt_length: int = 5, hidden: int = 768):
        super().__init__()
        self.P = nn.Parameter(torch.randn(n_components, prompt_length, hidden) * 0.02)
        self.K = nn.Parameter(torch.randn(n_components, hidden) * 0.02)
        self.A = nn.Parameter(torch.randn(n_components, hidden) * 0.02)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # query: (B, D) → feature-selective query a_q = q ⊙ A_m
        a_q = query.unsqueeze(1) * self.A.unsqueeze(0)  # (B, M, D)
        w = (F.normalize(a_q, dim=-1) * F.normalize(self.K, dim=-1).unsqueeze(0)).sum(-1)  # (B, M)
        prompt = torch.einsum("bm,mld->bld", w, self.P)  # (B, L_p, D)
        return prompt

    def ortho_penalty(self) -> torch.Tensor:
        def off_diag(x: torch.Tensor) -> torch.Tensor:
            xn = F.normalize(x, dim=-1)
            gram = xn @ xn.T
            eye = torch.eye(gram.shape[0], device=gram.device)
            return (gram - eye).pow(2).sum()

        return off_diag(self.K) + off_diag(self.A)


class CODAPrompt(PromptBasedMethod):
    """CODA-Prompt: attention-weighted decomposed prompt components."""

    name = "coda_prompt"

    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        n_components = config.get("n_components", 10)
        prompt_length = config.get("prompt_length", 5)
        self.lambda_ortho = config.get("lambda_ortho", 0.1)
        self._register_prompt_module(
            "coda", _CodaModule(n_components, prompt_length, hidden_dim)
        )

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = self.coda(query)  # (B, L_p, D)
        aux = self.lambda_ortho * self.coda.ortho_penalty()
        return prompt_embeds, aux
