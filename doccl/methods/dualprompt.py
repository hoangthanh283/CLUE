"""DualPrompt.

Reference: Wang et al., "DualPrompt: Complementary Prompting for Rehearsal-free
Continual Learning", ECCV 2022.

Two prompt families on a frozen backbone:
    * G-Prompt (general): a single prompt shared across all tasks, updated every
      task — captures task-invariant knowledge.
    * E-Prompt (expert): one task-specific prompt per task. During task t only
      expert t (and its key) is trained; its key is pulled toward the query. At
      test time the nearest expert key (over seen tasks) selects the expert.

Both are prepended to the token sequence (the AAAI-Lite input-prepend variant via
``LayoutLMv3Wrapper.forward_with_prompts``). Only the current expert is in the
optimiser each task, so past experts stay frozen — the key DualPrompt property.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from doccl.methods.prompt_base import PromptBasedMethod
from doccl.types import TaskInfo


class _DualPromptModules(nn.Module):
    """G-prompt (shared) + per-task E-prompts and their selection keys."""

    def __init__(self, n_experts: int, g_len: int, e_len: int, hidden: int):
        super().__init__()
        self.g_prompt = nn.Parameter(torch.randn(g_len, hidden) * 0.02)
        self.e_prompts = nn.ParameterList(
            [nn.Parameter(torch.randn(e_len, hidden) * 0.02) for _ in range(n_experts)]
        )
        self.e_keys = nn.ParameterList(
            [nn.Parameter(torch.randn(hidden) * 0.02) for _ in range(n_experts)]
        )


class DualPrompt(PromptBasedMethod):
    """DualPrompt: shared G-prompt + per-task expert E-prompts."""

    name = "dualprompt"

    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        self.n_experts = config.get("n_experts", 10)
        g_len = config.get("g_prompt_length", 5)
        e_len = config.get("e_prompt_length", 5)
        self.lambda_key = config.get("lambda_key", 0.5)
        self._task_idx = -1
        self._register_prompt_module(
            "dp", _DualPromptModules(self.n_experts, g_len, e_len, hidden_dim)
        )

    def before_task(self, task: TaskInfo, train_loader) -> None:
        self._task_idx += 1

    def _cur_idx(self) -> int:
        return min(max(self._task_idx, 0), self.n_experts - 1)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Only the current expert (+ key), the shared G-prompt, and the head."""
        idx = self._cur_idx()
        params = [self.dp.g_prompt, self.dp.e_prompts[idx], self.dp.e_keys[idx]]
        params += [p for p in self.model.model.classifier.parameters() if p.requires_grad]
        return params

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = query.shape[0]
        g = self.dp.g_prompt.unsqueeze(0).expand(B, -1, -1)  # (B, Lg, D)
        q = F.normalize(query, dim=-1)

        if self.model.training:
            idx = self._cur_idx()
            e = self.dp.e_prompts[idx].unsqueeze(0).expand(B, -1, -1)  # (B, Le, D)
            k = F.normalize(self.dp.e_keys[idx], dim=-1)
            aux = self.lambda_key * (1.0 - (q @ k)).mean()
        else:
            seen = self._cur_idx() + 1
            keys = F.normalize(torch.stack([self.dp.e_keys[i] for i in range(seen)]), dim=-1)
            pick = (q @ keys.T).argmax(dim=-1)  # (B,) nearest seen expert
            experts = torch.stack([self.dp.e_prompts[i] for i in range(seen)])  # (seen, Le, D)
            e = experts[pick]  # (B, Le, D)
            aux = torch.zeros((), device=query.device)

        return torch.cat([g, e], dim=1), aux
