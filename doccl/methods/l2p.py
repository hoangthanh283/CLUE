"""Learn to Prompt (L2P).

Reference: Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.

A frozen pre-trained backbone with a learnable prompt pool. For each input:
    1. Compute query q(x) = CLS embedding from frozen backbone
    2. Select top-K prompts from pool by cosine similarity with keys
    3. Prepend selected prompts to the input token sequence
    4. Forward through frozen backbone + trainable classifier

Loss = CE + λ * key_pull (pull selected keys toward query)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class PromptPool(nn.Module):
    """A pool of N learnable prompts, each of length L_p × hidden_dim.

    Attributes:
        prompts: (N, L_p, D) — the learnable prompt vectors
        keys:    (N, D)      — the learnable keys for top-K matching
    """

    def __init__(self, n_prompts: int = 10, prompt_length: int = 5, hidden_dim: int = 768):
        super().__init__()
        self.n_prompts = n_prompts
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim

        # Initialize prompts and keys with small random values
        self.prompts = nn.Parameter(torch.randn(n_prompts, prompt_length, hidden_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(n_prompts, hidden_dim) * 0.02)

    def select(self, query: torch.Tensor, top_k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Select top-K prompts per query by cosine similarity.

        Args:
            query: (B, D) query vectors
            top_k: number of prompts to select per example

        Returns:
            selected_prompts: (B, top_k, L_p, D)
            key_pull_loss: scalar — cosine distance between query and selected keys
        """
        # Normalize for cosine similarity
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        sim = q_norm @ k_norm.T  # (B, N)

        topk_sim, topk_idx = sim.topk(top_k, dim=-1)  # (B, top_k)

        # Gather selected prompts: (B, top_k, L_p, D)
        selected = self.prompts[topk_idx]

        # Key-pull loss: maximize cos sim → minimize 1 - cos sim
        key_pull = (1.0 - topk_sim).mean()

        return selected, key_pull


class L2P(NaiveFineTune):
    """Learn to Prompt with frozen LayoutLMv3 backbone + trainable prompt pool."""

    name = "l2p"

    def __init__(self, model, config):
        super().__init__(model, config)

        n_prompts = config.get("n_prompts", 10)
        prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 5)
        self.lambda_key = config.get("lambda_key", 0.5)
        hidden_dim = model.hidden_size

        # Freeze backbone, keep classifier trainable
        model.freeze_backbone()
        for p in model.model.classifier.parameters():
            p.requires_grad = True

        # Allocate prompt pool
        self.prompt_pool = PromptPool(n_prompts, prompt_length, hidden_dim).to(self.device)
        self.state.prompt_pool = self.prompt_pool

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Trainable params = prompt pool + classifier head."""
        return list(self.prompt_pool.parameters()) + [
            p for p in self.model.model.classifier.parameters() if p.requires_grad
        ]

    def _query(self, batch: dict) -> torch.Tensor:
        """Compute query vector q(x) from frozen backbone CLS token."""
        with torch.no_grad():
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = self.model.model.layoutlmv3(**inputs)
            # CLS token is at position 0
            cls = outputs.last_hidden_state[:, 0]  # (B, D)
        return cls

    def _forward_with_prompts(
        self, batch: dict, prompts: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with prompts prepended to text token embeddings.

        Implementation note: this is a simplified prepend-to-text-only injection.
        Full L2P-style "prefix tuning" injects at each transformer layer's K/V;
        we use the simpler input-prepend variant for AAAI Lite scope.

        Args:
            batch: standard input batch
            prompts: (B, top_k * L_p, D) flattened prompt sequence

        Returns:
            logits over original token positions only (B, L, num_classes)
        """
        # NOTE: This is a placeholder implementation. The full version requires
        # carefully constructing position_ids, bbox padding, attention_mask
        # extension. To be completed in Week 7 when L2P is implemented end-to-end.
        # For now, raise to avoid silent incorrectness.
        raise NotImplementedError(
            "L2P forward-with-prompts: implement in Week 7. "
            "Requires position_id and bbox handling for prompt slots."
        )

    def train_task(self, task: TaskInfo, train_loader: DataLoader) -> TrainMetrics:
        # Placeholder: full implementation deferred to Week 7
        # Once _forward_with_prompts works, training loop is:
        #   q = self._query(batch)
        #   prompts, key_pull = self.prompt_pool.select(q, self.top_k)
        #   B, K, Lp, D = prompts.shape
        #   prompts_flat = prompts.reshape(B, K * Lp, D)
        #   logits = self._forward_with_prompts(batch, prompts_flat)
        #   ce_loss = F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1), ignore_index=-100)
        #   loss = ce_loss + self.lambda_key * key_pull
        raise NotImplementedError("L2P training loop: Week 7 deliverable")
