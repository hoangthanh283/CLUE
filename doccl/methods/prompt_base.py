"""Shared machinery for prompt-based continual-learning methods.

All prompt methods on LayoutLMv3 (L2P, DualPrompt, CODA-Prompt, and the
modality-routed-prompt DocCL candidate) share the same skeleton:

    * freeze the backbone, train only the prompt modules + classifier head
    * extract a query q(x) = CLS embedding from the frozen backbone
    * select / assemble prompt vectors from that query
    * inject them via ``LayoutLMv3Wrapper.forward_with_prompts``
    * minimise CE over the (prompt-truncated) token logits + an auxiliary loss

Subclasses implement two hooks:
    ``_build_prompt_modules(config, hidden_dim)`` — register trainable modules
    ``_select_prompts(query, batch) -> (prompt_embeds (B, P, D), aux_loss)``

This keeps the (token-classification + prompt-slot truncation) loop in one place;
the prompt injection itself lives in ``LayoutLMv3Wrapper.forward_with_prompts``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.metrics import compute_token_f1
from doccl.methods.naive import NaiveFineTune
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics


class PromptPool(nn.Module):
    """A pool of N learnable prompts, each of length L_p x hidden_dim.

    Attributes:
        prompts: (N, L_p, D) — the learnable prompt vectors
        keys:    (N, D)      — the learnable keys for top-K matching
    """

    def __init__(self, n_prompts: int = 10, prompt_length: int = 5, hidden_dim: int = 768):
        super().__init__()
        self.n_prompts = n_prompts
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
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
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys, dim=-1)
        sim = q_norm @ k_norm.T  # (B, N)
        topk_sim, topk_idx = sim.topk(min(top_k, self.n_prompts), dim=-1)  # (B, top_k)
        selected = self.prompts[topk_idx]  # (B, top_k, L_p, D)
        key_pull = (1.0 - topk_sim).mean()
        return selected, key_pull


class PromptBasedMethod(NaiveFineTune):
    """Base for prompt-based CL methods. Freezes backbone; trains prompts + head."""

    name = "prompt_base"

    def __init__(self, model, config):
        super().__init__(model, config)
        model.freeze_backbone()
        for p in self.model.model.classifier.parameters():
            p.requires_grad = True
        self.hidden_dim = model.hidden_size
        self._prompt_modules: list[nn.Module] = []
        self._build_prompt_modules(config, self.hidden_dim)

    # ─── hooks for subclasses ───────────────────────────────────────────────
    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        raise NotImplementedError

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    # ─── shared helpers ─────────────────────────────────────────────────────
    def _register_prompt_module(self, name: str, module: nn.Module) -> None:
        """Attach a trainable prompt module, moved to device and tracked."""
        module = module.to(self.device)
        setattr(self, name, module)
        self._prompt_modules.append(module)
        if self.state.prompt_pool is None:
            self.state.prompt_pool = module

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for m in self._prompt_modules:
            params += list(m.parameters())
        params += [p for p in self.model.model.classifier.parameters() if p.requires_grad]
        return params

    @torch.no_grad()
    def _query(self, batch: dict) -> torch.Tensor:
        """q(x) = CLS embedding from the frozen backbone (backbone-agnostic)."""
        return self.model.encode_query(batch)  # (B, D)

    # ─── training / evaluation ──────────────────────────────────────────────
    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        # NOTE: prompt-based methods use a custom forward_with_prompts path, so the
        # base-class val-F1 helper (which calls self.model(**batch)) does not apply;
        # they retain the fixed epoch budget. val_loader is accepted for signature
        # compatibility only.
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"{self.name} T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                query = self._query(batch)
                prompt_embeds, aux = self._select_prompts(query, batch)
                logits = self.model.forward_with_prompts(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    pixel_values=batch.get("pixel_values"),
                    prompt_embeds=prompt_embeds,
                    attention_mask=batch.get("attention_mask"),
                )
                labels = batch["labels"][:, : logits.shape[1]]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = ce + aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"ce": f"{ce.item():.3f}", "aux": f"{float(aux):.4f}"})

        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        self.model.eval()
        results: dict[int, EvalMetrics] = {}
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        with torch.no_grad():
            for tid, loader in eval_loaders.items():
                all_preds, all_labels = [], []
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    query = self._query(batch)
                    prompt_embeds, _ = self._select_prompts(query, batch)
                    logits = self.model.forward_with_prompts(
                        input_ids=batch["input_ids"],
                        bbox=batch["bbox"],
                        pixel_values=batch.get("pixel_values"),
                        prompt_embeds=prompt_embeds,
                        attention_mask=batch.get("attention_mask"),
                    )
                    preds = logits.argmax(dim=-1)
                    labels = batch["labels"][:, : logits.shape[1]]
                    mask = labels != -100
                    all_preds.extend(preds[mask].cpu().tolist())
                    all_labels.extend(labels[mask].cpu().tolist())
                metrics = compute_token_f1(all_preds, all_labels, id_to_label)
                results[tid] = EvalMetrics(
                    task_id=tid,
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    n_samples=len(all_labels),
                )
        return results
