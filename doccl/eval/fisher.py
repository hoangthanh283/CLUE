"""Fisher information matrix utilities for pilot study and EWC.

We compute the empirical (diagonal) Fisher information per parameter group:
    F_i = E_{(x,y) ~ D} [(∂ log p(y|x; θ) / ∂θ_i)^2]

Used for:
    - Pilot study: per-component importance, Fisher drop = forgetting signal
    - EWC: regularization weights for elastic weight consolidation
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def empirical_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    n_samples: int = 200,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Compute diagonal of empirical Fisher information matrix.

    Args:
        model: model with classifier head producing logits
        dataloader: provides labeled batches (input_ids, bbox, pixel_values, labels)
        n_samples: number of samples to use (more = more accurate, slower)
        device: forward pass device

    Returns:
        {param_name: Fisher diagonal tensor (same shape as param)}
    """
    model.eval()
    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad
    }

    seen = 0
    pbar = tqdm(dataloader, desc="Fisher", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch["labels"]

        _zero_grads(model)
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Compute log-likelihood of (input, predicted-label) under the model.
        # Using empirical Fisher: condition on observed labels rather than sampling.
        # Mask out -100 (ignored) tokens.
        valid_mask = labels != -100
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log-probs at observed labels
        safe_labels = labels.clone()
        safe_labels[~valid_mask] = 0  # avoid index error; will mask out below
        nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        nll = nll[valid_mask].sum()

        nll.backward()

        # Accumulate squared gradients
        batch_size = batch["input_ids"].shape[0]
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2) * batch_size

        seen += batch_size
        if seen >= n_samples:
            break

    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= max(seen, 1)

    _zero_grads(model)
    return fisher


def fisher_per_group(
    fisher: dict[str, torch.Tensor],
    param_groups: dict[str, list[nn.Parameter]],
    model: nn.Module,
) -> dict[str, float]:
    """Aggregate Fisher diagonal into named parameter groups.

    Args:
        fisher: output of empirical_fisher_diagonal (param_name → tensor)
        param_groups: name → list of parameters (from model.param_groups property)
        model: needed to map parameters back to their names

    Returns:
        {group_name: scalar mean Fisher information}
    """
    # Build param-id → name lookup
    param_id_to_name = {id(p): name for name, p in model.named_parameters()}

    out: dict[str, float] = {}
    for group_name, params in param_groups.items():
        total = 0.0
        count = 0
        for p in params:
            name = param_id_to_name.get(id(p))
            if name is None or name not in fisher:
                continue
            total += float(fisher[name].sum())
            count += fisher[name].numel()
        out[group_name] = total / max(count, 1)
    return out


def fisher_drop(
    fisher_before: dict[str, float],
    fisher_after: dict[str, float],
) -> dict[str, float]:
    """Compute relative drop in Fisher per group between two checkpoints.

    Negative drop = parameters became LESS important to current task
    (forgetting signal).
    """
    out = {}
    for k in fisher_before:
        if k in fisher_after:
            base = fisher_before[k]
            if base > 1e-12:
                out[k] = (fisher_after[k] - base) / base
            else:
                out[k] = 0.0
    return out
