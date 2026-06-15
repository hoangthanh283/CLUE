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
    modality_mask=None,
) -> dict[str, torch.Tensor]:
    """Compute diagonal of empirical Fisher information matrix.

    Args:
        model: model with classifier head producing logits
        dataloader: provides labeled batches (input_ids, bbox, pixel_values, labels)
        n_samples: number of samples to use (more = more accurate, slower)
        device: forward pass device
        modality_mask: optional ``ModalityMask`` for the pilot conditions. When
            given, it is forwarded to the model so the Fisher estimate is taken
            under the **same** masked inputs the condition trained on (fixes the
            mask-agnostic per-condition Fisher flagged in review m3). Models
            whose ``forward`` does not accept ``modality_mask`` (e.g. the BERT
            baseline) should pass ``None``.

    Returns:
        {param_name: Fisher diagonal tensor (same shape as param)}
    """
    model.eval()
    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad
    }
    extra_kwargs = {} if modality_mask is None else {"modality_mask": modality_mask}

    seen = 0
    pbar = tqdm(dataloader, desc="Fisher", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch["labels"]

        _zero_grads(model)
        outputs = model(
            **{k: v for k, v in batch.items() if k != "labels"}, **extra_kwargs
        )
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


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of all trainable parameters, keyed by name.

    Used to record ``θ^{t-1}`` before training a task so the post-task
    displacement can be measured (see ``fisher_weighted_displacement``).
    """
    return {
        name: p.detach().clone().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def fisher_weighted_displacement(
    fisher_old: dict[str, torch.Tensor],
    params_before: dict[str, torch.Tensor],
    params_after: dict[str, torch.Tensor],
    param_groups: dict[str, list[nn.Parameter]],
    model: nn.Module,
    reduction: str = "mean",
) -> dict[str, float]:
    r"""Per-group old-task-Fisher-weighted parameter displacement.

    .. math::
        D_g = \sum_{\phi \in g} F^{(t-1)}_\phi \,
              \big(\theta^{(t)}_\phi - \theta^{(t-1)}_\phi\big)^2

    This is a **forgetting-specific** localizer (review C4): it is large where
    parameters that were important to the *previous* task (high ``F^{(t-1)}``)
    have *moved* most. Unlike raw Fisher *level*, which is largest at the head by
    construction of the loss geometry for any classifier, ``D_g`` says where the
    old task's knowledge was actually overwritten.

    Args:
        fisher_old: per-parameter Fisher diagonal computed on the OLD task
            (before training the new task), keyed by parameter name.
        params_before / params_after: ``θ^{t-1}`` / ``θ^{t}`` snapshots
            (``snapshot_params`` output), keyed by parameter name.
        param_groups: name → list of parameters (``model.param_groups`` or
            ``model.param_groups_by_depth``).
        model: used to map parameter objects back to their names.
        reduction: ``"mean"`` (per-parameter mean, comparable across
            differently-sized groups — default, used for the "where forgetting
            lives" plots) or ``"sum"`` (the raw quadratic above).

    Returns:
        {group_name: scalar displacement}
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    param_id_to_name = {id(p): name for name, p in model.named_parameters()}
    out: dict[str, float] = {}
    for group_name, params in param_groups.items():
        total = 0.0
        count = 0
        for p in params:
            name = param_id_to_name.get(id(p))
            if (
                name is None
                or name not in fisher_old
                or name not in params_before
                or name not in params_after
            ):
                continue
            f = fisher_old[name].cpu()  # same shape as θ^{t-1}
            before = params_before[name].cpu()
            after = params_after[name].cpu()
            # The classifier head grows across CIL boundaries: θ^t can be wider
            # than θ^{t-1}. Forgetting of the OLD task lives in the *old* rows, so
            # crop θ^t to the overlap before differencing.
            if after.shape != before.shape:
                crop = tuple(slice(0, min(a, b)) for a, b in zip(after.shape, before.shape))
                after = after[crop]
            disp = (after - before) ** 2
            total += float((f * disp).sum())
            count += f.numel()
        out[group_name] = total / max(count, 1) if reduction == "mean" else total
    return out
