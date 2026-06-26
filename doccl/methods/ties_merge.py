"""TIES task-vector merging — a faithful port of LCA's `helper.merge` (ICLR 2026).

Reference: github.com/tungts1101/LCA (`helper.py`). LCA merges per-task parameter sets by
forming task vectors ``θ_t − θ_base``, TIES-trimming each (keep the top-``topk``% by
magnitude), electing a per-coordinate sign from the summed signs, and averaging only the
sign-agreeing entries (disjoint mean with a ``clamp(min=1)`` denominator), then adding
``lamb ×`` the merged vector back to the base. This is the merge used by LCA; we reuse it
verbatim for the ``lca`` method so its consolidation matches the paper exactly.

Kept separate from ``head_merge.py`` (DocMERGE's bespoke head-only merge) because this is
the literal LCA algorithm operating on a *named parameter dict* (whole module: backbone +
head), not a single head tensor.
"""

from __future__ import annotations

import torch

__all__ = ["merge_state_dicts", "trim", "merge_task_vectors"]


def trim(tensor: torch.Tensor, topk: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TIES trim: keep the top-``topk``% of |tensor| entries, zero the rest.

    Returns (trimmed, sign(trimmed), |trimmed|) — the (τ, γ, μ) triple LCA uses. With
    ``topk=100`` nothing is trimmed (LCA's default), so TIES reduces to sign-election +
    disjoint mean.
    """
    flat = tensor.view(-1)
    mag = flat.abs()
    num_keep = max(1, int(len(flat) * topk / 100))
    threshold = torch.topk(mag, num_keep, largest=True, sorted=True).values[-1]
    mask = mag >= threshold
    trimmed = torch.where(mask, flat, torch.zeros((), dtype=tensor.dtype, device=tensor.device))
    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)
    return trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor)


def merge_task_vectors(
    trimmed_task_vectors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Disjoint mean of sign-agreeing trimmed task vectors (LCA `merge_task_vectors`)."""
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = gamma_tvs == gamma
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    merged = torch.where(
        mask, tau_tvs, torch.zeros((), dtype=tau_tvs.dtype, device=tau_tvs.device)
    ).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
    return merged


def merge_state_dicts(
    base_params: dict[str, torch.Tensor],
    tasks_params: list[dict[str, torch.Tensor]],
    method: str = "ties",
    lamb: float = 1.0,
    topk: int = 100,
) -> dict[str, torch.Tensor]:
    """Merge per-task param dicts onto a base (LCA `merge`).

    Args:
        base_params: name → base (pre-task) parameter tensor.
        tasks_params: list of name → post-task parameter tensors (one dict per task).
        method: ``ties`` | ``max`` | ``min`` | ``max_abs`` (LCA's options).
        lamb: scale on the merged task vector. topk: TIES keep-% (100 = no trim).

    Returns:
        name → merged parameter tensor (``base + lamb × merged_tv``).
    """
    if not tasks_params:
        return {k: v.clone() for k, v in base_params.items()}
    out: dict[str, torch.Tensor] = {}
    for name in tasks_params[0]:
        base_tv = base_params[name].clone()
        tvs = [tp[name] - base_tv for tp in tasks_params]
        if method == "ties":
            trimmed = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(trimmed)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "min":
            merged_tv = torch.min(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "max_abs":
            stacked = torch.stack(tvs, dim=0)
            idx = torch.argmax(stacked.abs(), dim=0)
            merged_tv = torch.gather(stacked, 0, idx.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f"unknown merge method {method!r}")
        out[name] = base_tv + lamb * merged_tv
    return out
