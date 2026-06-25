"""Head-only model merging for continual learning (DocMERGE).

The thesis localizes catastrophic forgetting to the classifier **head** (Fisher-weighted
displacement ~1e-4 in the head vs ~1e-8 in late backbone layers). With the backbone
frozen, every task's head is fine-tuned from the *same* base in the *same* loss basin
(Linear Mode Connectivity), so the per-task head deltas can be **merged** into a single
head that is jointly good for all tasks — the model-soup effect — which is the mechanism
that produces *positive* backward transfer (old tasks improve as new ones train).

This module implements only the merge arithmetic on flat weight/bias tensors. Three
rules, exposed as the ``merge_rule`` ablation axis of the method:

    * ``plain``  — uniform average of the deltas (Model Soups; the LMC-faithful floor).
    * ``ties``   — magnitude-trim + sign-election before averaging (TIES-Merging,
                   Yadav et al. NeurIPS 2023): keep the top fraction of each delta by
                   magnitude, elect the dominant sign per coordinate, average only the
                   agreeing, kept entries.
    * ``fisher`` — weight each task's delta by its head Fisher importance (the SAME
                   Fisher the method computes for the head-locus diagnostic). The signal
                   that *localizes* forgetting also *drives* the merge.

Merging operates on head **deltas** ``Δ_t = θ_t − θ_base`` rather than raw weights so
that, in class-incremental settings where the head grows into new logit rows, a task
that never saw a given row contributes a *zero* delta there (it leaves that row's base
init untouched) instead of dragging it toward zero. Deltas of differing width are padded
to the widest before merging; the merged head is ``θ_base + (merged delta)``.
"""

from __future__ import annotations

import torch

__all__ = ["merge_head_deltas", "MERGE_RULES"]

MERGE_RULES = ("plain", "ties", "fisher")


def _pad_to(t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Zero-pad ``t`` up to ``shape`` (rows/cols a CIL-grown head added after task t)."""
    if tuple(t.shape) == tuple(shape):
        return t
    out = torch.zeros(shape, dtype=t.dtype, device=t.device)
    crop = tuple(slice(0, s) for s in t.shape)
    out[crop] = t
    return out


def _stack_deltas(
    deltas: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Size]:
    """Pad a list of differently-shaped deltas to the widest, stack → (T, *shape)."""
    target = deltas[0].shape
    for d in deltas:
        target = torch.Size(
            max(a, b) for a, b in zip(_match_rank(target, d.shape), d.shape, strict=True)
        )
    stacked = torch.stack([_pad_to(d, target) for d in deltas], dim=0)
    return stacked, target


def _match_rank(a: torch.Size, b: torch.Size) -> torch.Size:
    """Guard: deltas of a head tensor always share rank (weight 2-D, bias 1-D)."""
    if len(a) != len(b):
        raise ValueError(f"cannot merge head deltas of differing rank: {tuple(a)} vs {tuple(b)}")
    return a


def _ties_elect(stacked: torch.Tensor, density: float) -> torch.Tensor:
    """TIES trim + sign-elect + disjoint-mean over a (T, *shape) delta stack.

    For each task: keep the top ``density`` fraction of |delta| (per task), zero the
    rest. Elect the per-coordinate sign by summed signed magnitude. Average only the
    kept entries whose sign agrees with the elected sign (disjoint mean). Coordinates
    with no agreeing kept entry merge to zero (fall back to base init).
    """
    T = stacked.shape[0]  # noqa: N806 — T = #tasks, standard merging notation
    flat = stacked.reshape(T, -1)  # (T, P)
    mag = flat.abs()
    if 0.0 < density < 1.0:
        k = max(1, int(round(density * flat.shape[1])))
        # Per-task magnitude threshold: keep the top-k coordinates of each delta.
        kth = mag.topk(k, dim=1).values[:, -1].unsqueeze(1)  # (T, 1)
        keep = mag >= kth
    else:
        keep = torch.ones_like(flat, dtype=torch.bool)
    trimmed = flat * keep
    # Elected sign per coordinate from the summed signed (kept) magnitude.
    elected = torch.sign(trimmed.sum(dim=0))  # (P,)
    agree = (torch.sign(trimmed) == elected.unsqueeze(0)) & keep  # (T, P)
    num = (trimmed * agree).sum(dim=0)  # (P,)
    den = agree.sum(dim=0).clamp(min=1)  # (P,)
    merged = num / den
    merged[elected == 0] = 0.0
    return merged.reshape(stacked.shape[1:])


def merge_head_deltas(
    deltas: list[torch.Tensor],
    rule: str = "plain",
    weights: list[float] | None = None,
    density: float = 0.2,
    count_aware: bool = True,
) -> torch.Tensor:
    """Merge per-task head deltas into one delta.

    Args:
        deltas: per-task ``Δ_t = θ_t − θ_base`` tensors (weight OR bias), possibly of
            differing width across CIL head growth; padded to the widest internally.
        rule: ``plain`` | ``ties`` | ``fisher``.
        weights: per-task non-negative weights for ``fisher`` (e.g. head Fisher mass per
            task); normalised internally. Ignored by ``plain``/``ties``. Length must
            match ``deltas``.
        density: kept fraction per task for ``ties`` (top-|delta| coordinates).
        count_aware: if True (default), a coordinate is averaged over only the tasks that
            actually wrote a NON-ZERO delta there, not over all T. This removes the 1/T
            shrinkage that a plain coordinate-mean inflicts on a task's OWN rows (where the
            other tasks contributed exactly zero — e.g. CIL disjoint logit rows, or any
            row a task never touched): such a row keeps its FULL learned magnitude instead
            of being divided by T. On rows written by every task (the DIL shared-label
            case) the divisor is still T, so genuinely-conflicting updates are still
            averaged. Set False to recover the classic Model-Soups coordinate-mean (the
            pre-RCA behaviour; kept for ablation). ``ties`` already uses an agreement-count
            denominator so it is unaffected by this flag.

    Returns:
        The merged delta tensor, shaped like the widest input.
    """
    if rule not in MERGE_RULES:
        raise ValueError(f"unknown merge rule {rule!r} (one of {MERGE_RULES})")
    if not deltas:
        raise ValueError("merge_head_deltas: empty delta list")
    stacked, _ = _stack_deltas(deltas)  # (T, *shape)

    if rule == "ties":
        return _ties_elect(stacked, density)

    # per-coordinate count of tasks that wrote a non-zero delta there (>=1 where any did).
    contributed = (stacked != 0).to(stacked.dtype) if count_aware else torch.ones_like(stacked)

    if rule == "plain":
        denom = contributed.sum(dim=0).clamp(min=1.0)  # avoid /0 on all-zero coords
        return stacked.sum(dim=0) / denom

    # rule == "fisher": weight each task by its (non-negative) Fisher mass, but only count
    # a task toward a coordinate's denominator where it actually contributed.
    T = stacked.shape[0]  # noqa: N806 — T = #tasks, standard merging notation
    if weights is None:
        w = torch.ones(T, dtype=stacked.dtype, device=stacked.device)
    else:
        if len(weights) != T:
            raise ValueError(f"fisher weights len {len(weights)} != n deltas {T}")
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).clamp(min=0.0)
    if float(w.sum()) <= 0.0:  # degenerate (all-zero importance) → uniform weights
        w = torch.ones(T, dtype=stacked.dtype, device=stacked.device)
    wshape = [T] + [1] * (stacked.dim() - 1)
    w_b = w.reshape(wshape)
    num = (stacked * w_b).sum(dim=0)
    # denominator = sum of weights of CONTRIBUTING tasks per coordinate (count-aware), or
    # total weight (classic) — clamp avoids /0 where no task contributed / all weights 0.
    denom = (contributed * w_b).sum(dim=0).clamp(min=1e-12)
    return num / denom
