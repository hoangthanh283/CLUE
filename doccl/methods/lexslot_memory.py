"""LexSlot slot-memory modules: logit-slots (head) and representation-slots (late layers).

Slots additively modulate the model: a LogitSlots block adds a per-class logit bias from
the head's token features; a ReprSlots block adds a low-rank shift to a late encoder
layer's hidden output. Each slot carries an owner (which task first claimed it) and a
grad_mask buffer; a gradient hook scales each slot's gradient by its mask entry, so the
lexical mask (set per task) gates which slots a task may update.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MaskedSlots(nn.Module):
    """Shared owner/mask bookkeeping for slot modules."""

    def __init__(self, n_slots: int):
        super().__init__()
        self.n_slots = n_slots
        self.register_buffer("grad_mask", torch.ones(n_slots))
        self.slot_owner: list[int] = [-1] * n_slots

    def set_grad_mask(self, mask: torch.Tensor) -> None:
        self.grad_mask.copy_(mask.to(self.grad_mask.device))

    def set_owner(self, slot_idx: list[int], task_id: int) -> None:
        for s in slot_idx:
            self.slot_owner[s] = task_id

    def _scale_rows(self, grad: torch.Tensor) -> torch.Tensor:
        # grad shape (n_slots, ...) -> scale dim 0 by grad_mask.
        shape = [self.n_slots] + [1] * (grad.dim() - 1)
        return grad * self.grad_mask.to(grad.dtype).reshape(shape)


class LogitSlots(_MaskedSlots):
    def __init__(self, n_slots: int, hidden_dim: int, n_labels: int):
        super().__init__(n_slots)
        self.hidden_dim = hidden_dim
        self.values = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)
        self.proj = nn.Parameter(torch.randn(n_slots, n_labels) * 0.02)  # small init
        self.values.register_hook(self._scale_rows)
        self.proj.register_hook(self._scale_rows)

    def logits_delta(self, feats: torch.Tensor) -> torch.Tensor:
        # feats (B,L,d); activation_s = <values_s, feats> -> (B,L,n_slots); @ proj -> (B,L,C)
        act = feats @ self.values.T  # (B,L,n_slots)
        return act @ self.proj  # (B,L,n_labels)

    @torch.no_grad()
    def expand_labels(self, new_n_labels: int) -> None:
        old = self.proj
        if new_n_labels <= old.shape[1]:
            return
        new = torch.zeros(self.n_slots, new_n_labels, device=old.device, dtype=old.dtype)
        new[:, : old.shape[1]] = old
        self.proj = nn.Parameter(new)
        self.proj.register_hook(self._scale_rows)


class ReprSlots(_MaskedSlots):
    def __init__(self, n_slots: int, hidden_dim: int, rank: int = 16):
        super().__init__(n_slots)
        self.down = nn.Parameter(torch.randn(n_slots, hidden_dim, rank) * 0.02)
        self.up = nn.Parameter(torch.randn(n_slots, rank, hidden_dim) * 0.02)  # small init
        self.down.register_hook(self._scale_rows)
        self.up.register_hook(self._scale_rows)

    def repr_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden (B,L,d); per slot: (hidden @ down_s) @ up_s, summed over slots.
        # einsum: bld,sdr->blsr then blsr,srh->blh
        mid = torch.einsum("bld,sdr->blsr", hidden, self.down)
        return torch.einsum("blsr,srh->blh", mid, self.up)
