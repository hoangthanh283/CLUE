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
        # Inference lexical gate (B, n_slots), set by LexSlot's forward pre-hook before
        # the slot's delta is read. None -> the delta is byte-identical to the ungated
        # path (opt-in / safe default for any caller that never sets it).
        self._infer_gate: torch.Tensor | None = None

    def set_grad_mask(self, mask: torch.Tensor) -> None:
        self.grad_mask.copy_(mask.to(self.grad_mask.device))

    def set_owner(self, slot_idx: list[int], task_id: int) -> None:
        for s in slot_idx:
            self.slot_owner[s] = task_id

    def _scale_rows(self, grad: torch.Tensor) -> torch.Tensor:
        # grad shape (n_slots, ...) -> scale dim 0 by grad_mask. Match grad's device+dtype
        # so the hook is safe even if the buffer and grad ever land on different devices.
        shape = [self.n_slots] + [1] * (grad.dim() - 1)
        return grad * self.grad_mask.to(device=grad.device, dtype=grad.dtype).reshape(shape)


class LogitSlots(_MaskedSlots):
    def __init__(self, n_slots: int, hidden_dim: int, n_labels: int):
        super().__init__(n_slots)
        self.hidden_dim = hidden_dim
        self.values = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)
        # proj zero-init -> logits_delta == 0 at init: an unclaimed / never-trained slot
        # is an EXACT no-op, so isolated slots (grad-masked to 0) inject no logit noise into
        # other tasks' predictions (the "isolated -> no forgetting" invariant). values can be
        # nonzero (the key side) since the bilinear act @ proj is still gated by proj==0.
        self.proj = nn.Parameter(torch.zeros(n_slots, n_labels))
        self.values.register_hook(self._scale_rows)
        self.proj.register_hook(self._scale_rows)

    def logits_delta(self, feats: torch.Tensor) -> torch.Tensor:
        # feats (B,L,d); activation_s = <values_s, feats> -> (B,L,n_slots); @ proj -> (B,L,C)
        act = feats @ self.values.T  # (B,L,n_slots)
        # Inference lexical gate: scale each slot's activation by its per-document gate
        # (B,n_slots) -> broadcast over the token dim (B,1,n_slots). None => no-op.
        if self._infer_gate is not None:
            act = act * self._infer_gate.unsqueeze(1).to(act.dtype)
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
        # up zero-init -> repr_delta == 0 at init (LoRA-style): an unclaimed / never-trained
        # slot is an EXACT no-op shift, so isolated slots perturb no other task's hidden state
        # (the "isolated -> no forgetting" invariant). down can be nonzero; the slot is gated
        # by up==0 until its owner trains it (grad w.r.t. down is 0 only on the first step).
        self.up = nn.Parameter(torch.zeros(n_slots, rank, hidden_dim))
        self.down.register_hook(self._scale_rows)
        self.up.register_hook(self._scale_rows)

    def repr_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden (B,L,d); per slot: (hidden @ down_s) @ up_s, summed over slots.
        # einsum: bld,sdr->blsr then blsr,srh->blh
        mid = torch.einsum("bld,sdr->blsr", hidden, self.down)
        # Inference lexical gate (B,n_slots) -> broadcast over (token, rank) dims:
        # (B,1,n_slots,1). None => no-op (byte-identical to the ungated path).
        if self._infer_gate is not None:
            mid = mid * self._infer_gate.unsqueeze(1).unsqueeze(-1).to(mid.dtype)
        return torch.einsum("blsr,srh->blh", mid, self.up)
