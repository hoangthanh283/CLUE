# doccl/methods/lexslot_mask.py
"""Lexical task-similarity and slot-gradient-mask numerics for LexSlot.

A task's signature is a sparse bag-of-OCR-tokens vector. Inter-task cosine similarity S
(measured: SROIE<->CORD 0.31, FUNSD 0.10) decides parameter sharing: when training task t,
a slot owned by a prior task tau is trainable in proportion to S[t, tau] (soft), above a
cutoff (hard), or not at all (off). Only claimed slots are trainable; unclaimed capacity
stays silent until a task owns it. The
returned mask scales the slot gradients, so low-S tasks cannot overwrite unrelated slots
while high-S tasks co-train shared slots.
"""

from __future__ import annotations

import torch


def signature_cosine(sig_a: torch.Tensor, sig_b: torch.Tensor) -> float:
    """Cosine of two signature vectors; 0.0 if either has zero norm."""
    na, nb = sig_a.norm(), sig_b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float((sig_a @ sig_b) / (na * nb))


def task_similarity_matrix(sigs: list[torch.Tensor]) -> torch.Tensor:
    """(T,) list of (V,) signatures -> (T,T) cosine matrix (diagonal forced to 1.0)."""
    t = len(sigs)
    mat = torch.zeros(t, t)
    for i in range(t):
        for j in range(t):
            mat[i, j] = 1.0 if i == j else signature_cosine(sigs[i], sigs[j])
    return mat


def slot_trainable_mask(
    task_id: int,
    slot_owner: list[int],
    S: torch.Tensor,  # noqa: N803 — S is the similarity matrix (standard name)
    sharing: str,
    threshold: float,
) -> torch.Tensor:
    """(n_slots,) float mask in [0,1] scaling each slot's gradient for the current task.

    own slot -> 1.0; unclaimed slot -> 0.0; prior-task slot -> soft: S[task_id,owner];
    hard: 1.0 if S>=threshold else 0.0 ; off: 0.0.
    """
    n = len(slot_owner)
    m = torch.zeros(n)
    for s in range(n):
        owner = slot_owner[s]
        if owner == task_id:
            m[s] = 1.0
            continue
        if owner == -1:
            continue
        if sharing == "off":
            m[s] = 0.0
        elif sharing == "hard":
            m[s] = 1.0 if float(S[task_id, owner]) >= threshold else 0.0
        else:  # soft
            m[s] = float(S[task_id, owner])
    return m
