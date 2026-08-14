# tests/methods/test_lexslot_mask.py
from __future__ import annotations

import torch

from doccl.methods.lexslot_mask import (
    signature_cosine,
    slot_trainable_mask,
    task_similarity_matrix,
)


def test_signature_cosine_basic():
    a = torch.tensor([1.0, 0.0, 1.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    # cos = 1/sqrt(2)
    assert abs(signature_cosine(a, b) - 0.70710678) < 1e-5
    assert signature_cosine(a, torch.zeros(3)) == 0.0  # all-zero guard


def test_task_similarity_matrix_diag_one_and_symmetric():
    sigs = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0])]
    S = task_similarity_matrix(sigs)  # noqa: N806
    assert S.shape == (3, 3)
    assert torch.allclose(torch.diag(S), torch.ones(3), atol=1e-6)
    assert torch.allclose(S, S.T, atol=1e-6)
    assert S[0, 2].item() == 0.0  # orthogonal


def test_slot_mask_own_slots_full_others_graded_soft():
    # 4 slots: slots 0,1 owned by task 0; slots 2,3 owned by task 1.
    slot_owner = [0, 0, 1, 1]
    # S[task2=2, 0]=0.1, S[2,1]=0.3  (current task is 2)
    S = torch.tensor([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])  # noqa: N806
    m = slot_trainable_mask(2, slot_owner, S, sharing="soft", threshold=0.5)
    # task2 owns none here; task0's slots graded by S[2,0]=0.1, task1's by S[2,1]=0.3
    assert torch.allclose(m, torch.tensor([0.1, 0.1, 0.3, 0.3]), atol=1e-6)


def test_slot_mask_own_slots_are_one():
    slot_owner = [0, 0, 1, 1]
    S = torch.tensor([[1.0, 0.2], [0.2, 1.0]])  # noqa: N806
    m = slot_trainable_mask(1, slot_owner, S, sharing="soft", threshold=0.5)
    # task1 owns slots 2,3 -> 1.0 ; task0's slots 0,1 graded by S[1,0]=0.2
    assert torch.allclose(m, torch.tensor([0.2, 0.2, 1.0, 1.0]), atol=1e-6)


def test_slot_mask_hard_threshold():
    slot_owner = [0, 0, 1, 1]
    S = torch.tensor([[1.0, 0.2, 0.1], [0.2, 1.0, 0.6], [0.1, 0.6, 1.0]])  # noqa: N806
    m = slot_trainable_mask(2, slot_owner, S, sharing="hard", threshold=0.5)
    # S[2,0]=0.1 <0.5 ->0 ; S[2,1]=0.6 >=0.5 ->1
    assert torch.allclose(m, torch.tensor([0.0, 0.0, 1.0, 1.0]), atol=1e-6)


def test_slot_mask_off_isolates():
    slot_owner = [0, 0, 1, 1]
    S = torch.tensor([[1.0, 0.9], [0.9, 1.0]])  # noqa: N806
    m = slot_trainable_mask(1, slot_owner, S, sharing="off", threshold=0.5)
    # off: only own slots trainable, no sharing regardless of high S
    assert torch.allclose(m, torch.tensor([0.0, 0.0, 1.0, 1.0]), atol=1e-6)


def test_slot_mask_unclaimed_slots_stay_silent():
    slot_owner = [0, -1, -1]  # slots 1,2 unclaimed
    S = torch.tensor([[1.0, 0.3], [0.3, 1.0]])  # noqa: N806
    m = slot_trainable_mask(1, slot_owner, S, sharing="soft", threshold=0.5)
    # slot0 is softly shared; unclaimed capacity stays silent until claimed.
    assert torch.allclose(m, torch.tensor([0.3, 0.0, 0.0]), atol=1e-6)
