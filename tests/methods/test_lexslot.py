"""Unit tests for LexSlot internals on synthetic tensors / stubs (fast, GPU-free). Full
lifecycle on a real LayoutLMv3 is in tests/methods/test_all_methods_e2e.py once registered."""

from __future__ import annotations

import torch

from doccl.methods.lexslot import LexSlot


def test_late_layer_indices_for_12():
    # late bucket = idx >= 2*(12//3) = 8 -> [8,9,10,11]
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_late") == [8, 9, 10, 11]


def test_late_layer_indices_head_only_empty():
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_only") == []


def test_late_layer_indices_uniform_is_all():
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="uniform") == list(range(12))


def test_late_layer_indices_head_late_mid():
    # mid+late = idx >= 12//3 = 4 -> [4..11]
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_late_mid") == list(
        range(4, 12)
    )


def test_derive_mask_soft_grades_by_similarity():
    m = LexSlot.__new__(LexSlot)
    m.slot_sharing = "soft"
    m.share_threshold = 0.5
    # 4 head slots: 0,1 owned by task0, 2,3 fresh; current task1; S[1,0]=0.3
    import torch as T  # noqa: N812

    S = T.tensor([[1.0, 0.3], [0.3, 1.0]])  # noqa: N806
    owner = [0, 0, -1, -1]
    mask = m._derive_mask(task_id=1, slot_owner=owner, S=S)
    assert torch.allclose(mask, torch.tensor([0.3, 0.3, 1.0, 1.0]), atol=1e-6)
