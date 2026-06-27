from __future__ import annotations

import torch

from doccl.methods.lexslot_memory import LogitSlots, ReprSlots


def test_logit_slots_delta_shape_and_zero_init():
    ls = LogitSlots(n_slots=4, hidden_dim=8, n_labels=5)
    feats = torch.randn(2, 6, 8)
    d = ls.logits_delta(feats)
    assert d.shape == (2, 6, 5)


def test_logit_slots_expand_labels_preserves():
    ls = LogitSlots(n_slots=3, hidden_dim=8, n_labels=5)
    with torch.no_grad():
        ls.proj.copy_(torch.arange(3 * 5, dtype=torch.float).reshape(3, 5))
    ls.expand_labels(7)
    assert ls.proj.shape == (3, 7)
    assert torch.allclose(ls.proj[:, :5], torch.arange(3 * 5, dtype=torch.float).reshape(3, 5))


def test_logit_slots_grad_mask_zeros_isolated_slot():
    ls = LogitSlots(n_slots=3, hidden_dim=4, n_labels=2)
    ls.set_grad_mask(torch.tensor([1.0, 0.0, 1.0]))  # slot 1 frozen
    feats = torch.randn(1, 3, 4)
    loss = ls.logits_delta(feats).pow(2).sum()
    loss.backward()
    assert torch.allclose(ls.values.grad[1], torch.zeros(4), atol=1e-6)  # masked
    assert ls.values.grad[0].abs().sum() > 0  # unmasked trains


def test_repr_slots_delta_shape():
    rs = ReprSlots(n_slots=4, hidden_dim=8, rank=2)
    h = torch.randn(2, 6, 8)
    d = rs.repr_delta(h)
    assert d.shape == (2, 6, 8)


def test_repr_slots_grad_mask_zeros_isolated_slot():
    rs = ReprSlots(n_slots=3, hidden_dim=4, rank=2)
    rs.set_grad_mask(torch.tensor([0.0, 1.0, 1.0]))  # slot 0 frozen
    h = torch.randn(1, 3, 4)
    loss = rs.repr_delta(h).pow(2).sum()
    loss.backward()
    assert torch.allclose(rs.down.grad[0], torch.zeros(4, 2), atol=1e-6)
    assert rs.down.grad[1].abs().sum() > 0


def test_set_owner():
    ls = LogitSlots(n_slots=4, hidden_dim=4, n_labels=2)
    ls.set_owner([0, 1], task_id=3)
    assert ls.slot_owner[0] == 3 and ls.slot_owner[1] == 3 and ls.slot_owner[2] == -1
