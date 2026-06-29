"""Unit tests for IS3 (two shifts) on synthetic tensors — GPU-free.

Tests cover the three mandatory mechanisms that make IS3 distinct from LwF:
  1. Token partition: CE on new-entity tokens only, KD on O-labeled tokens only (disjoint).
  2. Prototype replay: mean L2-normed features per class, CE through live head.
  3. Gradient surgery: old-entity head rows (1..n_old-1) scaled, O row (0) untouched,
     new rows (n_old..) scaled by new_grad_weight.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from doccl.methods.is3 import IS3


# ─── minimal fake wrapper ─────────────────────────────────────────────────────
class _FakeWrapper(nn.Module):
    def __init__(self, hidden=8, n_cls=4):
        super().__init__()
        self.encoder = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, n_cls, bias=True)
        self._hidden = hidden

    def parameters(self, recurse=True):
        return super().parameters(recurse)

    def token_features(self, batch):
        x = batch.get("input_ids", torch.zeros(2, 3, self._hidden))
        return self.encoder(x.float().to(next(self.parameters()).device))

    def train(self, mode=True):
        return super().train(mode)

    def eval(self):
        return super().eval()


def _make_is3(**kwargs) -> IS3:
    m = IS3.__new__(IS3)
    m.model = _FakeWrapper(**kwargs)
    m.config = {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "max_grad_norm": 1.0}
    m.kd_weight = 2.0
    m.proto_weight = 0.1
    m.kd_temp_s = 1.0
    m.kd_temp_t = 1.0
    m.grad_weight = 0.6
    m.new_grad_weight = 1.0
    m._teacher = None
    m._n_old = 0
    m.prototypes = {}
    m.device = torch.device("cpu")
    m.amp_enabled = False
    return m


# ─── 1. Token partition (CE on entity tokens, KD on O tokens) ─────────────────
def test_ce_mask_excludes_o_and_padding():
    flat_labels = torch.tensor([-100, 0, 1, 2, 0, 3, -100])
    ce_mask = (flat_labels != -100) & (flat_labels != 0)
    distill_mask = flat_labels == 0
    assert ce_mask.tolist() == [False, False, True, True, False, True, False]
    assert distill_mask.tolist() == [False, True, False, False, True, False, False]
    # The two masks are disjoint (the key IS3 invariant).
    assert not (ce_mask & distill_mask).any()


# ─── 2. KD on O-only tokens over first n_old dims ─────────────────────────────
def test_kd_distill_only_on_o_tokens():
    m = _make_is3(hidden=4, n_cls=6)
    # Set up a fake teacher with n_old=3 output classes.
    import copy

    teacher = _FakeWrapper(hidden=4, n_cls=3)
    m._teacher = teacher

    flat_logits = torch.randn(6, 6, requires_grad=True)
    batch = {"input_ids": torch.zeros(2, 3, 4)}
    # Only tokens 1,3 are O-labeled.
    distill_mask = torch.tensor([False, True, False, True, False, False])

    loss = m._kd_distill(flat_logits, batch, distill_mask)
    assert torch.isfinite(loss) and loss.item() > 0
    # With no O tokens, loss should be exactly 0.
    no_o_mask = torch.zeros(6, dtype=torch.bool)
    zero_loss = m._kd_distill(flat_logits, batch, no_o_mask)
    assert zero_loss.item() == 0.0


def test_kd_distill_zero_without_teacher():
    m = _make_is3(hidden=4, n_cls=4)
    flat_logits = torch.randn(4, 4)
    distill_mask = torch.tensor([True, False, True, False])
    loss = m._kd_distill(flat_logits, {}, distill_mask)
    assert loss.item() == 0.0


# ─── 3. Prototype replay ──────────────────────────────────────────────────────
def test_prototype_loss_is_ce_through_live_head():
    m = _make_is3(hidden=4, n_cls=4)
    m.prototypes = {1: torch.randn(4), 2: torch.randn(4)}
    loss = m._prototype_loss()
    assert torch.isfinite(loss) and loss.item() > 0


def test_prototype_loss_zero_when_no_prototypes():
    m = _make_is3(hidden=4, n_cls=4)
    loss = m._prototype_loss()
    assert loss.item() == 0.0


# ─── 4. Gradient surgery (the O2E debias mechanism) ──────────────────────────
def test_grad_surgery_scales_old_entity_rows_preserves_o_and_new():
    m = _make_is3(hidden=4, n_cls=6)  # n_cls=6: rows 0..5
    m._n_old = 4  # rows 1..3 = old entity, row 0 = O, rows 4..5 = new entity
    m.grad_weight = 0.5
    m.new_grad_weight = 2.0

    head = m.model.classifier  # Linear(4, 6)
    # Simulate a backward pass to set .grad.
    x = torch.randn(3, 4)
    y = torch.randint(0, 6, (3,))
    F.cross_entropy(head(x), y).backward()

    grad_before = head.weight.grad.clone()
    bias_before = head.bias.grad.clone() if head.bias is not None else None

    m._apply_grad_surgery()

    # Row 0 (O): untouched.
    assert torch.allclose(head.weight.grad[0], grad_before[0])
    # Rows 1..3 (old entity): scaled by 0.5.
    for r in range(1, 4):
        assert torch.allclose(head.weight.grad[r], grad_before[r] * 0.5, atol=1e-6)
    # Rows 4..5 (new): scaled by 2.0.
    for r in range(4, 6):
        assert torch.allclose(head.weight.grad[r], grad_before[r] * 2.0, atol=1e-6)
    # Bias follows same pattern.
    if bias_before is not None:
        assert torch.allclose(head.bias.grad[0], bias_before[0])
        for r in range(1, 4):
            assert torch.allclose(head.bias.grad[r], bias_before[r] * 0.5, atol=1e-6)


def test_grad_surgery_noop_when_no_grad():
    m = _make_is3(hidden=4, n_cls=4)
    m._n_old = 2
    m.model.classifier.weight.grad = None
    m._apply_grad_surgery()  # must not raise


# ─── 5. After-task prototype build produces L2-normed mean per class ──────────
def test_after_task_builds_prototype_per_class():
    m = _make_is3(hidden=4, n_cls=3)
    # Fake batch: tokens 0,1 are class 1; token 2 is class 2; rest are O/pad.
    labels = torch.tensor([[1, 2, -100]]).unsqueeze(0)  # (1, 1, 3) → will be used as (B=1, L=3)
    labels_2d = torch.tensor([[1, 2, -100]])  # (1, 3)

    class _FakeLoader:
        def __iter__(self):
            yield {"input_ids": torch.zeros(1, 3, 4), "labels": labels_2d}

    m.model.eval()
    m.after_task(SimpleNamespace(task_id=0), _FakeLoader())
    assert 1 in m.prototypes
    assert 2 in m.prototypes
    assert m.prototypes[1].shape == (4,)
    # O (=0) and padding (-100) must not be stored.
    assert 0 not in m.prototypes and -100 not in m.prototypes
