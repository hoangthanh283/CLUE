"""Unit tests for CPFD on synthetic tensors — GPU-free.

Tests cover the two core mechanisms:
  1. Pseudo-label relabeling (entropy < per-class threshold → pseudo old-class;
     entropy ≥ threshold → mask -100; non-O tokens untouched).
  2. Three-view attention-map MSE: head-pool, query-pool, key-pool per layer,
     averaged over layers. Returns 0 when attentions are absent.

The full lifecycle on a real LayoutLMv3 lives in test_all_methods_e2e.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from doccl.methods.cpfd import CPFD


def _make_cpfd() -> CPFD:
    m = CPFD.__new__(CPFD)
    m.threshold_floor = 0.001
    m.distill_weight = 2.0
    m.ref_temperature = 1.0
    m.adaptive_distill = True
    m.adaptive_ce = True
    m.adaptive_ce_min = 0.0
    m._teacher = None
    m._n_old = 0
    m._thresholds = None
    m.device = torch.device("cpu")
    return m


def test_entropy_uniform_is_log_n():
    m = _make_cpfd()
    import math

    probs = torch.full((2, 4, 3), 1 / 3)  # uniform over 3 classes
    entr = m._entropy(probs)  # shape (2, 4)
    expected = math.log(3)  # H(uniform) = log(n)
    assert torch.allclose(entr, torch.full_like(entr, expected), atol=1e-4)


def test_entropy_degenerate_is_zero():
    m = _make_cpfd()
    probs = torch.zeros(2, 4, 3)
    probs[:, :, 0] = 1.0  # one-hot → H = 0
    entr = m._entropy(probs)
    assert (entr.abs() < 1e-3).all()


def test_pseudo_label_confident_token_gets_old_class():
    m = _make_cpfd()
    m._n_old = 2
    # Per-class thresholds: class 0 → 0.5 (strict), class 1 → 10.0 (lenient).
    # Token 0: logits [5,-5] → argmax=0, entropy≈0 < 0.5 → confident → pseudo-label 0.
    # Token 1: logits [0,0]  → argmax=0 (tie), entropy≈0.693 > 0.5 → not confident → -100.
    m._thresholds = torch.tensor([0.5, 10.0])

    labels_2d = torch.tensor([[0, 0, 2]])  # (1, 3) — two O-tokens, one entity
    t_logits = torch.tensor(
        [
            [
                [5.0, -5.0],  # confident → class 0, entropy ≈ 0
                [0.0, 0.0],  # uncertain  → entropy ≈ 0.693 > 0.5
                [5.0, -5.0],
            ]
        ]
    )  # non-O → ignored
    new_labels, factor = m._apply_pseudo_labels(labels_2d, t_logits)

    assert new_labels[0, 0].item() == 0  # confident: class 0 pseudo-label assigned
    assert new_labels[0, 1].item() == -100  # uncertain: masked out
    assert new_labels[0, 2].item() == 2  # non-O entity: untouched


def test_pseudo_label_factor_is_coverage_fraction():
    m = _make_cpfd()
    m._n_old = 2
    m._thresholds = torch.tensor([10.0, 10.0])  # accept all (threshold very high)
    labels_2d = torch.tensor([[0, 0, 2]])  # 2 O-tokens, 1 entity
    t_logits = torch.zeros(1, 3, 2)  # uniform → entropy = log(2), accepted by 10.0 thresh
    _, factor = m._apply_pseudo_labels(labels_2d, t_logits)
    # Both O-tokens get pseudo-labels → coverage = 2/2 = 1.0
    assert abs(factor[0, 0].item() - 1.0) < 1e-4


def test_pseudo_label_nonzero_tokens_untouched():
    m = _make_cpfd()
    m._n_old = 3
    m._thresholds = torch.tensor([10.0, 10.0, 10.0])
    labels_2d = torch.tensor([[1, 2, 3, -100]])  # no O-tokens
    t_logits = torch.zeros(1, 4, 3)
    new_labels, factor = m._apply_pseudo_labels(labels_2d, t_logits)
    assert (new_labels == labels_2d).all()  # unchanged
    assert factor[0, 0].item() == 0.0  # 0/0 = 0 (no O-tokens → factor=0)


def test_adaptive_coef_root_schedule():
    import math

    m = _make_cpfd()
    # 3 old classes, 5 total (2 new): sqrt(2/2) = 1.0
    assert abs(m._adaptive_coef(3, 5) - math.sqrt(2 / 2)) < 1e-6
    # 5 old, 6 total (1 new): sqrt(4/1) = 2.0
    assert abs(m._adaptive_coef(5, 6) - math.sqrt(4 / 1)) < 1e-6


def test_adaptive_coef_disabled_returns_1():
    m = _make_cpfd()
    m.adaptive_distill = False
    assert m._adaptive_coef(10, 15) == 1.0


def test_attn_mse_identical_attentions_is_zero():
    m = _make_cpfd()
    attn = torch.randn(2, 4, 8, 8)  # (B, heads, L, L)
    loss = m._attn_mse((attn,), (attn,))
    assert abs(loss.item()) < 1e-6


def test_attn_mse_different_attentions_is_positive():
    m = _make_cpfd()
    a = torch.zeros(2, 4, 8, 8)
    b = torch.ones(2, 4, 8, 8)
    loss = m._attn_mse((a,), (b,))
    assert loss.item() > 0


def test_attn_mse_averaged_over_layers():
    m = _make_cpfd()
    # 2 layers, identical → 0 regardless
    attn = torch.randn(1, 2, 4, 4)
    loss = m._attn_mse((attn, attn), (attn, attn))
    assert abs(loss.item()) < 1e-6


def test_attn_mse_empty_returns_zero():
    m = _make_cpfd()
    loss = m._attn_mse((), ())
    assert loss.item() == 0.0
