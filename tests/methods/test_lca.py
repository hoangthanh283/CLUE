"""Unit tests for the LCA port — TIES merge (verbatim) + the align loss, on synthetic
tensors (fast, GPU-free). The full lifecycle on a real LayoutLMv3 is in
``tests/methods/test_all_methods_e2e.py`` once ``lca`` is registered there.
"""

from __future__ import annotations

import torch

from doccl.methods.ties_merge import merge_state_dicts, merge_task_vectors, trim


def test_trim_keeps_topk_by_magnitude():
    t = torch.tensor([0.1, -3.0, 0.2, 2.0])
    trimmed, gamma, mu = trim(t, topk=50)  # keep top 2 of 4
    # the two largest |.| are -3.0 and 2.0; others zeroed.
    assert trimmed.tolist() == [0.0, -3.0, 0.0, 2.0]
    assert gamma.tolist() == [0.0, -1.0, 0.0, 1.0]
    assert torch.allclose(mu, torch.tensor([0.0, 3.0, 0.0, 2.0]))


def test_trim_topk_100_keeps_everything():
    t = torch.tensor([0.1, -3.0, 0.2, 2.0])
    trimmed, _, _ = trim(t, topk=100)
    assert torch.allclose(trimmed, t)  # no trimming at 100%


def test_merge_task_vectors_sign_election_and_disjoint_mean():
    # coord 0: +2, +4 agree → mean(2,4)=3 ; coord 1: +1 vs -5 → elected sign = -(|−5|>|1|) →
    # sum signs = 0 → sign(0)=0 → only entries matching 0 kept = none → 0.
    tv0 = trim(torch.tensor([2.0, 1.0]), topk=100)
    tv1 = trim(torch.tensor([4.0, -5.0]), topk=100)
    merged = merge_task_vectors([tv0, tv1])
    assert abs(merged[0].item() - 3.0) < 1e-6
    assert merged[1].item() == 0.0  # sign conflict with summed-sign 0 → dropped


def test_merge_state_dicts_ties_is_base_plus_merged():
    base = {"w": torch.zeros(3)}
    t0 = {"w": torch.tensor([2.0, 0.0, 1.0])}
    t1 = {"w": torch.tensor([4.0, 0.0, 1.0])}
    out = merge_state_dicts(base, [t0, t1], method="ties", lamb=1.0, topk=100)
    # coord0: mean(2,4)=3 ; coord1: both 0 → 0 ; coord2: mean(1,1)=1.
    assert torch.allclose(out["w"], torch.tensor([3.0, 0.0, 1.0]), atol=1e-6)


def test_merge_state_dicts_lamb_scales_task_vector():
    base = {"w": torch.tensor([1.0, 1.0])}
    t0 = {"w": torch.tensor([3.0, 3.0])}  # tv = [2,2]
    out = merge_state_dicts(base, [t0], method="ties", lamb=0.5, topk=100)
    # base + 0.5 * merged([2,2]) = [1,1] + 0.5*[2,2] = [2,2].
    assert torch.allclose(out["w"], torch.tensor([2.0, 2.0]), atol=1e-6)


def test_merge_state_dicts_max_abs():
    base = {"w": torch.zeros(2)}
    t0 = {"w": torch.tensor([3.0, -1.0])}
    t1 = {"w": torch.tensor([-2.0, 5.0])}
    out = merge_state_dicts(base, [t0, t1], method="max_abs", lamb=1.0)
    # per coord pick the larger-|.| task vector: coord0 → 3.0 ; coord1 → 5.0.
    assert torch.allclose(out["w"], torch.tensor([3.0, 5.0]), atol=1e-6)


def test_merge_state_dicts_empty_tasks_returns_base_copy():
    base = {"w": torch.tensor([1.0, 2.0])}
    out = merge_state_dicts(base, [], method="ties")
    assert torch.allclose(out["w"], base["w"]) and out["w"] is not base["w"]


def test_align_loss_plain_ce_when_weights_zero():
    """With robust+entropy weights 0, the align loss is plain CE on sampled features."""
    from doccl.methods.lca import LCA

    m = LCA.__new__(LCA)
    m.ca_robust_weight = 0.0
    m.ca_entropy_weight = 0.0
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    x = torch.randn(8, 5)
    means = torch.randn(5, 5)
    loss = m._align_loss(logits, y, x, means, list(range(5)))
    assert torch.allclose(loss, torch.nn.functional.cross_entropy(logits, y), atol=1e-6)


def test_align_loss_robust_term_is_finite_and_nonneg():
    from doccl.methods.lca import LCA

    m = LCA.__new__(LCA)
    m.ca_robust_weight = 0.1
    m.ca_entropy_weight = 0.0
    torch.manual_seed(0)
    x = torch.randn(12, 6)
    means = torch.randn(4, 6)
    # labels = nearest mean so clusters are populated (exercises term2).
    y = torch.cdist(x, means).argmin(dim=1)
    logits = torch.randn(12, 4, requires_grad=True)
    loss = m._align_loss(logits, y, x, means, list(range(4)))
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()  # differentiable wrt logits
    assert logits.grad is not None
