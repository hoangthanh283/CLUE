"""CoLaR-OTC assignment rules on synthetic tensors (fast, GPU-free)."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F  # noqa: N812

from doccl.methods.colar_otc import CoLaROTC, quota_assign, sinkhorn_assign, threshold_assign


def _scores(n: int = 400, k: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g).softmax(-1)


def test_sinkhorn_meets_both_marginals_when_balanced():
    probs = _scores()
    col_mass = torch.tensor([280.0, 60.0, 40.0, 20.0])  # sums to N
    plan = sinkhorn_assign(probs.log(), col_mass, n_iter=200)
    assert torch.allclose(plan.sum(1), torch.ones(400), atol=1e-4)
    assert torch.allclose(plan.sum(0), col_mass, rtol=2e-2)


def test_sinkhorn_fixes_the_mass_of_a_teacher_biased_to_background():
    """The failure this method targets: the teacher ranks the hidden entities first but keeps
    O dominant on them, so argmax assigns nothing. The plan assigns the prior's amount."""
    n, n_hidden = 200, 30
    probs = torch.full((n, 2), 0.0)
    probs[:, 1] = torch.linspace(0.40, 0.01, n)  # ranking correct, never above 0.5
    probs[:, 0] = 1 - probs[:, 1]
    truth = (torch.arange(n) < n_hidden).long()
    assert threshold_assign(probs, 0.5).sum() == 0
    plan = sinkhorn_assign(probs.log(), torch.tensor([n - n_hidden, n_hidden]).float())
    assert (plan.argmax(-1) == truth).float().mean() > 0.95


def test_sinkhorn_gives_nothing_to_a_zero_mass_column():
    plan = sinkhorn_assign(_scores().log(), torch.tensor([300.0, 100.0, 0.0, 0.0]))
    assert plan[:, 2:].sum() == 0
    assert torch.isfinite(plan).all()


def test_relaxed_columns_assign_less_than_an_inflated_prior_demands():
    probs = torch.full((100, 2), 0.0)
    probs[:, 1] = torch.cat([torch.full((10,), 0.9), torch.linspace(0.2, 0.01, 90)])
    probs[:, 0] = 1 - probs[:, 1]
    inflated = torch.tensor([60.0, 40.0])  # the prior claims 40 entities; 10 exist
    hard = sinkhorn_assign(probs.log(), inflated).argmax(-1).sum()
    soft = sinkhorn_assign(probs.log(), inflated, col_relax=0.5).argmax(-1).sum()
    assert abs(int(hard) - 40) <= 2  # the balanced plan obeys the wrong prior (argmax: +-1)
    assert 10 <= soft < hard


def test_quota_assign_takes_top_k_per_class_and_resolves_conflicts_by_score():
    probs = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.5, 0.4],
            [0.2, 0.2, 0.6],
            [0.9, 0.05, 0.05],
        ]
    )
    out = quota_assign(probs, torch.tensor([1.0, 2.0, 2.0]))
    # class 1 claims rows 0,1; class 2 claims rows 2,1; row 1 goes to the higher score (1)
    assert out.tolist() == [1, 1, 2, 0]


def test_threshold_assign_keeps_background_and_low_confidence_tokens():
    probs = torch.tensor([[0.05, 0.95], [0.4, 0.6], [0.8, 0.2]])
    assert threshold_assign(probs, 0.9).tolist() == [1, 0, 0]


def test_current_oracle_reads_true_labels_of_head_classes_only():
    logits = torch.randn(1, 4, 3)  # head: O, class 1, class 2; class 3 is a later session
    labels = torch.tensor([[0, 2, 0, -100]])  # CIL-masked: the class-1 token reads O
    labels_full = torch.tensor([[1, 2, 3, 1]])
    m = SimpleNamespace(config={"current_oracle": True})
    loss = CoLaROTC._current_ce(
        m, SimpleNamespace(logits=logits), {"labels": labels, "labels_full": labels_full}
    )
    want = F.cross_entropy(logits[0], torch.tensor([1, 2, 0, -100]), ignore_index=-100)
    assert torch.allclose(loss, want)
