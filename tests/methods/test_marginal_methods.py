"""Unit tests for the marginal-anchored objectives (RCA kill-tests #4/#5) — hand-computed."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from doccl.methods.marginal_methods import LogitAdjust, MarginalAnchor, gold_marginal


class _StubModel(nn.Module):
    def __init__(self, num_labels: int = 3):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(config=SimpleNamespace(num_labels=num_labels))


def _loader(label_rows):
    return [{"labels": torch.tensor(rows)} for rows in label_rows]


def test_gold_marginal_ignores_minus_100():
    loader = _loader([[[0, 1, -100]], [[1, 1, 2]]])
    m = gold_marginal(loader, 3)
    assert torch.allclose(m, torch.tensor([1 / 5, 3 / 5, 1 / 5]))


def test_marginal_anchor_kl_zero_when_matched():
    method = MarginalAnchor(_StubModel(), {"lambda_kl": 1.0})
    method.task_marginals = [torch.tensor([0.5, 0.25, 0.25])]
    # logits whose softmax marginal equals the target exactly (single token per class mix)
    probs = torch.tensor([[0.5, 0.25, 0.25]]).log()
    outputs = SimpleNamespace(loss=torch.tensor(2.0), logits=probs.unsqueeze(0))
    batch = {"labels": torch.tensor([[0]])}
    loss = method._loss(outputs, batch)
    assert torch.isclose(loss, torch.tensor(2.0), atol=1e-5)  # KL term vanishes


def test_marginal_anchor_penalizes_extinct_class():
    method = MarginalAnchor(_StubModel(), {"lambda_kl": 1.0})
    method.task_marginals = [torch.tensor([0.5, 0.25, 0.25])]
    dead = torch.tensor([[[10.0, 10.0, -20.0]]])  # class 2 (in target) near-zero prob
    alive = torch.tensor([[[10.0, 10.0, 9.0]]])
    base = SimpleNamespace(loss=torch.tensor(0.0))
    batch = {"labels": torch.tensor([[0]])}
    l_dead = method._loss(SimpleNamespace(loss=base.loss, logits=dead), batch)
    l_alive = method._loss(SimpleNamespace(loss=base.loss, logits=alive), batch)
    assert l_dead > l_alive  # mode-covering KL punishes zeroing a target-mass class


def test_logit_adjust_noop_when_priors_equal():
    method = LogitAdjust(_StubModel(), {"tau": 1.0})
    method._task_marginal = torch.tensor([0.5, 0.25, 0.25])
    method._counts = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)  # q_cum == q_task
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]]])
    labels = torch.tensor([[0, 2]])
    outputs = SimpleNamespace(loss=torch.tensor(0.0), logits=logits)
    loss = method._loss(outputs, {"labels": labels})
    expected = F.cross_entropy(logits.view(-1, 3), labels.view(-1), ignore_index=-100)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_logit_adjust_shifts_toward_cumulative_prior():
    method = LogitAdjust(_StubModel(), {"tau": 1.0})
    # current task heavily favors class 0; cumulative is uniform
    method._task_marginal = torch.tensor([0.8, 0.1, 0.1])
    method._counts = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    logits = torch.tensor([[[0.0, 0.0, 0.0]]])
    labels = torch.tensor([[0]])
    loss = method._loss(SimpleNamespace(loss=None, logits=logits), {"labels": labels})
    # adjusted logit for class 0 is +log(0.8/(1/3)) — CE for gold 0 gets EASIER than raw,
    # so the model needs less raw score on the majority class (anti-snap direction).
    raw = F.cross_entropy(logits.view(-1, 3), labels.view(-1))
    assert loss < raw
