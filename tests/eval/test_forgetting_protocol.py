"""Forgetting-localisation protocol on a synthetic backbone (fast, GPU-free)."""

from __future__ import annotations

import torch
from torch import nn

from doccl.diag import (
    BoundaryReport,
    ForgettingLocaliser,
    dominance_test,
    freeze_groups,
    shared_location_test,
)
from doccl.models import param_grouping

D, L, NL, N_LAYERS = 8, 5, 3, 3


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, h, attention_mask=None):
        return (torch.tanh(self.lin(h)),)


class _Backbone(nn.Module):
    """Tiny model honouring the wrapper contract the protocol relies on."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.embeddings = nn.Embedding(10, D)
        inner.encoder = nn.Module()
        inner.encoder.layer = nn.ModuleList(_Layer() for _ in range(N_LAYERS))
        self.model = nn.Module()
        self.model.enc = inner
        self.model.classifier = nn.Linear(D, NL)

    @property
    def param_groups_by_depth(self):
        return param_grouping.param_groups_by_depth(self.model, N_LAYERS)

    @property
    def cka_layers(self):
        return ["model.enc.encoder.layer.1", "model.classifier"]

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        h = self.model.enc.embeddings(input_ids)
        for layer in self.model.enc.encoder.layer:
            h = layer(h, attention_mask)[0]
        logits = self.model.classifier(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _loader(n=4):
    torch.manual_seed(0)
    return [
        {
            "input_ids": torch.randint(0, 10, (2, L)),
            "attention_mask": torch.ones(2, L, dtype=torch.long),
            "labels": torch.randint(0, NL, (2, L)),
        }
        for _ in range(n)
    ]


def _localise(perturb: str) -> BoundaryReport:
    torch.manual_seed(1)
    m = _Backbone()
    loader = _loader()
    loc = ForgettingLocaliser(m, device="cpu", fisher_n=8, cka_n=40)
    loc.arm(loader)
    with torch.no_grad():
        if perturb == "head":
            m.model.classifier.weight.add_(0.5)
        else:  # early layer only
            m.model.enc.encoder.layer[0].lin.weight.add_(0.5)
    return loc.measure(loader, "0_to_1")


def test_localise_attributes_drift_to_the_moved_group():
    head = _localise("head")
    assert head.dominant() == "head"
    assert head.displacement["early"] == 0.0 and head.displacement["mid"] == 0.0
    assert head.cka["model.classifier"] < 1.0 - 1e-3  # head output drifted
    early = _localise("early")
    assert early.dominant() == "early"
    assert early.displacement["head"] == 0.0
    # moving layer 0 drifts everything downstream; CKA at layer 1 must drop too
    assert early.cka["model.enc.encoder.layer.1"] < 1.0 - 1e-3
    assert abs(sum(head.profile().values()) - 1.0) < 1e-6


def test_measure_requires_arm():
    loc = ForgettingLocaliser(_Backbone(), device="cpu")
    try:
        loc.measure(_loader(), "x")
    except RuntimeError:
        return
    raise AssertionError("measure() must refuse to run before arm()")


def _rep(**d):
    return BoundaryReport("b", d, d, {}, {})


def test_dominance_test_flags_head_when_it_dominates_across_seeds():
    reps = [_rep(head=10.0 + i, late=1.0, early=0.1) for i in range(4)]
    out = dominance_test(reps)
    assert out["dominant"] == "head" and out["share"] > 0.85 and out["n_seeds"] == 4
    assert out["reject_H0"]  # 4 vs 4 Mann-Whitney at p ~ 0.03 < 0.025? Bonferroni 0.05/2 = 0.025
    flat = [_rep(head=1.0, late=1.0, early=1.0) for _ in range(3)]
    assert not dominance_test(flat)["reject_H0"]


def test_shared_location_test_distinguishes_same_vs_shifted_profiles():
    # a and b are exchangeable draws from one profile (interleaved jitter); c is relocated
    a = [_rep(head=9.0 + 0.4 * i, late=1.0 + 0.1 * i, early=0.1) for i in range(5)]
    b = [_rep(head=9.2 + 0.4 * i, late=1.05 + 0.1 * i, early=0.1) for i in range(5)]
    c = [_rep(head=0.1, late=1.0, early=9.0 + 0.1 * i) for i in range(5)]
    assert shared_location_test(a, b, n_perm=300)["shared_location"]
    assert not shared_location_test(a, c, n_perm=300)["shared_location"]


def test_freeze_groups_protects_only_named_groups():
    m = _Backbone()
    n = freeze_groups(m, ["head", "late"])
    assert n == sum(p.numel() for p in m.model.classifier.parameters()) + sum(
        p.numel() for p in m.model.enc.encoder.layer[-1].parameters()
    )
    assert all(p.requires_grad for p in m.model.enc.encoder.layer[0].parameters())
