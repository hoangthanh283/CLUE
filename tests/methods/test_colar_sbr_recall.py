"""CoLaR-SBR-Recall — SessionLoRA checks (no backbone)."""

import torch
from torch import nn

from doccl.methods.colar_sbr_recall import SessionLoRA


def test_fresh_delta_is_identity_and_base_only_when_inactive():
    torch.manual_seed(0)
    lin = nn.Linear(6, 4)
    m = SessionLoRA(lin, rank=2, alpha=4)
    x = torch.randn(3, 5, 6)
    assert torch.equal(m(x), lin(x))  # no delta yet
    i = m.add_session()
    m.active = i
    assert torch.allclose(m(x), lin(x))  # B = 0 -> identity at init
    m.active = None
    assert torch.equal(m(x), lin(x))


def test_deltas_are_independent_per_session():
    torch.manual_seed(0)
    lin = nn.Linear(6, 4)
    m = SessionLoRA(lin, rank=2, alpha=4)
    a, b = m.add_session(), m.add_session()
    with torch.no_grad():
        m.B[a].fill_(1.0)
    x = torch.randn(2, 6)
    m.active = a
    ya = m(x)
    m.active = b
    yb = m(x)
    assert not torch.allclose(ya, lin(x))
    assert torch.allclose(yb, lin(x))  # session b untouched by session a's write
    assert sum(p.numel() for p in m.parameters()) == lin.weight.numel() + lin.bias.numel() + 2 * (
        2 * 6 + 4 * 2
    )
