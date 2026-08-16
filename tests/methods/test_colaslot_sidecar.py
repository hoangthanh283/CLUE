from types import SimpleNamespace

import torch
from torch import nn

from doccl.methods.colaslot_sidecar import CoLaSlotSidecar


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.0))
        self.slot = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        _ = torch.rand(1)
        return SimpleNamespace(loss=(self.base + self.slot).square())


def test_slot_loss_is_rng_isolated_and_slot_only():
    method = object.__new__(CoLaSlotSidecar)
    method.model = _ToyModel()
    method._sidecar_read = False
    method._slot_parameters = lambda: [method.model.slot]
    rng_before = torch.random.get_rng_state()

    loss = method._backward_slot_loss({})

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert method.model.base.requires_grad and method.model.base.grad is None
    assert method.model.slot.grad is not None
    assert not method._sidecar_read
    assert not loss.requires_grad


def test_stage_policy_suppresses_only_newest_owner():
    slots = SimpleNamespace(_infer_gate=torch.ones(1, 4), slot_owner=[0, 0, 1, 1])

    CoLaSlotSidecar._apply_read_policy(slots, read=True, newest=1, stage_eval=True)

    assert torch.equal(slots._infer_gate, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_diagnostic_forward_uses_routed_stage_policy():
    method = object.__new__(CoLaSlotSidecar)
    method._sidecar_read = False
    method._stage_eval = False

    with method.diagnostic_forward_context():
        assert method._sidecar_read and method._stage_eval

    assert not method._sidecar_read and not method._stage_eval
