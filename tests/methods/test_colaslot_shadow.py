from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from doccl.methods.colaslot_shadow import CoLaSlotShadow, CoLaSlotShadowReplay
from doccl.methods.colaslot_sidecar import CoLaSlotSidecar


def test_shadow_initialization_preserves_training_rng():
    layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    model = nn.Module()

    def fake_parent_init(method, _model, _config):
        method.model = model
        method.device = torch.device("cpu")
        method.hidden_dim = 4
        method.repr_rank = 2
        method.split_layer_k = 1
        method._encoder_layers = lambda: layers

    rng_before = torch.random.get_rng_state()
    with patch.object(CoLaSlotSidecar, "__init__", fake_parent_init):
        method = CoLaSlotShadow(model, {})

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert sum(parameter.numel() for parameter in method.shadow_slots.parameters()) > 0


def test_shadow_gate_follows_any_aged_owner_route():
    shadow = {"4": SimpleNamespace(_infer_gate=None), "5": SimpleNamespace(_infer_gate=None)}
    owner_gate = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    CoLaSlotShadow._apply_shadow_gate(shadow, owner_gate)

    expected = torch.tensor([[1.0], [0.0]])
    assert all(torch.equal(module._infer_gate, expected) for module in shadow.values())


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.0))
        self.slot = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        _ = torch.rand(1)
        return SimpleNamespace(loss=(self.base + self.slot).square())


def test_shadow_reuses_replay_without_base_grad_or_rng_drift():
    method = object.__new__(CoLaSlotShadow)
    method.model = _ToyModel()
    method.device = torch.device("cpu")
    method._sidecar_read = False
    method._slot_parameters = lambda: [method.model.slot]
    replay = {"token": torch.tensor(7)}
    seen = []
    method._sidecar_replay_loss = lambda supplied: (seen.append(supplied) or method.model().loss)
    rng_before = torch.random.get_rng_state()

    loss = method._backward_slot_loss({}, replay)

    assert seen == [replay]
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert method.model.base.requires_grad and method.model.base.grad is None
    assert method.model.slot.grad is not None
    assert not loss.requires_grad


def test_replay_shadow_is_off_for_current_and_on_for_replay_or_aged_eval():
    method = object.__new__(CoLaSlotShadowReplay)
    method.shadow_slots = {"4": SimpleNamespace(_infer_gate=None)}
    method._stage_eval = False
    method._routing_input_ids = None

    def install_owner_gate(_self, _module, _args, _kwargs):
        method.shadow_slots["4"]._infer_gate = torch.ones(2, 1)

    with patch.object(CoLaSlotShadow, "_install_infer_gate", install_owner_gate):
        method._install_infer_gate(None, (), {})
        assert not method.shadow_slots["4"]._infer_gate.any()

        method._routing_input_ids = torch.ones(2, 3, dtype=torch.long)
        method._install_infer_gate(None, (), {})
        assert method.shadow_slots["4"]._infer_gate.all()

        method._routing_input_ids = None
        method._stage_eval = True
        method._install_infer_gate(None, (), {})
        assert method.shadow_slots["4"]._infer_gate.all()
