from types import MethodType, SimpleNamespace

import torch

from doccl.methods.colaslot_proto import CoLaSlotProto


def _method() -> CoLaSlotProto:
    method = object.__new__(CoLaSlotProto)
    method._slot_reads_enabled = True
    method._prototypes = {0: {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}}
    method._prototype_scale = {0: 2.0}
    method._support_keys = {0: torch.tensor([[0.0, 1.0]])}
    method._support_enabled = {0: True}
    method.head_slots = SimpleNamespace(_infer_gate=torch.ones(1, 1), slot_owner=[0])
    method.model = SimpleNamespace(
        model=SimpleNamespace(classifier=SimpleNamespace(out_features=2))
    )
    return method


def test_prototype_margin_is_zero_when_slot_reads_are_disabled():
    method = _method()
    feats = torch.tensor([[[1.0, 0.0]]])
    method._slot_reads_enabled = False
    assert torch.equal(method._head_delta(feats), torch.zeros(1, 1, 2))


def test_prototype_margin_targets_entity_relative_to_o():
    method = _method()
    delta = method._head_delta(torch.tensor([[[0.0, 1.0]]]))
    assert delta.shape == (1, 1, 2)
    assert delta[0, 0, 1] > 0


def test_prototype_fitting_preserves_rng_and_model_mode():
    method = object.__new__(CoLaSlotProto)
    method.model = torch.nn.Linear(2, 2)
    method.model.train()
    method.store = [{}]
    method._store_task_ids = [0]
    method._support_enabled = {0: True}
    method._slot_reads_enabled = True
    method._forced_slot_owner = None
    method.diagnostic_metrics = {"prototype": {}}
    replay = {
        "labels": torch.tensor([[0, 1]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    method._stack_replay = lambda _docs: replay

    def replay_forward(self, _replay):
        torch.rand(1)
        self._cur_feats = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        return SimpleNamespace(logits=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

    method._replay_forward = MethodType(replay_forward, method)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    method._fit_prototypes(task_id=1)

    assert torch.equal(torch.rand(3), expected)
    assert method.model.training
    assert method._slot_reads_enabled
    assert method._forced_slot_owner is None
