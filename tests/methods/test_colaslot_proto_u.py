from types import MethodType, SimpleNamespace

import torch

from doccl.methods.colaslot_proto_u import CoLaSlotProtoUtility


def test_entity_f1_counts_wrong_entity_as_false_positive_and_negative():
    predicted = torch.tensor([0, 1, 2, 2])
    labels = torch.tensor([0, 1, 1, 0])
    assert CoLaSlotProtoUtility._entity_f1(predicted, labels, o_id=0) == 0.4


def test_owner_utility_runs_leave_one_document_out():
    method = object.__new__(CoLaSlotProtoUtility)
    method.device = torch.device("cpu")
    method.head_slots = SimpleNamespace(slot_owner=[0])
    method.model = SimpleNamespace(label_to_id={"O": 0})
    docs = []
    for entity_feature in (torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.995])):
        docs.append(
            {
                "labels": torch.tensor([[0, 1]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "feats": torch.stack((torch.tensor([1.0, 0.0]), entity_feature)).unsqueeze(0),
                "logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
            }
        )
    method._stack_replay = lambda batch: batch[0]

    def replay_forward(self, replay):
        self._cur_feats = replay["feats"]
        return SimpleNamespace(logits=replay["logits"])

    method._replay_forward = MethodType(replay_forward, method)
    assert method._owner_utility(0, docs) == (1.0, 1.0)
