"""ER-SBR (raw-document memory x blocked readout) — synthetic tensors, no backbone, no dataset."""

import torch
from torch import nn

from doccl.methods.er_sbr import ERSBR
from doccl.types import TaskInfo

D, L, NL = 16, 6, 5  # head: O + B/I-x (block 0) + B/I-y (block 1)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.embeddings = nn.Embedding(10, D)
        inner.encoder = nn.Module()
        inner.encoder.layer = nn.ModuleList(_Layer() for _ in range(4))
        hf = nn.Module()
        hf.layoutlmv3 = inner
        hf.classifier = nn.Linear(D, NL)
        hf.config = type("C", (), {"hidden_size": D, "num_labels": NL})()
        self.model = hf
        self.label_to_id = {"O": 0, "B-x": 1, "I-x": 2, "B-y": 3, "I-y": 4}

    def freeze_backbone(self):
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = False

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, bbox, pixel_values, attention_mask=None, labels=None):
        h = self.model.layoutlmv3.embeddings(input_ids) + pixel_values.mean() + bbox.float().mean()
        for layer in self.model.layoutlmv3.encoder.layer:
            h = layer(h, attention_mask)[0]
        return type("Out", (), {"loss": None, "logits": self.model.classifier(h)})()


def _batch(b=2, hi=3):
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "pixel_values": torch.randn(b, 3, 4, 4),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, hi, (b, L)),
        "labels_full": torch.zeros(b, L, dtype=torch.long),
    }


def _task(i, labels):
    return TaskInfo(i, f"funsd_cil_t{i}", ["O", *labels], is_first=i == 0)


def test_raw_memory_no_freeze_and_full_network_replay():
    torch.manual_seed(0)
    m = ERSBR(_Wrapper(), {"docs_per_task": 3, "replay_batch_size": 2, "split_layer_k": 2})
    loader = [_batch(), _batch()]
    m.before_task(_task(0, ["B-x", "I-x"]), loader)
    m.after_task(_task(0, ["B-x", "I-x"]), loader)

    # nothing frozen after session 0
    assert all(p.requires_grad for p in m.model.parameters())
    # raw inputs banked (quota 3 of 4), no activations, session id, labels_full not kept
    assert len(m.store) == 3
    doc = m.store[0]
    assert set(doc) == {"input_ids", "bbox", "pixel_values", "attention_mask", "labels", "session"}
    assert torch.equal(doc["input_ids"], loader[0]["input_ids"][0])
    assert doc["pixel_values"].dtype == torch.float32 and int(doc["session"]) == 0
    assert m.memory_bytes() == sum(t.numel() * t.element_size() for t in doc.values()) * 3

    # session 1: replay is a plain forward of the stored docs; gradients reach the embeddings
    m.before_task(_task(1, ["B-y", "I-y"]), loader)
    replay = m._stack_replay(m.store[:2])
    out = m._replay_forward(replay)
    ref = m.model(**{k: v for k, v in replay.items() if k not in ("session", "labels")})
    assert torch.allclose(out.logits, ref.logits)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert m.model.model.layoutlmv3.embeddings.weight.grad.abs().sum() > 0
    assert m.model.model.layoutlmv3.encoder.layer[0].lin.weight.grad.abs().sum() > 0
