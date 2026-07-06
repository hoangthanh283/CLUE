"""Unit tests for LatentReplay internals (synthetic tensors, fast, GPU-free).

Covers the three load-bearing mechanics: the early-freeze map (layers < k +
embeddings frozen, layers >= k + head plastic), activation capture into the
fp16 CPU store, and the pre-hook injection path (stored hidden actually
replaces the input to layer k; replay gradients reach only the plastic part).
Full lifecycle on real LayoutLMv3 is in tests/methods/test_all_methods_e2e.py.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.latent_replay import LatentReplay

D, L, NL, N_LAYERS, K = 16, 6, 4, 4, 2


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Wrapper(nn.Module):
    """Minimal stand-in matching the attribute paths LatentReplay touches."""

    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.embeddings = nn.Embedding(10, D)
        encoder = nn.Module()
        encoder.layer = nn.ModuleList(_Layer() for _ in range(N_LAYERS))
        inner.encoder = encoder
        hf = nn.Module()
        hf.layoutlmv3 = inner
        hf.classifier = nn.Linear(D, NL)
        self.model = hf
        self.processor = type("P", (), {"tokenizer": type("T", (), {"pad_token_id": 1})()})()

    def freeze_backbone(self):
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = False

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, bbox, pixel_values, attention_mask=None, labels=None):
        h = self.model.layoutlmv3.embeddings(input_ids)
        for layer in self.model.layoutlmv3.encoder.layer:
            h = layer(h, attention_mask)[0]  # module __call__ so pre-hooks fire
        logits = self.model.classifier(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=2):
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "pixel_values": torch.randn(b, 3, 4, 4),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, NL, (b, L)),
    }


def _method(docs=2):
    torch.manual_seed(0)
    return LatentReplay(
        _Wrapper(), {"split_layer_k": K, "docs_per_task": docs, "replay_batch_size": 2}
    )


def test_freeze_map_splits_at_k():
    lr = _method()
    lr._apply_freeze_map()
    layers = lr._encoder_layers()
    assert all(not p.requires_grad for p in lr.model.model.layoutlmv3.embeddings.parameters())
    assert all(not p.requires_grad for la in layers[:K] for p in la.parameters())
    assert all(p.requires_grad for la in layers[K:] for p in la.parameters())
    assert all(p.requires_grad for p in lr.model.model.classifier.parameters())


def test_capture_banks_quota_fp16_cpu():
    lr = _method(docs=3)
    lr._capture_task([_batch(2), _batch(2)])  # 4 docs available, quota 3
    assert len(lr.store) == 3
    doc = lr.store[0]
    assert doc["hidden"].dtype == torch.float16
    assert doc["hidden"].device.type == "cpu"
    assert doc["hidden"].shape == (L, D)
    assert lr._capture is None  # flag reset even on success


def test_injection_replaces_layer_k_input():
    lr = _method(docs=2)
    lr._capture_task([_batch(2)])
    const = torch.full((L, D), 0.5)
    lr.store = [{**lr.store[0], "hidden": const.to(torch.float16)}]
    replay = lr._sample_replay()
    out = lr._replay_forward(replay)
    with torch.no_grad():
        h = const.unsqueeze(0).float()
        for layer in lr._encoder_layers()[K:]:
            h = layer(h)[0]
        expected = lr.model.model.classifier(h)
    assert lr._inject is None  # flag reset after the pass
    assert torch.allclose(out.logits, expected, atol=1e-3)


def test_replay_grads_reach_only_plastic_layers():
    lr = _method(docs=2)
    lr._apply_freeze_map()
    lr._capture_task([_batch(2)])
    out = lr._replay_forward(lr._sample_replay())
    out.loss.backward()
    layers = lr._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )
    assert lr.model.model.classifier.weight.grad.abs().sum() > 0


def test_sample_replay_empty_store_is_none():
    lr = _method(docs=0)
    assert lr._sample_replay() is None
