"""Unit tests for CoresetMemory internals — synthetic, fast, GPU-free.

Covers the fit (k-means real-activation centroids per class), reconstruction into a
label-coupled injectable batch drawn from REAL centroids, the inherited plastic-only
gradient invariant, and the wider-hidden-than-mask (image-patch) case. CoresetMemory is
the one feature-replay variant not falsified at Gate 0.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.coreset_memory import CoresetMemory
from doccl.types import TaskInfo

D, L, NL, N_LAYERS, K = 16, 8, 4, 4, 2


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = D
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

    def forward(self, input_ids, bbox, pixel_values=None, attention_mask=None, labels=None):
        h = self.model.layoutlmv3.embeddings(input_ids)
        for layer in self.model.layoutlmv3.encoder.layer:
            h = layer(h, attention_mask)[0]
        logits = self.model.classifier(h)
        if labels is not None:
            text_logits = logits[:, : labels.shape[1], :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        else:
            loss = None
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=4):
    labels = torch.randint(0, NL, (b, L))
    labels[:, 0] = -100
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": labels,
    }


def _method(mpc=3, carriers=2):
    torch.manual_seed(0)
    return CoresetMemory(
        _Wrapper(),
        {
            "split_layer_k": K,
            "centroids_per_class": mpc,
            "attn_keep": 1.0,
            "carriers_per_task": carriers,
            "replay_batch_size": 2,
        },
    )


def _task(tid=0):
    return TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=[f"L{i}" for i in range(NL)])


def test_fit_produces_real_centroids_per_class():
    cm = _method(mpc=3, carriers=2)
    cm._fit_coreset(_task(0), [_batch(6), _batch(6)])
    assert 0 in cm._coreset
    entry = cm._coreset[0]
    assert entry["centroids"].shape[1] == D  # centroids live in feature space
    assert len(entry["rows_for"]) >= 1  # at least one class
    # rows_for indices must index into the centroid block
    total = entry["centroids"].shape[0]
    for rows in entry["rows_for"].values():
        assert rows.max().item() < total
        assert rows.numel() <= 3  # at most centroids_per_class per class


def test_sample_replay_uses_real_centroids():
    """Every replayed feature vector must EQUAL one of the stored real centroids (not a
    synthesised sample) — this is what distinguishes coreset from the falsified Gaussian variants.
    """
    cm = _method(mpc=3, carriers=2)
    cm._fit_coreset(_task(0), [_batch(6)])
    replay = cm._sample_replay()
    assert replay is not None
    assert replay["hidden"].shape == (2, L, D)
    cents = cm._coreset[0]["centroids"].to(torch.float16)
    hid = replay["hidden"]
    # each token vector matches some stored centroid row exactly (real, not synthetic)
    for b in range(hid.shape[0]):
        for t in range(hid.shape[1]):
            v = hid[b, t]
            match = (cents == v).all(dim=1).any()
            assert match, "replayed vector is not a stored real centroid"


def test_injection_reaches_only_plastic_layers():
    cm = _method(mpc=3, carriers=2)
    cm._apply_freeze_map()
    cm._fit_coreset(_task(0), [_batch(6)])
    out = cm._replay_forward(cm._sample_replay())
    out.loss.backward()
    layers = cm._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_empty_bank_replays_none():
    cm = _method()
    assert cm._sample_replay() is None


def test_reconstructs_at_wider_hidden_than_mask():
    N_PATCH = 5  # noqa: N806 — constant

    class _PatchLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, D)

        def forward(self, hidden_states, attention_mask=None, **kwargs):
            h = self.lin(hidden_states)
            if h.shape[1] == L:
                h = torch.cat([h, torch.zeros(h.shape[0], N_PATCH, D)], dim=1)
            return (h,)

    torch.manual_seed(0)
    w = _Wrapper()
    w.model.layoutlmv3.encoder.layer = nn.ModuleList(_PatchLayer() for _ in range(N_LAYERS))
    cm = CoresetMemory(
        w,
        {
            "split_layer_k": K,
            "centroids_per_class": 3,
            "attn_keep": 1.0,
            "carriers_per_task": 2,
            "replay_batch_size": 2,
        },
    )
    cm._fit_coreset(_task(0), [_batch(6)])
    assert cm._hidden_width == L + N_PATCH
    replay = cm._sample_replay()
    assert replay["hidden"].shape == (2, L + N_PATCH, D)
