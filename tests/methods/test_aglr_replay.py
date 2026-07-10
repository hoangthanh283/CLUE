"""Unit tests for AGLRReplay (AGLR-CL port) internals — synthetic, fast, GPU-free.

Covers the comparator's memory representation: per-(task x class) Gaussians in layer-k
activation space fit on salience-filtered tokens, reconstruction into an injectable replay
batch keyed by carrier labels, and the inherited plastic-only gradient invariant. This is
the Gate-0 baseline SLR must beat; the tests pin its mechanics so the comparison is honest.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.aglr_replay import AGLRReplay
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
        # Mirror LayoutLMv3: slice logits to text length before the loss.
        if labels is not None:
            text_logits = logits[:, : labels.shape[1], :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        else:
            loss = None
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=2, with_ignore=True):
    labels = torch.randint(0, NL, (b, L))
    if with_ignore:
        labels[:, 0] = -100  # some padding/ignored tokens, as in real collated batches
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": labels,
    }


def _method(attn_keep=0.5, carriers=3):
    torch.manual_seed(0)
    return AGLRReplay(
        _Wrapper(),
        {
            "split_layer_k": K,
            "attn_keep": attn_keep,
            "carriers_per_task": carriers,
            "replay_batch_size": 2,
        },
    )


def _task(tid=0):
    return TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=[f"L{i}" for i in range(NL)])


def test_fit_produces_per_class_gaussians_and_carriers():
    m = _method(attn_keep=1.0, carriers=3)  # keep all valid tokens
    m._fit_gaussians(_task(0), [_batch(4), _batch(4)])
    assert 0 in m._bank
    g = m._bank[0]["gaussians"]
    assert len(g) >= 1
    assert all(0 <= c < NL for c in g)  # class keys in the label range; -100 excluded
    any_c = next(iter(g))
    assert g[any_c]["mean"].shape == (D,)
    assert g[any_c]["var"].shape == (D,)
    assert (g[any_c]["var"] > 0).all()  # variance floor applied
    assert len(m._bank[0]["carriers"]) == 3


def test_salience_filter_reduces_tokens():
    """attn_keep < 1 must fit on strictly fewer tokens than keep = 1."""
    full = _method(attn_keep=1.0)
    half = _method(attn_keep=0.5)
    xf, _, _ = full._capture_with_labels([_batch(4), _batch(4)])
    xh, _, _ = half._capture_with_labels([_batch(4), _batch(4)])
    assert xh.shape[0] < xf.shape[0]
    assert xh.shape[1] == D


def test_sample_replay_returns_injectable_batch():
    m = _method(attn_keep=1.0, carriers=2)
    m._fit_gaussians(_task(0), [_batch(4)])
    replay = m._sample_replay()
    assert replay is not None
    assert replay["hidden"].shape == (2, L, D)
    assert replay["hidden"].dtype == torch.float16
    for k in ("bbox", "attention_mask", "labels"):
        assert replay[k].shape[0] == 2


def test_injection_reaches_only_plastic_layers():
    m = _method(attn_keep=1.0, carriers=2)
    m._apply_freeze_map()
    m._fit_gaussians(_task(0), [_batch(4)])
    out = m._replay_forward(m._sample_replay())
    out.loss.backward()
    layers = m._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_empty_bank_replays_none():
    m = _method()
    assert m._sample_replay() is None


def test_reconstructs_at_wider_hidden_than_mask():
    """Image-patch tokens make the layer-k hidden wider than the text mask; capture must
    slice to text for the class-Gaussian fit and replay must reconstruct at full width."""
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
    m = AGLRReplay(
        w, {"split_layer_k": K, "attn_keep": 1.0, "carriers_per_task": 2, "replay_batch_size": 2}
    )
    m._fit_gaussians(_task(0), [_batch(4)])
    assert m._hidden_width == L + N_PATCH
    replay = m._sample_replay()
    assert replay["hidden"].shape == (2, L + N_PATCH, D)
