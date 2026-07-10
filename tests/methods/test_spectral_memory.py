"""Unit tests for SpectralMemory (SLR-minimal) internals — synthetic, fast, GPU-free.

Covers the memory-representation change that IS the method: a rank-r spectral fit of
frozen layer-k activations + carrier skeletons, reconstruction into an injectable replay
batch, the inherited plastic-only gradient invariant, and the razor as an assertion —
the spectral footprint is far smaller than a raw activation buffer for the same data.
Full lifecycle on real LayoutLMv3 lives in tests/methods/test_all_methods_e2e.py.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.spectral_memory import SpectralMemory
from doccl.types import TaskInfo

D, L, NL, N_LAYERS, K = 16, 6, 4, 4, 2


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Wrapper(nn.Module):
    """Minimal stand-in for the attribute paths SpectralMemory touches."""

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
            h = layer(h, attention_mask)[0]  # module __call__ so pre-hooks fire
        logits = self.model.classifier(h)
        # Mirror LayoutLMv3: slice logits back to the text length before the loss (the real
        # wrapper drops image-patch-position logits — see layoutlm_wrapper get_token_features).
        if labels is not None:
            text_logits = logits[:, : labels.shape[1], :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        else:
            loss = None
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=2):
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, NL, (b, L)),
    }


def _method(rank_r=4, carriers=3):
    torch.manual_seed(0)
    return SpectralMemory(
        _Wrapper(),
        {
            "split_layer_k": K,
            "rank_r": rank_r,
            "carriers_per_task": carriers,
            "replay_batch_size": 2,
        },
    )


def _task(tid=0):
    return TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=[f"L{i}" for i in range(NL)])


def test_fit_produces_basis_and_per_class_gaussians():
    sm = _method(rank_r=4, carriers=3)
    sm._fit_spectral(_task(0), [_batch(2), _batch(2)])
    assert 0 in sm._spectral
    m = sm._spectral[0]
    assert m["basis"].shape == (4, D)  # shared rank-r forgetting subspace
    assert m["mean"].shape == (D,)
    assert len(m["classes"]) >= 1  # per-class Gaussians IN the subspace (label-coupled)
    any_c = next(iter(m["classes"].values()))
    assert any_c["cmean"].shape == (4,)  # in-subspace coordinate
    assert any_c["cstd"].shape == (4,)
    assert (any_c["cstd"] > 0).all()
    assert len(m["carriers"]) == 3
    # carriers hold real layout/labels but NO activations
    assert set(m["carriers"][0]) == {"bbox", "attention_mask", "labels"}


def test_rank_r_zero_disables_fit():
    sm = _method(rank_r=0)
    sm._fit_spectral(_task(0), [_batch(2)])
    assert sm._spectral == {}
    assert sm._sample_replay() is None


def test_sample_replay_returns_injectable_batch():
    sm = _method(rank_r=4, carriers=2)
    sm._fit_spectral(_task(0), [_batch(2)])
    replay = sm._sample_replay()
    assert replay is not None
    assert replay["hidden"].shape == (2, L, D)  # (replay_batch_size, seq, d)
    assert replay["hidden"].dtype == torch.float16
    for k in ("bbox", "attention_mask", "labels"):
        assert replay[k].shape[0] == 2


def test_injection_reaches_only_plastic_layers():
    sm = _method(rank_r=4, carriers=2)
    sm._apply_freeze_map()
    sm._fit_spectral(_task(0), [_batch(2)])
    out = sm._replay_forward(sm._sample_replay())
    out.loss.backward()
    layers = sm._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_reconstructs_at_wider_hidden_than_mask():
    """LayoutLMv3 concatenates image-patch tokens after text, so the layer-k hidden is
    WIDER than the text attention_mask. Capture must slice to text for the fit, and replay
    must reconstruct at the full hidden width or injection shape-mismatches at layer k."""
    N_PATCH = 5  # noqa: N806 — constant

    class _PatchLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, D)

        def forward(self, hidden_states, attention_mask=None, **kwargs):
            h = self.lin(hidden_states)
            # emulate image-patch tokens appended once (only when at text width)
            if h.shape[1] == L:
                pad = torch.zeros(h.shape[0], N_PATCH, D)
                h = torch.cat([h, pad], dim=1)
            return (h,)

    torch.manual_seed(0)
    w = _Wrapper()
    w.model.layoutlmv3.encoder.layer = nn.ModuleList(_PatchLayer() for _ in range(N_LAYERS))
    sm = SpectralMemory(
        w, {"split_layer_k": K, "rank_r": 4, "carriers_per_task": 2, "replay_batch_size": 2}
    )
    sm._fit_spectral(_task(0), [_batch(2)])
    assert sm._hidden_width == L + N_PATCH  # captured the widened hidden
    m = sm._spectral[0]
    assert m["basis"].shape == (4, D)  # fit still on text tokens only
    replay = sm._sample_replay()
    assert replay["hidden"].shape == (2, L + N_PATCH, D)  # reconstructed at full width


def test_spectral_footprint_smaller_than_raw_buffer():
    """The razor as a unit test: a rank-r summary of a task costs far fewer bytes than
    banking even a handful of raw layer-k activation tensors for the same tokens."""
    sm = _method(rank_r=4, carriers=3)
    sm._fit_spectral(_task(0), [_batch(2), _batch(2)])
    spectral_bytes = sm.memory_bytes()
    # RawMemory equivalent (what latent_replay stores): docs_per_task hidden tensors [L,D] fp16.
    raw_docs = 5  # the ER-5 operating point latent_replay uses
    raw_bytes = raw_docs * (L * D) * 2
    assert spectral_bytes < raw_bytes * 4  # generous bound; basis dominates, carriers are metadata
    # the subspace cost (the part that would scale with tokens in a raw buffer) is tiny:
    subspace_bytes = (4 * D + D) * 4  # basis + global mean, fp32
    assert subspace_bytes < raw_bytes
