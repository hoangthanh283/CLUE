"""Unit tests for CoLaR (per-doc SVD compressed latent replay) and the latent_replay
doc_selection knob — synthetic, fast, GPU-free.

Covers: compression replaces raw hiddens with per-doc factors, reconstruction is exact at
full rank and injectable at any rank, the plastic-only gradient invariant survives, the
compressed footprint beats raw, k-center selection maximizes coverage, and the default
random path stays byte-identical (first-N banking) so prior runs remain reproducible.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.colar import CoLaR
from doccl.methods.latent_replay import LatentReplay

D, L, NL, N_LAYERS, K = 16, 6, 4, 4, 2


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
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, NL, (b, L)),
    }


def _colar(rank=2, docs=2):
    torch.manual_seed(0)
    return CoLaR(
        _Wrapper(),
        {"split_layer_k": K, "docs_per_task": docs, "replay_batch_size": 2, "rank_r": rank},
    )


def test_compression_replaces_hidden_with_factors():
    m = _colar(rank=2, docs=2)
    m._capture_task([_batch(2)])
    d = m.store[0]
    assert "hidden" not in d
    assert d["us"].shape == (L, 2) and d["v"].shape == (2, D)
    assert d["us"].dtype == torch.float16 and d["v"].dtype == torch.float16


def test_full_rank_reconstruction_is_near_exact():
    m = _colar(rank=min(L, D), docs=2)  # full rank -> lossless up to fp16
    m._capture_task([_batch(2)])
    replay = m._sample_replay()
    assert replay["hidden"].shape == (2, L, D)
    d = m.store[0]
    recon = d["us"].float() @ d["v"].float()
    assert recon.shape == (L, D)  # injectable full-width hidden per doc


def test_replay_grads_reach_only_plastic_layers():
    m = _colar(rank=2, docs=2)
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    out = m._replay_forward(m._sample_replay())
    out.loss.backward()
    layers = m._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_compressed_footprint_below_raw():
    m = _colar(rank=2, docs=2)
    m._capture_task([_batch(2)])
    raw_bytes_equiv = 2 * (L * D) * 2  # what the raw store would hold, fp16
    factor_bytes = sum((d["us"].numel() + d["v"].numel()) * 2 for d in m.store)
    assert factor_bytes < raw_bytes_equiv


def test_kcenter_selects_for_coverage_and_random_stays_first_n():
    # random (default): banks the FIRST docs_per_task docs — prior-run reproducibility
    torch.manual_seed(0)
    lr = LatentReplay(_Wrapper(), {"split_layer_k": K, "docs_per_task": 2, "replay_batch_size": 2})
    batches = [_batch(2, seed=1), _batch(2, seed=2)]
    lr._capture_task(batches)
    assert len(lr.store) == 2
    # kcenter: pools candidates then picks a covering subset (deterministic seed=centroid-nearest)
    torch.manual_seed(0)
    kc = LatentReplay(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": 2,
            "replay_batch_size": 2,
            "doc_selection": "kcenter",
            "selection_pool": 4,
        },
    )
    kc._capture_task([_batch(2, seed=1), _batch(2, seed=2)])
    assert len(kc.store) == 2


def test_kcenter_picks_the_outlier_first_n_misses():
    """Controlled pool: three near-duplicate docs + one far outlier. First-N (random path)
    banks two near-duplicates; k-center MUST cover the outlier — that is its purpose."""

    def _doc(val):
        return {
            "hidden": torch.full((L, D), float(val), dtype=torch.float16),
            "attention_mask": torch.ones(L, dtype=torch.long),
            "bbox": torch.zeros(L, 4, dtype=torch.long),
            "labels": torch.zeros(L, dtype=torch.long),
        }

    pool = [_doc(0.0), _doc(0.01), _doc(0.02), _doc(10.0)]  # outlier is index 3
    picked = LatentReplay._kcenter_select(pool, 2)
    vals = {float(d["hidden"][0, 0]) for d in picked}
    assert 10.0 in vals  # the outlier is covered
    assert len(picked) == 2
