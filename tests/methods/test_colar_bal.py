"""CoLaR-Bal (CoLaR + soft-target replay) — synthetic, fast, GPU-free.

Covers: soft-label banking compresses logits per-doc (lg_us/lg_v), replay dict carries
reconstructed logits, soft-CE replay backprops to plastic layers only, and soft_labels=False
banks no logits (byte-identical-to-CoLaR path).
"""

from __future__ import annotations

import torch

from doccl.methods.colar_bal import CoLaRBal

# Reuse the CoLaR test's synthetic wrapper + batch (same encoder/head/inject hook).
from tests.methods.test_colar import NL, K, L, _batch, _Wrapper


def _colar_bal(rank=2, docs=2, soft=True):
    torch.manual_seed(0)
    return CoLaRBal(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": docs,
            "replay_batch_size": 2,
            "rank_r": rank,
            "soft_labels": soft,
        },
    )


def test_soft_banking_compresses_logits_per_doc():
    m = _colar_bal(rank=2, docs=2, soft=True)
    m._capture_task([_batch(2)])
    d = m.store[0]
    assert "lg_us" in d and "lg_v" in d
    assert d["lg_us"].shape == (L, 2) and d["lg_v"].shape == (2, NL)
    assert d["lg_us"].dtype == torch.float16 and d["lg_v"].dtype == torch.float16


def test_replay_dict_carries_reconstructed_logits():
    m = _colar_bal(rank=min(L, NL), docs=2, soft=True)
    replay = (m._capture_task([_batch(2)]), m._sample_replay())[1]
    assert "logits" in replay
    assert replay["logits"].shape == (2, L, NL)


def test_soft_replay_grads_reach_only_plastic_layers():
    m = _colar_bal(rank=2, docs=2, soft=True)
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    out = m._replay_forward(m._sample_replay())
    out.loss.backward()
    layers = m._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_soft_off_banks_no_logits_and_falls_back_to_hard_ce():
    m = _colar_bal(rank=2, docs=2, soft=False)
    m._capture_task([_batch(2)])
    assert "lg_us" not in m.store[0]
    replay = m._sample_replay()
    assert "logits" not in replay
    # hard fallback: inherited CE path still produces a finite, backprop-able loss
    out = m._replay_forward(replay)
    assert torch.isfinite(out.loss)
