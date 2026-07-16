"""Unit tests for CoLaR-kNN (read-side memory) and LatentDatastore — synthetic, GPU-free.

Covers: the lambda=0 control is identical to plain CoLaR eval, the datastore view grows
with the store and rebuilds on invalidate, the kNN readout is a valid probability simplex,
lambda=1 is a pure memory head (parametric logits ignored), and class_balance reweights an
O-dominated datastore.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.colar import CoLaR
from doccl.methods.colar_knn import CoLaRKNN
from doccl.methods.latent_datastore import LatentDatastore

D, L, NL, N_LAYERS, K = 16, 6, 4, 4, 2
ID_TO_LABEL = {0: "O", 1: "B-A", 2: "B-B", 3: "B-C"}


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
        self.id_to_label = ID_TO_LABEL
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


def _batch(b=2, seed=None, label_value=None):
    if seed is not None:
        torch.manual_seed(seed)
    labels = (
        torch.full((b, L), label_value, dtype=torch.long)
        if label_value is not None
        else torch.randint(0, NL, (b, L))
    )
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": labels,
    }


def _method(cls, knn_lambda=0.3, rank=2, docs=2, **extra):
    torch.manual_seed(0)
    cfg = {"split_layer_k": K, "docs_per_task": docs, "replay_batch_size": 2, "rank_r": rank}
    if cls is CoLaRKNN:
        cfg.update({"knn_lambda": knn_lambda, "knn_top_k": 3, "knn_tau": 0.1, **extra})
    return cls(_Wrapper(), cfg)


def test_lambda_zero_matches_plain_colar_eval():
    base = _method(CoLaR)
    knn = _method(CoLaRKNN, knn_lambda=0.0)
    for m in (base, knn):
        m._capture_task([_batch(2, seed=1)])
    loaders = {0: [_batch(2, seed=2)]}
    r_base = base.evaluate(loaders)[0]
    r_knn = knn.evaluate(loaders)[0]
    assert r_base.f1 == r_knn.f1
    assert r_base.precision == r_knn.precision
    assert r_base.n_samples == r_knn.n_samples


def test_datastore_rebuilds_after_task_and_grows():
    m = _method(CoLaRKNN)
    m._capture_task([_batch(2, seed=1)])
    m.datastore.build(m.store)
    n1 = len(m.datastore)
    assert n1 > 0
    m._capture_task([_batch(2, seed=2)])
    m.datastore.invalidate()
    m.datastore.build(m.store)
    assert len(m.datastore) > n1


def test_knn_readout_is_probability_simplex():
    m = _method(CoLaRKNN)
    m._capture_task([_batch(2, seed=1)])
    m.datastore.build(m.store)
    p = m.datastore.knn_label_dist(torch.randn(3, L, D), top_k=3, tau=0.1, n_labels=NL)
    assert p.shape == (3, L, NL)
    assert torch.allclose(p.sum(-1), torch.ones(3, L), atol=1e-4)
    assert (p >= 0).all()


def test_pure_memory_head_lambda_one_ignores_classifier_logits():
    m = _method(CoLaRKNN, knn_lambda=1.0)
    m._capture_task([_batch(2, seed=1, label_value=2)])  # datastore votes only class 2
    eval_batch = _batch(2, seed=3)
    with torch.no_grad():  # make the parametric head prefer a DIFFERENT class everywhere
        m.model.model.classifier.weight.zero_()
        m.model.model.classifier.bias.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
    m.evaluate({0: [eval_batch]})  # exercises the blended path end-to-end
    query = torch.randn(1, L, D)
    p = m.datastore.knn_label_dist(query, top_k=3, tau=0.1, n_labels=NL)
    assert (p.argmax(-1) == 2).all()


def test_class_balance_reweights_dominant_label():
    h = torch.ones(10, D)  # identical features -> cosine ties, top_k covers all tokens
    doc = {
        "hidden": h,
        "attention_mask": torch.ones(10, dtype=torch.long),
        "labels": torch.tensor([0] * 9 + [1]),
        "bbox": torch.zeros(10, 4, dtype=torch.long),
    }
    query = torch.ones(1, D)
    plain, balanced = LatentDatastore(), LatentDatastore()
    for ds in (plain, balanced):
        ds.build([doc])
    p_plain = plain.knn_label_dist(query, top_k=10, tau=None, n_labels=NL)
    p_bal = balanced.knn_label_dist(query, top_k=10, tau=None, n_labels=NL, class_balance=True)
    assert p_plain[0, 0] > 0.8  # O-dominated vote
    assert p_bal[0, 1] > p_plain[0, 1]  # minority label gains mass under balancing
    assert torch.allclose(p_bal[0, 0], p_bal[0, 1])  # 9/1 split fully equalized here
