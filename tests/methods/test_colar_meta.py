"""Unit tests for CoLaR-Meta (Benna-Fusi metaplastic consolidation) — synthetic, GPU-free.

Covers: the consolidator covers only plastic params (frozen trunk untouched), meta_m=1 is
a true no-op control (zero state, weights bit-identical), diffusion relaxes a perturbed
weight back toward the slow anchor without diverging, state_bytes is exact, and the
consolidator is built lazily at task 1 (never during task 0).
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.colar_meta import CoLaRMeta, MetaplasticConsolidator
from doccl.types import TaskInfo

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


def _method(**extra):
    torch.manual_seed(0)
    cfg = {
        "split_layer_k": K,
        "docs_per_task": 2,
        "replay_batch_size": 2,
        "rank_r": 2,
        "epochs": 1,
        "meta_m": 3,
        "meta_eta": 0.1,
        **extra,
    }
    return CoLaRMeta(_Wrapper(), cfg)


def _task(tid):
    return TaskInfo(task_id=tid, task_name=f"t{tid}", label_set=["O", "B-A", "B-B", "B-C"])


def test_consolidator_touches_only_plastic_params():
    m = _method()
    m._apply_freeze_map()
    frozen = [p.clone() for p in m.model.model.layoutlmv3.encoder.layer[0].parameters()]
    cons = MetaplasticConsolidator(m.trainable_parameters(), m=3, eta=0.1)
    for p in cons.params:
        p.data.add_(1.0)  # simulate an optimizer step on the plastic bucket
    cons.step()
    for before, after in zip(
        frozen, m.model.model.layoutlmv3.encoder.layer[0].parameters(), strict=True
    ):
        assert torch.equal(before, after.data)


def test_meta_m_one_is_noop_control():
    m = _method(meta_m=1)
    cons = MetaplasticConsolidator(m.trainable_parameters(), m=1)
    assert cons.state_bytes() == 0
    before = [p.clone() for p in cons.params]
    cons.step()
    for b, p in zip(before, cons.params, strict=True):
        assert torch.equal(b, p.data)


def test_consolidation_relaxes_perturbed_weight_toward_anchor():
    p = nn.Parameter(torch.zeros(4))
    cons = MetaplasticConsolidator([p], m=3, g_base=2.0, eta=0.1)
    p.data.add_(1.0)  # fast perturbation away from the (zero) equilibrated chain
    prev = float(p.data.abs().sum())
    for _ in range(50):
        cons.step()
        cur = float(p.data.abs().sum())
        assert cur <= prev + 1e-6  # relaxes toward the slow anchor, never diverges
        prev = cur
    assert prev < 4.0 * 0.9  # actually moved


def test_state_bytes_matches_expected_param_count():
    m = _method()
    params = m.trainable_parameters()
    cons = MetaplasticConsolidator(params, m=3)
    assert cons.state_bytes() == 2 * sum(p.numel() for p in params) * 4


def test_consolidator_built_at_task_one_not_task_zero():
    m = _method()
    m.train_task(_task(0), [_batch(2, seed=1)])
    assert m._consolidator is None
    assert m.consolidator_state_bytes() == 0
    m.after_task(_task(0), [_batch(2, seed=1)])  # applies the freeze map + banks docs
    m.train_task(_task(1), [_batch(2, seed=2)])
    assert m._consolidator is not None
    n_plastic = len(m.trainable_parameters())
    assert len(m._consolidator.params) == n_plastic
    assert m.consolidator_state_bytes() > 0
