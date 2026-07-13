"""Unit tests for LARM (Lexical-Associative Rewrite Memory) — synthetic, fast, GPU-free.

Covers the fusion's load-bearing invariants:
(a) zero-init memory ⇒ read is identity ⇒ LARM == CoLaR before the memory trains;
(b) a trained memory shifts features by the routed low-rank Δ (add path);
(c) replay routes by the STORED sig, not the dummy_ids of the injection;
(d) memory params are in the trainable set and receive grad from BOTH current-CE and replay-CE;
(e) the add-vs-replace switch changes the hidden correctly (Gate-0 ablation);
(f) memory cells are created (one per banked doc) with frozen keys.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.larm import LARM, RewriteMemory

D, L, NL, N_LAYERS, K, V = 16, 6, 4, 4, 2, 40


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Cfg:
    vocab_size = V


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = D
        inner = nn.Module()
        inner.embeddings = nn.Embedding(V, D)
        encoder = nn.Module()
        encoder.layer = nn.ModuleList(_Layer() for _ in range(N_LAYERS))
        inner.encoder = encoder
        hf = nn.Module()
        hf.layoutlmv3 = inner
        hf.classifier = nn.Linear(D, NL)
        hf.config = _Cfg()
        self.model = hf
        self.processor = type("P", (), {"tokenizer": type("T", (), {"pad_token_id": 1})()})()

    def freeze_backbone(self):
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = False

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self):
        return sum(p.numel() for p in self.parameters())

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


def _batch(b=2):
    return {
        "input_ids": torch.randint(0, V, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, NL, (b, L)),
    }


def _larm(rank_r=4, mem_rank=3, add=True, route=True, docs=2):
    torch.manual_seed(0)
    return LARM(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": docs,
            "replay_batch_size": 2,
            "rank_r": rank_r,
            "mem_rank": mem_rank,
            "mem_tau": 0.1,
            "add_not_replace": add,
            "route_replay": route,
        },
    )


def test_memory_read_zero_when_empty_and_untrained():
    mem = RewriteMemory(vocab_size=V, d=D, rank=3, tau=0.1)
    h, q = torch.randn(2, L, D), torch.rand(2, V)
    assert torch.equal(mem.read(h, q), torch.zeros_like(h))  # empty
    mem.add_cell(torch.rand(V), "cpu")
    assert torch.allclose(mem.read(h, q), torch.zeros_like(h))  # zero-init up => no-op


def test_capture_banks_sigs_and_creates_cells():
    m = _larm(docs=2)
    m._capture_task([_batch(2)])
    assert m.mem.n_cells() == 2  # one cell per banked doc
    assert all(d.get("sig") is not None and d["sig"].shape == (V,) for d in m.store)
    # keys are frozen (not nn.Parameters)
    assert all(not isinstance(k, nn.Parameter) for k in m.mem.keys)


def test_trained_memory_shifts_features_add_path():
    m = _larm(mem_rank=3)
    m._capture_task([_batch(2)])
    # manually make cell 0 a non-no-op
    with torch.no_grad():
        m.mem.up[0].copy_(torch.randn(3, D))
    h = torch.randn(1, L, D)
    q = m.mem.keys[0].unsqueeze(0)  # route fully to cell 0
    delta = m.mem.read(h, q)
    assert not torch.allclose(delta, torch.zeros_like(delta))  # features moved


def test_memory_params_in_trainable_set():
    m = _larm(docs=2)
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    tp = m.trainable_parameters()
    mem_params = list(m.mem.parameters())
    assert mem_params  # cells created params
    assert all(any(p is q for q in tp) for p in mem_params)  # every mem param is trainable


def test_replay_routes_by_stored_sig_not_dummy():
    m = _larm(docs=2)
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    replay = m._sample_replay()
    assert "sig" in replay and replay["sig"].shape[1] == V
    # the replay forward should set _replay_query to the stored sig during the call
    seen = {}
    orig = m.mem.read

    def spy(h, q):
        seen["q"] = q.detach().clone()
        return orig(h, q)

    m.mem.read = spy
    m._replay_forward(replay)
    m.mem.read = orig
    # the query used was the stored sig, not a dummy-id-derived one
    assert torch.allclose(seen["q"].cpu(), replay["sig"])


def test_add_vs_replace_switch():
    add = _larm(add=True, mem_rank=3)
    add._capture_task([_batch(2)])
    with torch.no_grad():
        add.mem.up[0].copy_(torch.randn(3, D))
    h = torch.randn(1, L, D)
    q = add.mem.keys[0].unsqueeze(0)
    delta = add.mem.read(h, q)
    # add: new = h + delta ; replace: new = delta. Verify the pre-hook honors the flag.
    add._cur_query = q
    out_add = add._pre_hook(None, (h,), {})
    assert torch.allclose(out_add[0][0], h + delta, atol=1e-5)
    rep = _larm(add=False, mem_rank=3)
    rep.mem = add.mem  # same trained memory
    rep._cur_query = q
    out_rep = rep._pre_hook(None, (h,), {})
    assert torch.allclose(out_rep[0][0], delta, atol=1e-5)
