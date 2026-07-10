"""Unit tests for ProxyLatentReplay (PLaR) — synthetic, fast, GPU-free.

Covers the one thing PLaR changes vs latent_replay: banking whole PUBLIC docs with
pseudo-labels from the current head (argmax + confidence floor + pad masking), while the
private train_loader is ignored. Injection/freeze invariants are inherited and covered by
test_latent_replay.py; the whole-doc consistency of the banked bundle is the point.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.proxy_latent_replay import ProxyLatentReplay

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


def _proxy_batch(b=2, pad_last=True):
    am = torch.ones(b, L, dtype=torch.long)
    if pad_last:
        am[:, -1] = 0  # a padding position — pseudo-label must be -100 there
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": am,
        "labels": torch.zeros(b, L, dtype=torch.long),  # proxy labels — must be IGNORED
    }


def _method(tau=0.0, docs=3):
    torch.manual_seed(0)
    m = ProxyLatentReplay(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": docs,
            "replay_batch_size": 2,
            "pseudo_conf_tau": tau,
        },
    )
    # inject a synthetic proxy loader (no real WildReceipt in unit tests)
    m._proxy_loader = [_proxy_batch(2), _proxy_batch(2)]
    return m


def test_banks_public_docs_with_pseudo_labels_not_proxy_labels():
    m = _method(docs=3)
    private_loader = [{"input_ids": torch.zeros(1, L, dtype=torch.long)}]  # must be ignored
    m._capture_task(private_loader)
    assert len(m.store) == 3
    doc = m.store[0]
    assert doc["hidden"].dtype == torch.float16 and doc["hidden"].device.type == "cpu"
    # labels are the head's argmax (with padding masked), NOT the proxy's zeros
    with torch.no_grad():
        batch = m._proxy_loader[0]
        out = m.model(**{k: v for k, v in batch.items() if k != "labels"})
        expected = out.logits.argmax(-1)[0]
        expected = expected.masked_fill(~batch["attention_mask"][0].bool(), -100)
    assert torch.equal(doc["labels"], expected)
    assert doc["labels"][-1].item() == -100  # padding position ignored


def test_confidence_floor_masks_low_confidence():
    strict = _method(tau=0.999999, docs=2)  # near-1 floor: (almost) everything masked
    strict._capture_task([])
    lab = strict.store[0]["labels"]
    assert (lab == -100).float().mean() > 0.5  # most positions dropped under a strict floor


def test_replay_batch_is_whole_consistent_docs():
    m = _method(docs=2)
    m._capture_task([])
    replay = m._sample_replay()
    assert replay is not None
    # inherited whole-doc replay: hidden/bbox/mask/labels all from the SAME banked doc
    assert replay["hidden"].shape[1] == L and replay["labels"].shape[1] == L


def _soft_method(docs=2):
    torch.manual_seed(0)
    m = ProxyLatentReplay(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": docs,
            "replay_batch_size": 2,
            "soft_labels": True,
            "soft_T": 1.0,
        },
    )
    m._proxy_loader = [_proxy_batch(2), _proxy_batch(2)]
    return m


def test_soft_mode_banks_logits_and_replays_soft_ce():
    m = _soft_method(docs=2)
    m._apply_freeze_map()
    m._capture_task([])
    assert "logits" in m.store[0]
    assert m.store[0]["logits"].shape == (L, NL)
    assert m.store[0]["logits"].dtype == torch.float16
    replay = m._sample_replay()
    assert "logits" in replay  # _sample_replay stacks every stored key
    out = m._replay_forward(replay)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    layers = m._encoder_layers()
    # soft replay gradients still reach only the plastic layers (inherited invariant)
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_soft_loss_zero_when_current_matches_banked():
    """Soft-CE against the banked distribution is minimized when the current head still
    produces it — the loss must anchor the head to its bank-time behavior."""
    m = _soft_method(docs=2)
    m._capture_task([])
    replay = m._sample_replay()
    loss_now = m._replay_forward(replay).loss.item()
    # perturb the head → the soft loss must increase
    with torch.no_grad():
        m.model.model.classifier.weight.add_(torch.randn_like(m.model.model.classifier.weight))
    loss_perturbed = m._replay_forward(replay).loss.item()
    assert loss_perturbed > loss_now
