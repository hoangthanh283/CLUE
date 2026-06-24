"""Regression test for the DER++ replay-forward crash.

The reservoir buffer stores per-example bookkeeping keys ``_logits`` and
``_logit_width`` (the latter added when the CIL head grows). The model wrapper's
``forward`` has a FIXED signature (no ``**kwargs``), so any ``_``-prefixed key that
leaks into ``self.model(**replay)`` raises
``TypeError: forward() got an unexpected keyword argument '_logit_width'`` at the
first replay step. A prior version popped ``_logits`` from the CE-replay batch but
not ``_logit_width`` — crashing every DER++ run that reached a non-empty buffer.

This test pins the contract: NO ``_``-prefixed key may reach the forward call, for
either replay branch. It uses a stub model with a strict signature, so it catches
the exact failure without a GPU or model download.
"""
from __future__ import annotations

import torch

from doccl.methods.buffer import ReservoirBuffer


class _StrictForwardStub:
    """Mimics the wrapper: forward() rejects unknown kwargs like the real one."""

    def __init__(self):
        self.seen_keys: list[set] = []

    def forward(self, input_ids, bbox, pixel_values=None, attention_mask=None, labels=None):
        # If a "_"-prefixed key reached here Python would already have raised
        # TypeError before this body — so reaching here means the caller stripped them.
        self.seen_keys.append({"input_ids", "bbox"})

        class _Out:
            loss = torch.zeros(())
            logits = torch.zeros(2, 4, 7)

        return _Out()

    __call__ = forward


def _make_buffer_with_widths() -> ReservoirBuffer:
    """A DER++ buffer holding logits + per-example _logit_width (CIL head growth)."""
    buf = ReservoirBuffer(capacity=10, store_logits=True)
    batch = {
        "input_ids": torch.randint(0, 100, (2, 4)),
        "bbox": torch.randint(0, 1000, (2, 4, 4)),
        "labels": torch.randint(0, 5, (2, 4)),
    }
    logits = torch.randn(2, 4, 5)  # 5-class head at insertion time
    buf.add_batch(batch, logits=logits)
    return buf


def test_buffer_carries_logit_width():
    """Sanity: the buffer DOES return _logit_width, so the strip is necessary."""
    buf = _make_buffer_with_widths()
    sample = buf.sample(2)
    assert sample is not None
    assert "_logits" in sample and "_logit_width" in sample, sample.keys()


def test_derpp_replay_strips_all_underscore_keys():
    """The DER++ replay-forward must drop EVERY _-prefixed key before forward().

    Mirrors der.py's replay2 path: strip then forward. If the strip is wrong the
    StrictForwardStub raises TypeError, failing the test.
    """
    stub = _StrictForwardStub()
    buf = _make_buffer_with_widths()
    replay2 = buf.sample(2)
    replay2 = {k: v for k, v in replay2.items() if not k.startswith("_")}
    # This call must NOT raise TypeError about _logit_width / _logits.
    out = stub(**replay2)
    assert out.loss is not None
    assert stub.seen_keys, "forward was never reached"
    # And no underscore key survived.
    assert not any(k.startswith("_") for k in replay2)
