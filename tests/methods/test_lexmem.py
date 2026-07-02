"""Unit tests for LexMem internals (synthetic tensors, fast, GPU-free).

Covers the LexicalMemoryHead numerics: exact zero-init no-op, top-k retrieval
shapes, gradient masking of unselected slot rows, TF-IDF slot selection against
a background corpus, access counting under an attention mask, key init from
feature samples, and CIL label-column growth. Full lifecycle on a real
LayoutLMv3 is in tests/methods/test_all_methods_e2e.py once registered.
"""

from __future__ import annotations

import torch

from doccl.methods.lexmem_memory import LexicalMemoryHead


def _head(n_slots=8, d=8, n_labels=5, top_k=4, temp=0.05) -> LexicalMemoryHead:
    torch.manual_seed(0)
    return LexicalMemoryHead(n_slots, d, n_labels, top_k=top_k, temp=temp)


def test_delta_zero_init_is_exact_noop():
    mem = _head()
    feats = torch.randn(2, 4, 8)
    out = mem.delta(feats)
    assert out.shape == (2, 4, 5)
    assert torch.all(out == 0), "zero-init values must give an exact zero delta"


def test_delta_shapes_and_no_feat_grad():
    mem = _head()
    with torch.no_grad():
        mem.values.uniform_(-0.1, 0.1)
    feats = torch.randn(3, 7, 8, requires_grad=True)
    out = mem.delta(feats)
    assert out.shape == (3, 7, 5)
    out.sum().backward()
    # Retrieval detaches the query: the frozen backbone must receive no gradient.
    assert feats.grad is None or torch.all(feats.grad == 0)


def test_grad_mask_blocks_unselected_rows():
    mem = _head(top_k=8)  # top_k = n_slots -> every slot is accessed
    counts = torch.ones(8)
    mem.note_background_batch(torch.zeros(8))  # 1 bg batch touching nothing
    mem.select_topt(counts, t=2, task_id=1, mode="tf")
    selected = mem.grad_mask.nonzero().flatten().tolist()
    assert len(selected) == 2
    feats = torch.randn(2, 4, 8)
    mem.delta(feats).sum().backward()
    grad = mem.values.grad
    unselected = [i for i in range(8) if i not in selected]
    assert torch.all(grad[unselected] == 0), "masked rows must get exactly zero grad"
    assert grad[selected].abs().sum() > 0, "selected rows must receive gradient"


def test_sgd_step_only_updates_selected_rows():
    mem = _head(top_k=8)
    mem.select_topt(torch.ones(8), t=3, task_id=0, mode="tf")
    selected = set(mem.grad_mask.nonzero().flatten().tolist())
    opt = torch.optim.SGD([mem.values], lr=1.0)
    labels = torch.randint(0, 5, (2, 4))
    logits = mem.delta(torch.randn(2, 4, 8))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
    loss.backward()
    opt.step()
    for i in range(8):
        if i in selected:
            continue
        assert torch.all(mem.values[i] == 0), f"unselected row {i} moved"


def test_tfidf_prefers_task_specific_slots():
    mem = _head(top_k=8)
    # Background: 3 batches, all hitting slot 0 and 1 (generic/structural slots).
    bg = torch.zeros(8)
    bg[0], bg[1] = 10.0, 10.0
    for _ in range(3):
        mem.note_background_batch(bg)
    # New task: hammers shared slot 0, lightly touches fresh slots 3, 4.
    counts = torch.zeros(8)
    counts[0], counts[3], counts[4] = 100.0, 10.0, 8.0
    idx = mem.select_topt(counts, t=2, task_id=1, mode="tfidf")
    assert set(idx.tolist()) == {3, 4}, f"TF-IDF must pick fresh slots, got {idx.tolist()}"


def test_tf_mode_prefers_raw_frequency():
    mem = _head(top_k=8)
    mem.note_background_batch(torch.zeros(8))
    counts = torch.zeros(8)
    counts[0], counts[3] = 100.0, 10.0
    idx = mem.select_topt(counts, t=1, task_id=1, mode="tf")
    assert idx.tolist() == [0]


def test_random_mode_selects_among_accessed():
    mem = _head(top_k=8)
    counts = torch.zeros(8)
    counts[2], counts[5], counts[6] = 1.0, 1.0, 1.0
    idx = mem.select_topt(counts, t=2, task_id=1, mode="random")
    assert set(idx.tolist()) <= {2, 5, 6}
    assert len(idx) == 2


def test_select_sets_owner_first_claim_only():
    mem = _head(top_k=8)
    mem.select_topt(torch.tensor([9.0, 9, 0, 0, 0, 0, 0, 0]), t=2, task_id=0, mode="tf")
    assert mem.slot_owner[0] == 0 and mem.slot_owner[1] == 0
    # Task 1 re-selects slot 0 -> ownership must NOT be reassigned.
    mem.select_topt(torch.tensor([9.0, 0, 9, 0, 0, 0, 0, 0]), t=2, task_id=1, mode="tf")
    assert mem.slot_owner[0] == 0, "first owner must be preserved"
    assert mem.slot_owner[2] == 1
    # grad_mask is per-task: exactly the current selection is trainable.
    assert set(mem.grad_mask.nonzero().flatten().tolist()) == {0, 2}


def test_counting_respects_attention_mask():
    mem = _head(n_slots=8, d=8, top_k=1, temp=1.0)
    with torch.no_grad():
        mem.keys.copy_(torch.eye(8))  # token i routes to slot i (top-1)
    feats = torch.eye(8)[[0, 1, 2, 3]].reshape(1, 4, 8)
    mem._count_mask = torch.tensor([[1, 1, 0, 0]])
    mem.start_counting()
    mem.delta(feats)
    counts = mem.stop_counting()
    assert counts[0] == 1 and counts[1] == 1
    assert counts[2] == 0 and counts[3] == 0, "pad positions must not be counted"


def test_background_accumulates_batch_hits():
    mem = _head()
    c1 = torch.zeros(8)
    c1[0] = 5.0
    c2 = torch.zeros(8)
    c2[0], c2[1] = 1.0, 1.0
    mem.note_background_batch(c1)
    mem.note_background_batch(c2)
    assert mem.n_bg_batches == 2
    assert mem.bg_hits[0] == 2 and mem.bg_hits[1] == 1 and mem.bg_hits[2] == 0


def test_init_keys_sample_and_pad_paths():
    mem = _head(n_slots=8, d=8)
    # Fewer feats than slots: use all, remaining rows stay (normalized) random.
    feats = torch.randn(3, 8)
    mem.init_keys(feats, mode="sample")
    norms = mem.keys.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), "all keys must be unit norm"
    got = torch.nn.functional.normalize(feats.float(), dim=-1)
    assert torch.allclose(mem.keys[:3], got, atol=1e-5)
    # More feats than slots: subsample down to n_slots.
    mem2 = _head(n_slots=8, d=8)
    mem2.init_keys(torch.randn(50, 8), mode="sample")
    assert torch.allclose(mem2.keys.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_init_keys_kmeans_runs_and_normalizes():
    torch.manual_seed(1)
    mem = _head(n_slots=4, d=8)
    feats = torch.randn(64, 8)
    mem.init_keys(feats, mode="kmeans", iters=3)
    assert torch.allclose(mem.keys.norm(dim=-1), torch.ones(4), atol=1e-4)


def test_expand_labels_preserves_and_masks_after():
    mem = _head(top_k=8, n_labels=5)
    mem.select_topt(torch.ones(8), t=2, task_id=0, mode="tf")
    with torch.no_grad():
        mem.values[:, :].uniform_(-0.1, 0.1)
    old = mem.values.data.clone()
    mem.expand_labels(9)
    assert mem.values.shape == (8, 9)
    assert torch.allclose(mem.values.data[:, :5], old)
    assert torch.all(mem.values.data[:, 5:] == 0)
    # Grad hook must survive the Parameter replacement.
    selected = set(mem.grad_mask.nonzero().flatten().tolist())
    mem.delta(torch.randn(2, 4, 8)).sum().backward()
    unselected = [i for i in range(8) if i not in selected]
    assert torch.all(mem.values.grad[unselected] == 0)
