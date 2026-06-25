"""Unit tests for DocMERGE head-merging — synthetic tensors only (fast, GPU-free).

These pin the merge arithmetic invariants the method's positive-BWT claim rests on:

  * merging identical deltas reproduces that delta (plain & fisher & ties),
  * fisher-merge with equal weights == plain average,
  * fisher-merge concentrates on the high-importance task,
  * ties zeroes a low-magnitude sign-conflicting coordinate,
  * differently-sized (CIL-grown) deltas merge by zero-padding the narrower,
  * the lexical-memory address still separates disjoint vocabularies (reused HRP
    invariant — the memory half of the method).

The full lifecycle on a real LayoutLMv3 (merge load-back + diag.json/routing.json) is
covered by ``tests/methods/test_all_methods_e2e.py`` once ``doc_merge`` is registered.
"""

from __future__ import annotations

import torch

from doccl.methods.head_merge import MERGE_RULES, merge_head_deltas


def test_merge_identical_deltas_is_identity():
    """Merging T copies of the same delta returns that delta.

    Holds unconditionally for plain/fisher. For ties it holds only when nothing is
    trimmed (density=1.0) — at density<1 ties intentionally keeps only each task's
    top-|delta| coordinates, so identical deltas are still trimmed (that is the point
    of ties, and is exercised by the dedicated density test).
    """
    d = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.0, -1.5]])
    for rule in ("plain", "fisher"):
        merged = merge_head_deltas([d, d, d], rule=rule, weights=[1.0, 1.0, 1.0])
        assert torch.allclose(merged, d, atol=1e-6), f"{rule} not identity on identical deltas"
    ties_full = merge_head_deltas([d, d, d], rule="ties", density=1.0)
    assert torch.allclose(ties_full, d, atol=1e-6), "ties(density=1) must be identity"
    assert MERGE_RULES == ("plain", "ties", "fisher")  # guard the public rule set


def test_fisher_equal_weights_equals_plain_average():
    d0 = torch.tensor([2.0, 0.0, -4.0])
    d1 = torch.tensor([0.0, 6.0, 2.0])
    plain = merge_head_deltas([d0, d1], rule="plain")
    fisher = merge_head_deltas([d0, d1], rule="fisher", weights=[1.0, 1.0])
    assert torch.allclose(plain, fisher, atol=1e-6)
    assert torch.allclose(plain, (d0 + d1) / 2, atol=1e-6)


def test_fisher_weights_concentrate_on_important_task():
    """A dominant Fisher weight pulls the merge toward that task's delta."""
    d0 = torch.tensor([10.0, 10.0])
    d1 = torch.tensor([0.0, 0.0])
    merged = merge_head_deltas([d0, d1], rule="fisher", weights=[9.0, 1.0])
    # 0.9 * d0 + 0.1 * d1
    assert torch.allclose(merged, torch.tensor([9.0, 9.0]), atol=1e-6)


def test_fisher_zero_total_falls_back_to_plain():
    d0 = torch.tensor([2.0, 4.0])
    d1 = torch.tensor([6.0, 8.0])
    merged = merge_head_deltas([d0, d1], rule="fisher", weights=[0.0, 0.0])
    assert torch.allclose(merged, (d0 + d1) / 2, atol=1e-6)


def test_ties_zeroes_sign_conflicting_low_magnitude_coordinate():
    """A coordinate where a small minority delta disagrees in sign is elected by the
    dominant sign; with density=1 the disagreeing entry is excluded from the mean."""
    # Coord 0: two strong +, one weak − → elected +, mean of the two + only.
    d0 = torch.tensor([4.0, 0.0])
    d1 = torch.tensor([4.0, 0.0])
    d2 = torch.tensor([-1.0, 0.0])
    merged = merge_head_deltas([d0, d1, d2], rule="ties", density=1.0)
    assert merged[0].item() == 4.0  # mean of the two agreeing (+4, +4)
    assert merged[1].item() == 0.0


def test_ties_density_trims_small_coordinates_per_task():
    """With density<1 each task keeps only its top-|delta| coordinates."""
    # Task keeps top-1 of 3; the two small coords are trimmed to 0 for that task.
    d0 = torch.tensor([10.0, 0.1, 0.1])
    d1 = torch.tensor([10.0, 0.1, 0.1])
    merged = merge_head_deltas([d0, d1], rule="ties", density=0.34)  # keep 1 of 3
    assert merged[0].item() == 10.0
    assert merged[1].item() == 0.0 and merged[2].item() == 0.0


def test_merge_pads_differently_sized_cil_grown_deltas():
    """A task whose head was narrower contributes zero delta in the new rows."""
    narrow = torch.tensor([1.0, 1.0])  # task 0 head had 2 logit rows
    wide = torch.tensor([2.0, 2.0, 4.0, 4.0])  # task 1 head grew to 4 rows
    merged = merge_head_deltas([narrow, wide], rule="plain")
    # rows 0-1: mean(1,2)=1.5 ; rows 2-3: mean(0,4)=2.0 (narrow padded with 0)
    assert torch.allclose(merged, torch.tensor([1.5, 1.5, 2.0, 2.0]), atol=1e-6)


def test_merge_pads_2d_weight_deltas():
    """Weight (2-D) deltas of differing label-count merge by zero-padding new rows."""
    narrow = torch.ones(2, 3)  # 2 labels x hidden 3
    wide = torch.full((4, 3), 2.0)  # grew to 4 labels
    merged = merge_head_deltas([narrow, wide], rule="plain")
    assert merged.shape == (4, 3)
    assert torch.allclose(merged[:2], torch.full((2, 3), 1.5), atol=1e-6)  # mean(1,2)
    assert torch.allclose(merged[2:], torch.full((2, 3), 1.0), atol=1e-6)  # mean(0,2)


def test_lexical_memory_separates_disjoint_vocabularies():
    """The memory half: sparse-OCR routing sends disjoint-vocab docs to their own block.

    Mirrors the HRP invariant — confirms the reused HybridPromptPool addressing is intact
    in the DocMERGE context.
    """
    from doccl.methods.hybrid_routed_prompt import HybridPromptPool, sparse_doc_vectors

    vocab, dim = 50, 16
    pool = HybridPromptPool(
        n_tasks=2, slots_per_task=1, prompt_length=2, hidden_dim=dim, vocab_size=vocab
    )
    ids_t0 = torch.randint(0, 10, (6, 8))
    ids_t1 = torch.randint(20, 30, (6, 8))
    pool.accumulate_signature(0, ids_t0)
    pool.accumulate_signature(1, ids_t1)
    for true_task, ids in [(0, ids_t0), (1, ids_t1)]:
        sparse_q = sparse_doc_vectors(ids, vocab, pool.idf)
        idx, _ = pool.route(torch.zeros(ids.shape[0], dim), sparse_q, top_k=1, router="sparse")
        assert (pool.slot_to_block(idx[:, 0]) == true_task).all()
