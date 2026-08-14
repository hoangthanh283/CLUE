"""Unit tests for the Hybrid-Routed Prompt pool (HRP) — router + sparse signature.

These run on synthetic tensors only (no model, no network), so they are fast and
GPU-free. They pin the routing invariants that make the method's claim falsifiable:

  * shapes of ``route`` / ``gather_prompts`` are correct,
  * RRF of two identical rankings reproduces that ranking,
  * the dense path matches the L2P cosine-top-k selection (dense HRP == L2P),
  * **sparse routing on two disjoint token vocabularies routes each document to its
    own task's block with hit-rate 1.0** — the core "sparse carries task identity"
    invariant the feasibility study is built on.

The full lifecycle on a real LayoutLMv3 (and the routing.json emission) is covered by
``tests/methods/test_all_methods_e2e.py`` once ``hrp`` is registered there.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from doccl.methods.hybrid_routed_prompt import HybridPromptPool, sparse_doc_vectors
from doccl.methods.prompt_base import PromptPool

D = 16
V = 50


def _pool(n_tasks=3, slots_per_task=2, prompt_length=4):
    torch.manual_seed(0)
    return HybridPromptPool(
        n_tasks=n_tasks,
        slots_per_task=slots_per_task,
        prompt_length=prompt_length,
        hidden_dim=D,
        vocab_size=V,
        rrf_k=60,
    )


def test_route_and_gather_shapes():
    pool = _pool()
    B, top_k = 5, 2
    dense_q = torch.randn(B, D)
    sparse_q = F.normalize(torch.rand(B, V), dim=-1)
    idx, key_pull = pool.route(dense_q, sparse_q, top_k=top_k, router="hybrid")
    assert idx.shape == (B, top_k)
    assert key_pull.dim() == 0 and torch.isfinite(key_pull)
    prompts = pool.gather_prompts(idx)
    assert prompts.shape == (B, top_k * pool.prompt_length, D)


def test_rrf_of_identical_rankings_preserves_order():
    """Fusing a score matrix with itself must rank slots in the same order."""
    pool = _pool()
    scores = torch.tensor([[0.1, 0.9, 0.4, 0.7, 0.2, 0.3]])  # N = 6 = 3*2
    fused = pool._rrf(scores, scores, k=60)
    assert torch.equal(fused.argsort(descending=True), scores.argsort(descending=True))


def test_dense_router_matches_l2p_topk():
    """router='dense' must select the same slots as L2P's cosine top-k on shared keys."""
    pool = _pool(n_tasks=5, slots_per_task=1)  # N = 5, like an L2P pool of 5
    ref = PromptPool(n_prompts=pool.n_prompts, prompt_length=pool.prompt_length, hidden_dim=D)
    with torch.no_grad():
        ref.keys.copy_(pool.keys)
        ref.prompts.copy_(pool.prompts)
    dense_q = torch.randn(4, D)
    sparse_q = torch.zeros(4, V)  # irrelevant for dense routing
    idx, _ = pool.route(dense_q, sparse_q, top_k=3, router="dense")
    _, l2p_topk_idx = (F.normalize(dense_q, dim=-1) @ F.normalize(ref.keys, dim=-1).T).topk(
        3, dim=-1
    )
    assert torch.equal(idx.sort(dim=-1).values, l2p_topk_idx.sort(dim=-1).values)


def test_sparse_routing_separates_disjoint_vocabularies():
    """Two tasks with disjoint token vocabularies must route to their own block.

    Build a 2-task pool, give each block a signature over a disjoint token range,
    then route documents drawn from each range. Sparse routing must hit the right
    block every time (hit-rate 1.0) — this is the signal HRP is built to exploit.
    """
    pool = _pool(n_tasks=2, slots_per_task=1)
    # Task 0 owns tokens [0, 10); task 1 owns tokens [20, 30).
    ids_t0 = torch.randint(0, 10, (6, 8))
    ids_t1 = torch.randint(20, 30, (6, 8))
    pool.accumulate_signature(0, ids_t0)
    pool.accumulate_signature(1, ids_t1)

    for true_task, ids in [(0, ids_t0), (1, ids_t1)]:
        sparse_q = sparse_doc_vectors(ids, V, pool.idf)
        dummy_dense = torch.zeros(ids.shape[0], D)
        idx, _ = pool.route(dummy_dense, sparse_q, top_k=1, router="sparse")
        routed_block = pool.slot_to_block(idx[:, 0])
        assert (
            routed_block == true_task
        ).all(), f"sparse routing missed task {true_task}: {routed_block.tolist()}"


def test_sparse_doc_vectors_are_normalized_and_clamp_oov():
    """sparse_doc_vectors L2-normalises rows and never indexes past vocab_size."""
    ids = torch.tensor([[0, 1, 1, 2], [V + 5, V + 9, 3, 3]])  # second row has OOV ids
    vecs = sparse_doc_vectors(ids, V)
    assert vecs.shape == (2, V)
    norms = vecs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # token 1 appears twice in row 0 → its weight is the largest there.
    assert vecs[0].argmax().item() == 1


def test_sparse_doc_vectors_ignore_padding_and_special_tokens():
    ids = torch.tensor([[0, 5, 1, 7, 7, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 1]])
    vecs = sparse_doc_vectors(
        ids,
        V,
        attention_mask=attention_mask,
        ignored_token_ids=(0, 1, 2),
    )
    assert vecs[0, 5] == 1.0
    assert vecs[0, [0, 1, 2, 7]].abs().sum() == 0


def test_idf_none_before_any_task_then_populated():
    pool = _pool()
    assert pool.idf is None
    pool.accumulate_signature(0, torch.randint(0, 10, (4, 8)))
    idf = pool.idf
    assert idf is not None and idf.shape == (V,) and torch.isfinite(idf).all()


def test_block_slots_for_tasks_maps_to_own_block():
    """Replayed examples must map to their OWN task's block, not the active one."""
    from doccl.methods.hybrid_routed_prompt import HybridRoutedPrompt

    m = HybridRoutedPrompt.__new__(HybridRoutedPrompt)
    m.prompt_pool = _pool(n_tasks=4, slots_per_task=2)
    # tasks [0, 2, 3] → blocks [0-1, 4-5, 6-7]
    task_ids = torch.tensor([0, 2, 3])
    slots = m._block_slots_for_tasks(task_ids)
    assert slots.tolist() == [[0, 1], [4, 5], [6, 7]]
    # out-of-range task id is clamped, never indexes past the pool
    assert m._block_slots_for_tasks(torch.tensor([99])).max().item() < m.prompt_pool.n_prompts


def test_write_routing_log_emits_json_when_out_dir_set(tmp_path):
    """_write_routing_log must produce routing.json with a correct overall hit-rate.

    Regression: the first feasibility runs produced no routing.json. This pins the
    writer in isolation (no model) — given hits/totals it writes the file and computes
    overall = sum(hits)/sum(totals).
    """
    from doccl.methods.hybrid_routed_prompt import HybridRoutedPrompt

    # Build a method shell without running __init__ (no model needed for the writer).
    m = HybridRoutedPrompt.__new__(HybridRoutedPrompt)
    m.router_mode = "hybrid"
    m.out_dir = str(tmp_path)
    m.prompt_pool = _pool(n_tasks=2, slots_per_task=2)

    m._write_routing_log(hits={0: 3, 1: 1}, totals={0: 4, 1: 4}, confusion={0: [3, 1], 1: [3, 1]})

    f = tmp_path / "routing.json"
    assert f.exists(), "routing.json was not written"
    import json

    d = json.loads(f.read_text())
    assert d["router"] == "hybrid"
    assert abs(d["overall_hit_rate"] - 0.5) < 1e-9  # (3+1)/(4+4)
    assert d["per_task_hit_rate"]["0"] == 0.75
