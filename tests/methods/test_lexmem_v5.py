"""Unit tests for LexMem v5's RelationalMemory graph (synthetic, fast, GPU-free).

Covers the graph-as-memory invariants that the method's correctness rests on:
append-only growth (past nodes/edges never rewritten), the edges-on vs edges-off
reconstruction contract (the Claim-A ablation must be a real behavioural fork),
reconstruction shapes and label bookkeeping, deterministic seeded sampling, and
storage accounting. Full lifecycle on a real LayoutLMv3 lives in the e2e suite.
"""

from __future__ import annotations

import torch

from doccl.methods.lexmem_v5 import ClassNode, RelationalMemory

_D = 8


def _node(task_id: int, label: int, mu_val: float, var: float = 1.0, count: int = 100) -> ClassNode:
    mu = torch.full((_D,), mu_val)
    return ClassNode(
        task_id=task_id,
        label=label,
        mu=mu,
        cov_diag=torch.full((_D,), var),
        cov_lowrank=torch.empty(0, _D),
        count=count,
        doc_labels=frozenset({label}),
    )


def _graph(n_hops: int = 2) -> RelationalMemory:
    return RelationalMemory(hidden_dim=_D, cov_rank=0, n_hops=n_hops, edge_temp=1.0)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_add_task_nodes_wires_symmetric_edges():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 2, 1.0)])
    assert len(g.nodes) == 2
    assert g.edges[0][1] == g.edges[1][0], "edges must be symmetric"
    assert 0 not in g.edges[0], "no self-edges"


def test_append_only_does_not_rewrite_prior_edges():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 2, 1.0)])
    prior = {(i, j): g.edges[i][j] for i in g.edges for j in g.edges[i]}
    g.add_task_nodes([_node(1, 1, 0.5)])  # a later task adds a node
    for (i, j), w in prior.items():
        assert g.edges[i][j] == w, "prior edges must be immutable after a new task"
    assert len(g.nodes) == 3
    assert 2 in g.edges and len(g.edges[2]) == 2, "new node wires to all prior nodes"


def test_node_ids_for_tasks_filters_by_task():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 2, 1.0)])
    g.add_task_nodes([_node(1, 3, 2.0)])
    assert g.node_ids_for_tasks({0}) == [0, 1]
    assert g.node_ids_for_tasks({1}) == [2]
    assert g.node_ids_for_tasks({0, 1}) == [0, 1, 2]


def test_reconstruct_shapes_and_labels():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 5, 3.0)])
    feats, labels = g.reconstruct(
        [0, 1], per_node=4, edges_enabled=False, generator=_gen(), device=torch.device("cpu")
    )
    assert feats.shape == (8, _D)  # 2 nodes * 4 per node
    assert labels.tolist() == [1, 1, 1, 1, 5, 5, 5, 5]


def test_reconstruct_empty_node_list():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0)])
    feats, labels = g.reconstruct(
        [], per_node=4, edges_enabled=True, generator=_gen(), device=torch.device("cpu")
    )
    assert feats.shape == (0, _D)
    assert labels.shape == (0,)


def test_edges_on_vs_off_are_different_distributions():
    """The ablation must be a real behavioural fork: message-passing pulls a node's
    reconstruction toward its neighbours, so the edges-on mean differs from the
    isolated-node (edges-off) mean when a neighbour sits far away in feature space."""
    g = _graph(n_hops=2)
    # node 0 at 0.0, its only neighbour node 1 at 10.0 -> message passing shifts node 0
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 2, 10.0)])
    off, _ = g.reconstruct(
        [0], per_node=2000, edges_enabled=False, generator=_gen(1), device=torch.device("cpu")
    )
    on, _ = g.reconstruct(
        [0], per_node=2000, edges_enabled=True, generator=_gen(1), device=torch.device("cpu")
    )
    off_mean = off.mean().item()
    on_mean = on.mean().item()
    assert abs(off_mean - 0.0) < 0.2, "edges-off samples the node's own Gaussian (~0.0)"
    assert on_mean > off_mean + 0.5, "edges-on must pull toward the far neighbour (10.0)"


def test_reconstruction_is_seed_deterministic():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0), _node(0, 2, 1.0)])
    a, _ = g.reconstruct(
        [0, 1], per_node=8, edges_enabled=True, generator=_gen(42), device=torch.device("cpu")
    )
    b, _ = g.reconstruct(
        [0, 1], per_node=8, edges_enabled=True, generator=_gen(42), device=torch.device("cpu")
    )
    assert torch.allclose(a, b), "same seed must give identical reconstruction"


def test_edge_weight_boosts_same_label_cooccurrence():
    g = _graph()
    # same mu so cosine term is equal; label match should raise the co-occurrence term
    same = g._edge_weight(_node(0, 1, 1.0), _node(1, 1, 1.0))
    diff = g._edge_weight(_node(0, 1, 1.0), _node(1, 9, 1.0))
    assert same > diff, "shared label (DIL unified space) must strengthen the edge"


def test_storage_accounting_grows_with_nodes():
    g = _graph()
    g.add_task_nodes([_node(0, 1, 0.0)])
    one = g.storage_bytes()
    g.add_task_nodes([_node(1, 2, 1.0), _node(1, 3, 2.0)])
    three = g.storage_bytes()
    assert three > one, "storage must grow as nodes are added"
    # each node stores mu + cov_diag (2 * D floats) at minimum
    assert one >= 2 * _D * 4
