# LexMem v5 — Relational Reconstruction Memory (graph-as-memory)

**Date:** 2026-07-02
**Status:** DESIGN (not implemented; nothing runs until user green-lights and GPU is clear)
**Builds on:** `doccl/methods/lexmem.py` (v3b spine), `configs/method/lexmem_v3b.yaml`
**Supersedes the goal of:** LexSlot, LexMem KV-slot, Lexical Ledger — all falsified as
*stores of answers*. v5 stores the *ability to reconstruct training signal*, relationally.

---

## 1. The idea, stated precisely

Do not store the memory. Store a **sparse cue** and a **reconstruction operator** such that,
when triggered, the *current* head can regenerate the training signal it needs to retain a
past task. Concretely we regenerate **(feature, label) pairs in today's encoder space** and
consolidate them into the head each session — synthesizing the balanced multi-task gradient
that replay uniquely provides (Finding 3 of `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`),
without storing documents.

The novel object is that the memory is a **graph, not a bank**. Reconstruction of a past
task is *message-passing over that task's node and its neighbors* — so competence is
reconstituted from relationships, not read from an isolated slot. The falsifiable bet:
**relational reconstruction beats independent-node reconstruction at equal storage**, because
edges carry recoverable continual-learning signal that a bank of identical nodes cannot.

### Why each failed method's killer is architecturally absent here
- LexSlot died: residual corrections stored in a *stale* feature space. → v5 stores nothing
  as a fixed answer; cues are re-projected through the **current trunk** before completion.
- Ledger died: cumulative keys *collided and served stale outputs*. → v5 keys never serve
  outputs; they *trigger reconstruction of training signal*. A fuzzy/colliding cue completes
  to a nearby basin (graceful) instead of returning a wrong logit.
- LexMem asymptoted (AA 66): head only saw *task-0-biased* online-Fisher gradients. → v5
  head sees *reconstructed gradients from ALL past tasks every session* — the balanced signal
  only real replay gave.

---

## 2. Architecture (Complementary Learning Systems instantiation)

Three components, each *already validated in isolation* by the falsification chain:

| CLS role | v5 component | validated by |
|----------|-------------|--------------|
| Cortex (slow, semantic) | v3b trunk: plastic-diagnosed backbone + frozen base head, EWC λ=300 (CKA ~0.9-0.95) | LexMem v3b: trunk stable, task-0 drop −4.4 |
| Hippocampus (fast, sparse, episodic index) | **relational memory graph** G (nodes + edges below) | Ledger: input-structural keys are the drift-immune ones |
| Consolidation (generative replay) | fire cues → message-pass → reconstruct (feature,label) → train head | Finding 3: only replayed gradients ground the head |

### The graph G (grows, never overwrites — the load-bearing invariant)
- **Nodes** = (task_id, class_id) pairs. Node payload (the sparse trace, ~few KB/node):
  - `mu`, `L` — per-class feature **centroid + low-rank/diagonal covariance** in the encoder
    space *at the time the task was learned* (a Gaussian; the reconstruction seed).
  - `anchor` — a handful of input-structural keys (lexeme × layout-cell) for that class,
    reused from the Ledger key construction (drift-immune cue).
- **Edges** = relations between class-nodes. Edge weight = relational statistic that is
  *encoder-coordinate-free* (drift-stable): co-occurrence in the same document/region,
  layout adjacency (below/right-of), label-schema similarity, feature-centroid cosine.
- **Append-only history:** nodes/edges for tasks < current are **frozen**. A new task adds
  new nodes and adds edges *from new nodes to existing ones*; it never rewrites old payload.
  This is the recursion-terminator — the graph memory itself cannot catastrophically forget
  because past structure is immutable; only the frontier is plastic.

### Reconstruction operator (the "settle the attractor" step)
To reconstruct training signal for past task t at session k:
1. Re-project each stored anchor/centroid of t's nodes **through the current trunk** →
   current-space seed features (this is the freshness/drift-immunity step).
2. **Message-pass** over t's nodes *and their graph neighbors* (K hops): a node's reconstructed
   feature distribution is a function of its own payload **and** its neighbors' — so a task is
   regenerated from its relationships, and later-added edges *reshape* older reconstructions
   (reconstructive memory, by design).
3. Sample (feature, label) pairs from the message-passed distributions; assemble a balanced
   batch across all past tasks.
4. Train the head (and the plastic trunk subset) on: new-task real batch **+** reconstructed
   past batch, with the v3b EWC penalty retained on the trunk.

---

## 3. Two-stage build (Stage 2 gated behind Stage 1 — do NOT skip the gate)

### Stage 1 — fixed-edge relational reconstruction + the edge-ablation gate
Cheapest possible test of "do relations carry CL signal at all," no new-forgetting risk.
- Edges are **computed, not learned**: from co-occurrence / layout-adjacency / centroid cosine.
- Reconstruction operator = message-passing with fixed edge weights over per-class Gaussians
  (start with mean-aggregation; no trainable GNN params yet).
- **Decisive ablation (isolates the graph's entire contribution):** same nodes, same storage,
  run (a) **edges ON** (full message-passing) vs (b) **edges = 0** (bank of identical Gaussians,
  i.e. FeCAM-style independent-node reconstruction). Everything else byte-identical.
- **Gate to proceed to Stage 2:** DIL AA(edges-on) − AA(edges-off) ≥ +3 pts AND edges-on
  clears the LexMem-v3b asymptote (AA > 66). If edges add nothing, relations are decoration
  → fall back to the pure Gaussian bank (still a clean, publishable *negative* result about
  relational structure in doc-IE), do NOT build Stage 2.

### Stage 2 — free-parameter growing graph (only if Stage 1 passes)
The graph *is* the memory: node/edge features + a small **GNN** are learnable parameters,
trained so message-passing regenerates good training signal.
- New task: add nodes/edges, run a few steps of **graph optimization** (the "sleep" step)
  training ONLY the new/frontier parameters; frozen history untouched (invariant from §2).
- Reconstruction operator = the learned GNN readout (a learned pattern-completer over G).
- Escalation path if Gaussian seeds prove too weak even with edges: replace mean-aggregation
  with a modern-Hopfield / attention readout (true pattern completion over the node bank).

---

## 4. Falsifiable claims & bars (the paper's spine)

| # | Claim | Bar | Decides |
|---|-------|-----|---------|
| A | Relational reconstruction > independent-node reconstruction at equal storage | AA(edges-on) − AA(edges-off) ≥ +3 on DIL | is the *graph* the contribution? (Stage 1 gate) |
| B | Cue-driven reconstruction ≈ real-data replay at a fraction of storage | v5 AA within ~3 of ER (≈88) at ≤ 1/20th ER's stored bytes | is memory-as-reconstruction a real alternative to the buffer? |
| C | Growing-graph memory does not itself forget | frozen-history nodes' reconstruction quality flat across sessions | did we terminate the recursion, or rebuild LexMem? |

Storage accounting is mandatory in every table (bytes stored per task: v5 = nodes+edges;
ER = raw examples). The equal-storage axis is the whole point — report AA-vs-stored-bytes as
the headline figure.

---

## 5. Implementation map (reuse, don't rebuild)

New file `doccl/methods/lexmem_v5.py`, subclassing the v3b LexMem class — **inherit** the
trunk-freeze map, EWC machinery, AMP helpers, early-stopper, and the classifier-input hook
(`_capture_feats` already hands us raw token features; that is the reconstruction target space).

Replace only the memory object and the task-≥1 training body:
- Swap the KV-slot `mem.delta` for a `RelationalMemory` module holding G (nodes = Gaussians +
  anchors, edges = fixed [Stage 1] / learned [Stage 2]) with a `reconstruct(task_ids) →
  (feats, labels)` method (the message-passing operator).
- In `_train_memory` (see `lexmem.py:~217`), each session build the batch as
  `real_new ∪ reconstruct(all_past_tasks)`; keep the existing EWC-on-trunk term.
- `after_task`: extract per-class feature Gaussians from `token_features` on the current trunk
  (the wrapper method at `doccl/models/layoutlm_wrapper.py:384`), add nodes, add edges to
  existing nodes; freeze all prior nodes/edges.

Wiring: export from `doccl/methods/__init__.py`; add to `METHOD_REGISTRY` in `scripts/train.py`;
`configs/method/lexmem_v5.yaml` inheriting the v3b recipe (lr 5e-5, EWC λ300, freeze_late_n 4)
plus graph knobs (`edge_types`, `n_hops`, `cov_rank`, `recon_per_class`, `edges_enabled` for
the ablation, `graph_lr` for Stage 2). Add `tests/methods/test_lexmem_v5.py` (synthetic-tensor:
graph grows append-only, reconstruct returns current-space feats, edges-off == bank).

---

## 6. Novelty & honest risk (state in the paper)

- Generative replay (Shin 2017; van de Ven 2020) and CLS framing are **known**. Feature-Gaussian
  exemplar-free CIL (FeCAM/RanPAC/FeTrIL) is **known** — that is exactly the edges-OFF baseline.
- **The unclaimed object:** memory as a *growing relational graph whose message-passing
  reconstructs training signal*, with component placement driven by our diagnostic (input-cue
  index because those keys were *proven* drift-immune; feature-space reconstruction because the
  trunk was *proven* stable; head-side consolidation because forgetting was *localized* there),
  in structured document IE. Claim A's ablation is what earns the "relational" novelty — if it
  fails, the honest paper reports it and keeps the bank.
- **Primary risk:** Stage 2's free-parameter graph can itself forget. Mitigation is the
  append-only/frozen-history invariant (§2); Claim C is its explicit test. If C fails, stop —
  a plastic graph memory is LexMem with extra steps.
- **Scope caveat:** all bars are DIL first (where replay works and lexical routing holds at
  0.92); CIL and multi-backbone generalization are follow-on, not part of the gate.

---

## 7. Execution order (nothing runs until GPU clear + user go)
1. Kill the rogue `lexslot_fm` loop / free the GPU (pending user decision, separate).
2. Implement Stage 1 + tests (synthetic-tensor, local-box safe).
3. Run edges-on vs edges-off on DIL seed 42 → **Claim A gate**.
4. If gate passes: multi-seed (7,123) + Stage 2. If not: write the negative result, keep bank.
