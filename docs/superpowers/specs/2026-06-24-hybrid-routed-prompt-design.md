# Hybrid-Routed Prompt Pool (HRP) — feasibility design

**Date:** 2026-06-24
**Status:** feasibility prototype (pre-A* method development)
**Owner:** thanh

## Context

Inspired by *Continual Learning via Sparse Memory Finetuning* (arXiv 2510.15103), the
idea is a **side memory with key–value routing**: each task activates its own region of
a prompt/memory pool, so a frozen backbone never overwrites old knowledge. The repo
already contains the skeleton of this mechanism — L2P / CODA-Prompt
(`doccl/methods/prompt_base.py`) keep a learnable key pool, match a CLS query to keys by
cosine similarity, select top-k prompt slots, and inject them into a **frozen**
LayoutLMv3 via `forward_with_prompts`.

### Honest reframing (the critique that shaped this design)

The user's original framing made three claims that do not survive scrutiny; the design
keeps the sound core and drops the rest:

1. **"No forgetting anymore" is overclaimed.** Frozen backbone + per-task slots only
   removes forgetting *if the router sends each test document to the slot trained on its
   task*. Forgetting does not vanish — it **moves into the router**. Misrouting a
   task-2 document to a task-1 slot collapses accuracy exactly like catastrophic
   forgetting. This is the documented weak point of L2P-style methods, and it is *worse*
   for document KIE: CLS embeddings of layout-heavy forms (FUNSD/CORD/SROIE) overlap
   heavily, so dense-only routing is hard precisely where it matters.

2. **The "acceleration / 1M→5k→rerank" framing does not transfer.** We have ~10–40
   slots, not a million; a full similarity scan is free. Selling speed invites a
   desk-reject ("why ANN over 40 vectors?"). **The value is routing *accuracy*, not
   speed.**

3. **"Dense fixes sparse for OOD" is backwards** as stated, but correct as a *fusion*
   argument: sparse (BM25 over OCR tokens) pins exact entities/vocabulary, dense (CLS)
   catches paraphrase/layout, RRF fuses them. The novel, defensible ingredient is the
   **sparse signal over the document OCR token stream** — no doc-CL baseline here uses
   it, and it directly fixes dense misrouting when two forms look alike but use different
   vocabulary ("TOTAL/TAX" vs "Name/Date").

**Therefore HRP is:** a hybrid **dense+sparse router** over a task-pinned prompt pool,
whose contribution is *measured routing accuracy*, not "no forgetting" and not speed.

## Goal of this prototype

A fast, honest feasibility check answering: **does the sparse-over-OCR signal improve
task routing over dense-only routing, and does that translate to higher AA?** Success
bar (locked with user):

- **Primary:** routing accuracy = % of eval documents routed to their own task's slot
  block, **hybrid vs dense-only** (the diagnostic that makes this A*-worthy).
- **Secondary:** AA / BWT vs `der_pp` and `doccl` on a small scenario.

A routing-accuracy win even at comparable AA is the publishable result.

## Architecture

One new method file, subclassing the existing `PromptBasedMethod` — the **only**
extension points are `_build_prompt_modules` and `_select_prompts`. No changes to
`train.py`, the CL loop, the wrapper, or metrics. The frozen-backbone training/eval loop,
prompt injection, and prompt-aware early stopping are all inherited.

### Components

1. **Task-pinned prompt pool** (`HybridPromptPool`, a small `nn.Module`).
   - `prompts: (N, L_p, D)` and dense `keys: (N, D)` — same as `PromptPool`.
   - Slots are partitioned into **per-task blocks** of size `slots_per_task` (e.g. 5
     tasks × 2 slots = 10). `before_task(task)` records the active block for
     `task.task_id`. This task→block map is what makes a routing-accuracy metric
     well-defined and mirrors DualPrompt's task-pinned design.
   - A **sparse signature per slot**: a sparse term-weight vector accumulated from the
     OCR tokens of the documents that trained that slot's block (a length-`vocab`
     bag-of-token-ids count, L2-normalised → acts as the slot's BM25/TF-IDF "key").

2. **Hybrid router** (inside `_select_prompts`).
   - **Dense score:** cosine(query CLS, slot dense key) → `(B, N)`. (existing path)
   - **Sparse score:** build a per-document sparse vector from `batch["input_ids"]`
     (token-id counts with IDF weighting computed from the running document frequencies),
     cosine against each slot's sparse signature → `(B, N)`.
   - **Fusion:** Reciprocal Rank Fusion (RRF) over the two score rankings →
     `fused (B, N)`; select top-k blocks. RRF is rank-based so it needs no score
     calibration between the two engines — the right default for heterogeneous signals.
   - **Ablation switch** `router ∈ {dense, sparse, hybrid}` (config) so the *same* method
     runs dense-only (== L2P-ish), sparse-only, and hybrid — this is the controlled
     experiment, not three separate methods.

3. **Routing-accuracy logger.** During `evaluate(eval_loaders)`, the method knows the
   true task id of each loader. For each document it records whether the top-1 routed
   block == the true task's block, accumulates a confusion matrix over blocks, and writes
   `results/<run>/routing.json` (`{router, overall_hit_rate, per_task_hit_rate,
   confusion}`). Written from inside the method (it has `out_dir` via config) — **zero**
   `train.py` changes. If `out_dir` is unavailable it logs to the run logger and skips
   the file (never fails a run).

### Data flow (per training step, inherited loop)

```
batch ──► _query(batch) = frozen CLS  ─────────────► dense query (B,D)
   │                                                      │
   └─► batch["input_ids"] ─► sparse doc vector (B,V) ─────┤
                                                          ▼
                              HybridPromptPool.route(dense_q, sparse_q)
                                  dense cos (B,N) ─┐
                                  sparse cos (B,N) ─┤─ RRF ─► top-k blocks
                                                          ▼
                              prompt_embeds (B, k·L_p, D)  +  key_pull aux loss
                                                          ▼
                         model.forward_with_prompts(...)  (inherited)
```

`after_task`: accumulate the OCR-token signature for the just-trained block from
`train_loader`, and freeze that block's slots/keys (DualPrompt-style) so later tasks
cannot overwrite them — this is what enforces "each task owns its region."

## Components & boundaries

| Unit | Responsibility | Depends on |
|---|---|---|
| `HybridPromptPool` (nn.Module) | hold prompts + dense keys + sparse signatures; `route()` returns fused top-k + aux | torch only |
| `HybridRoutedPrompt` (method) | wire pool into `_build_prompt_modules`/`_select_prompts`; pin blocks in `before_task`; accumulate signature + freeze in `after_task`; log routing in `evaluate` | `PromptBasedMethod`, pool |
| sparse encoder (small helper) | `input_ids → sparse doc vector` with running IDF | torch only |

Each is independently testable on synthetic tensors (the `_TinyKIEDataset` pattern in
`tests/methods/test_all_methods_e2e.py`).

## Error handling

- Empty pool / `top_k > N`: clamp (existing `PromptPool` already does `min(top_k, N)`).
- First task has no prior signatures: sparse score is zero → fusion degenerates to dense
  on task 0 (correct — there is nothing to route against yet).
- Missing `out_dir`: routing log skipped with a warning, run continues.
- Sparse vocab size = backbone vocab (≈50k); store signatures as dense `(N, V)` float —
  N≤40 so this is a few MB, no sparse-tensor machinery needed (YAGNI).

## Testing

1. **Unit (CPU, synthetic):** `HybridPromptPool.route` returns correct shapes; RRF of two
   identical rankings == that ranking; sparse-only routing on two disjoint token
   vocabularies routes each to its own block with hit-rate 1.0; dense-only path matches
   the existing `PromptPool.select` top-k indices.
2. **E2E lifecycle (CPU, real LayoutLMv3, 1 epoch):** add `HybridRoutedPrompt` to the
   `tests/methods/test_all_methods_e2e.py` registry — 2-task CIL + DIL lifecycle runs and
   produces a `routing.json` with finite hit-rates. Marked `slow`.
3. **Feasibility run (small GPU/CPU):** `dil` scenario (FUNSD→SROIE→CORD, 3 disjoint
   tasks — best for routing since tasks are genuinely different documents), seed 42,
   few epochs, `wandb.mode=offline`, for `router ∈ {dense, sparse, hybrid}` + `der_pp` +
   `doccl`. Read AA/BWT from `analyze_results.py` and hit-rate from `routing.json`.

## What is explicitly out of scope (YAGNI)

- No ANN / acceleration / million-slot machinery — irrelevant at ≤40 slots.
- No cross-encoder reranker — RRF first; add only if RRF underperforms.
- No backbone training, no replay buffer — keep the router isolated so a win is
  attributable to routing alone (the user can fold in replay later when developing the
  full A* method).
- No new scenario / dataset / backbone — reuse `dil` + LayoutLMv3.
- No grid integration yet — this is feasibility; grid wiring happens only if it works.

## Files

- **New:** `doccl/methods/hybrid_routed_prompt.py` (pool + sparse encoder + method).
- **New:** `configs/method/hrp.yaml` (`name: hrp`, `router`, `n_prompts`,
  `slots_per_task`, `prompt_length`, `top_k`, `lambda_key`, `rrf_k`, `epochs`, `lr`).
- **New:** `tests/methods/test_hybrid_routed_prompt.py` (unit + registry hook).
- **Edit:** `scripts/train.py` — one line: add `"hrp": HybridRoutedPrompt` to
  `METHOD_REGISTRY` (it is a prompt method, so it stays *out* of `_STD_FORWARD`).
- **Edit:** `tests/methods/test_all_methods_e2e.py` — register `hrp` in the method map.

## Verification

- `uv run pytest tests/methods/test_hybrid_routed_prompt.py -v` — unit + routing-hit-rate
  invariants pass on CPU in seconds.
- `uv run pytest tests/methods/test_all_methods_e2e.py -k hrp` — full lifecycle on real
  LayoutLMv3 produces `routing.json`.
- Feasibility: three `router` variants + `der_pp` + `doccl` on `dil` seed 42; expect
  `hybrid` hit-rate ≥ `dense` hit-rate (the go/no-go signal) and report AA/BWT side by
  side. If `sparse`/`hybrid` does **not** beat `dense` routing accuracy, the idea is
  falsified cheaply — that is a successful feasibility outcome too.
