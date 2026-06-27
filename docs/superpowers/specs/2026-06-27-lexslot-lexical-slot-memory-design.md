# LexSlot — Lexical-Slot Head Memory for continual document IE

**Date:** 2026-06-27
**Status:** design (pre-implementation) — lit-checked NOVEL
**Owner:** thanh

## Context

**Why this is being built.** The thesis needs a method that protects the (head-localized)
locus of forgetting in continual document-IE while *also* allowing positive transfer between
related tasks — and does so without the failure modes we have already measured and ruled out:

- **Replay (ER/DER++)** works (DIL AA ~88) but stores raw exemplars.
- **DocCL** (depth/head-targeted EWC + KD + small head replay) already **matches replay on DIL**
  (AA ~85, BWT ~−3) and its `uniform`-vs-`head_only` ablation shows targeting the head is **2×**
  better than spreading the budget (AA 85 vs 42) — strong support for the head-locus thesis. But
  DocCL still uses one *shared* head penalised uniformly, and it collapses on CIL (growing head).
- **Head-merging (DocMERGE)**: conflicting head-deltas cancel → fails.
- **LCA** (merge + Gaussian align): over-regularises per-token features → AA ≈ naive.
- **Gradient-subspace transfer (HGT/CUBER)**: α is a stability↔plasticity dial, no positive BWT.
- **Prompt/lexical routing (HRP)**: sparse-OCR routing is excellent (0.92 vs 0.29 dense, drift-
  immune) but routing the *prompts* doesn't stop the *shared head* drifting → AA caps ~30.

The throughline: forgetting lives in the **shared classifier head**; the only thing that protects
it is re-grounding with old-task information; routing *around* it cannot. **LexSlot makes the head
itself not-fully-shared, and uses the validated drift-immune OCR signal to decide the sharing.**

**The idea.** A small **key-value adapter of `k` learnable slots** beside the classifier head of a
(mostly-frozen) LayoutLMv3. During training on task `t`, the **OCR lexical signature** (BM25/TF-IDF
over the page's tokens) selects which slots task-`t` may *update*. Inter-task **vocabulary overlap
`S` directly determines parameter-sharing**: lexically similar tasks (SROIE↔CORD, measured `S=0.31`,
shared "total/tax/cash") write to **shared slots** (overlapping knowledge co-trains → transfer);
lexically distinct tasks (FUNSD, `S≈0.10`) get **isolated slots** (not overwritten → no forgetting).
Sharing is **soft/graded by `S`** (the measured 0.31-vs-0.10 is graded, not binary). At inference
**all slots are active and additive** into the logits — **no routing needed**, which is what sidesteps
HRP's "0.92≠1.0, misroute=wrong answer" bottleneck. The slot routing is a **deterministic function
of document vocabulary** — zero learned routing parameters, immune to model-state drift.

### Novelty (lit-checked — `research-validator`, brutal pass)

| Axis | Status |
|---|---|
| Input-lexical (OCR BM25/TF-IDF) statistics selecting which params to UPDATE in CL | **NOVEL** (all of PackNet/HAT/SupSup/DEN/MoE-CL use learned gate / task-id / activation / magnitude) |
| Vocabulary-overlap → parameter-sharing structure (shared+isolated in one mechanism) | **NOVEL** |
| Head-specific lexical-slot adapter on LayoutLMv3 doc-IE | **NOVEL** |
| Zero learned routing params (deterministic from vocab, drift-immune) | **NOVEL (the moat)** — every competitor's router is trainable & can drift |

**Mandatory baselines / citations (from the check):**
- **Sparse Memory Finetuning (arXiv 2510.15103)** — *rank-1 threat*. Its "TF-IDF" is over **slot
  activation counts**, not document tokens. Needs a dedicated related-work paragraph **AND a direct
  ablation: activation-count-TF-IDF slot routing vs our input-lexical-TF-IDF routing** on the same
  sequence (surgically closes the attack).
- **MoLoRE** (EMNLP'25 Findings, 2025.findings-emnlp.718) — continual-IE LoRA-experts; **mandatory
  experimental baseline**; distinction = lexical (0 learned params) vs trained gating net.
- **LoDA** (ICML'26, arXiv 2603.00191) — shared/isolated subspace decomposition; closest structural
  analogue; related-work distinction = activation-energy signal (drifts, needs old data) vs lexical.
- **AMD-Proj** (MDPI'26) — the only doc-IE CL method; **mandatory comparison, must beat it.**
- Gururangan ACL'20 (vocab-overlap→transfer, dataset-level) + DEN ICLR'18 (data-stats param select):
  cite as motivation/precursor, distinguish (we operate at training-time slot granularity / drift-
  immune pre-model signal). Not blocking.

## The method (LexSlot)

Builds on `DocCL` (the working method). Head `W ∈ ℝ^{C×d}` (classifier `nn.Linear`, grows in CIL),
backbone frozen except the head (DocCL's `head_only`-style regime, which was the best DocCL variant).

**Slot memory.** A `LexSlotMemory` module: `k` slots, each a learnable value `V_s ∈ ℝ^{d}` (and a
per-slot logit projection `P_s ∈ ℝ^{C}` so a slot contributes a `(C,)` logit bias). A token's final
logit = `W·f + Σ_s a_s · (P_s ⊙ <V_s, f>)` (slots additively modulate the head; exact form an
implementation detail, kept small). `k` ≈ 32–64.

**Lexical slot ownership (the novel core).** Per task `t`, compute its OCR signature `sig_t`
(reuse `sparse_doc_vectors` / `accumulate_signature` from HRP). Define inter-task similarity
`S[t,τ] = cos(sig_t, sig_τ)` (reuse the validated S computation — measured 0.31 SROIE↔CORD,
0.10 FUNSD). Maintain a **slot→task soft-ownership** matrix. For task `t`, its **trainable slot
mask** `m_t ∈ [0,1]^k` is: a fixed block of *fresh* slots task-`t` owns, PLUS *graded* write-access
to prior tasks' slots weighted by `S[t,τ]` (high-S → can update τ's slots; low-S → frozen out).
Concretely `m_t[s] = max over owning-task τ of (S[t,τ] if τ≠t else 1)`, thresholded softly. The
mask scales the slot gradients (a grad hook, like HGT), so low-S tasks cannot overwrite an
unrelated task's slots, while high-S tasks co-train shared slots.

**Lifecycle** (on `ContinualMethod` / subclass `DocCL` to inherit its head-replay + KD + Fisher):
- `before_task(t)`: compute `sig_t` (from a pass over the loader's input_ids), update `S`, derive
  `m_t`, register the slot-gradient mask hook on `LexSlotMemory` params.
- `train_task`: DocCL's loop (CE + KD + small head-replay + Fisher penalty on `W`), unchanged,
  plus the slot memory in the forward; the mask hook gates slot updates by `m_t`.
- `after_task(t)`: freeze the fresh slots task-`t` claimed (write-once for *isolated* slots; shared
  slots remain updatable by future high-S tasks); accumulate `sig_t`.
- `evaluate`: standard forward, **all slots active** (additive) — no routing.

**Config axes (the experiments live here):**
- `slot_sharing: {soft, hard, off}` — `soft` = graded by S (the design); `hard` = threshold cutoff;
  `off` = every task fully isolated (no-transfer control, the "is sharing the source of transfer?"
  ablation). Default `soft`.
- `n_slots`, `slots_per_task` (fresh block size), `share_threshold` (for hard).
- `lexical_signal: {ocr, activation}` — `ocr` = our input-token TF-IDF (default); `activation` =
  the **Sparse-Memory-Finetuning-style activation-count TF-IDF** → the rank-1-threat ablation.

### Why it can succeed where the others failed (the load-bearing hypothesis — to be tested)

- vs HRP: the head is no longer fully shared → the drift channel that capped HRP is structurally
  reduced (isolated slots are frozen).
- vs merge/HGT: transfer is *positive sharing* (high-S tasks co-train the same lexical→label rule),
  not weight-averaging (cancels) or gradient-projection (no transfer).
- vs frozen-isolation: shared slots give transfer; isolated slots give protection — **both at once**.
- **The risk (explicit):** shared slots ARE updated by multiple tasks, so they *can* drift. The bet
  (user's call) is that this drift is *helpful* — it is the same "total→B-TOTAL" rule being refined
  by another receipt task → should manifest as **positive transfer on the SROIE↔CORD pair**, not
  forgetting. If shared slots instead just drift (old-task F1 on shared labels drops), the sharing
  is harmful and we fall back to `slot_sharing=off` (isolation-only) or rethink. **The first
  experiment's `soft`-vs-`off` contrast tests exactly this.**

## Files

**New — `doccl/methods/lexslot_memory.py`**: `LexSlotMemory(nn.Module)` (k slots, additive logit
modulation, `slot_grad_mask` buffer + a `register_hook` that scales slot grads by the active mask).
**New — `doccl/methods/lexslot.py`**: `LexSlot(DocCL)` — overrides `before_task`/`after_task` to
compute `sig_t`, update `S`, derive `m_t`, attach the slot memory + mask hook; reuses DocCL's
`train_task` (CE+KD+replay+Fisher) and adds the slot memory to the forward via the wrapper.
- Reuse: `sparse_doc_vectors` / signature accumulation (`hybrid_routed_prompt.py`), the S-matrix
  computation (`experiments/compute_task_similarity_S.py`), DocCL's machinery, `token_features`
  (for the slot `<V_s, f>` term), the grad-mask-hook pattern (from `hgt.py`).
**New — `configs/method/lexslot.yaml`**: keys above + DocCL's (lambda_/kd_alpha/fisher/replay).
**Edit — `scripts/train.py`**: register `lexslot` in `METHOD_REGISTRY` + `_STD_FORWARD`.
**New tests** — `tests/methods/test_lexslot.py` (synthetic: mask from a known S routes high-S tasks
to shared slots & isolates low-S; mask hook zeroes isolated-slot grads; soft vs hard vs off; slot
memory forward shape; reuse the HRP disjoint-vocab invariant) + register in the e2e lifecycle test.

## Verification

1. **Unit/lint:** `pytest tests/methods/test_lexslot.py`; ruff + black clean. Invariants:
   given S=[[1,.3,.1],[.3,1,.05],[.1,.05,1]], task-2's mask grants graded write to task-1's slots
   (S=.3) and ~none to task-0's (S=.1); `slot_sharing=off` → every task's mask is its own block only;
   mask hook scales slot grads correctly; all-active additive forward reproduces a logit shape (C).
2. **E2E:** `pytest tests/methods/test_all_methods_e2e.py -k lexslot` — CIL+DIL lifecycle on real
   LayoutLMv3, head growth survived, slot memory + mask hook run clean.
3. **Core experiment (the go/no-go) — local 2060, VRAM-safe (`batch_size=2 gradient_checkpointing=
   true num_workers=0 wandb.mode=offline`), ONE dataset builder at a time:**
   - On **`dil`** (FUNSD/SROIE/CORD — the measured-S benchmark): `lexslot {soft, off}` + `DocCL`
     (head_only) + `ER` + `naive`. `analyze_results.py --source local`.
   - **Headline tests:** (a) does `soft` beat `off` on **per-task F1 of SHARED labels** (SROIE/CORD
     "total" etc.) — i.e. does lexical slot-sharing produce **positive transfer** on the S=0.31 pair?
     (b) does `lexslot` match DocCL/ER AA on DIL while isolating FUNSD (no drift on the S≈0.10 task)?
     (c) BWT vs DocCL.
   - A flat/negative `soft`-vs-`off` (sharing doesn't transfer, or shared slots drift) → the sharing
     hypothesis is falsified; fall back to isolation-only or the analysis framing. Either is a result.
4. **If go:** add the rank-1 ablation (`lexical_signal=activation` vs `ocr`) and the mandatory
   baselines (MoLoRE, LoDA, AMD-Proj) — heavy, on a rented GPU; and `cil_cord` (CIL stress).

## Notes / risks
- **Build core + go/no-go FIRST** (user's call); baselines (MoLoRE/LoDA/AMD-Proj) only after the
  mechanism is shown to work — avoids investing in 3 method ports before knowing the idea is real.
- **Shared-slot drift is the make-or-break** — `soft`-vs-`off` is designed to test it; honest
  negative is publishable (completes the "what does/doesn't transfer on the head" story).
- Builds on DocCL (already ~85 on DIL) → strong floor; LexSlot must *improve transfer/CIL* over it
  to justify itself, not just match.
- Local box limits (RTX 2060, <5 GB VRAM / <14 GB RAM); the heavy baseline suite + cil_cord → rented GPU.
- The lit-check's rank-1 threat (2510.15103 TF-IDF naming collision) is handled by BOTH a related-work
  paragraph AND the `lexical_signal=activation` ablation — make sure both ship.
