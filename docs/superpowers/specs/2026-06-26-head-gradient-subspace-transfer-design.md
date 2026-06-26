# HGT — Head-localized Gradient-subspace Transfer for continual learning

**Date:** 2026-06-26
**Status:** design (pre-implementation)
**Owner:** thanh

## Context

**Why this is being built.** The thesis target is a **general** continual-learning method —
backbone-agnostic the way Experience Replay (ER) is — that beats ER on the two axes ER cannot
win: it is **buffer-free** (stores no raw exemplars) and delivers **positive backward transfer**
(BWT > 0; old tasks *improve* as new ones train). This direction follows a sequence of measured
negatives that together form the paper's motivation:

- **Model-merging of the head fails** (our RCA): per-task head deltas conflict and cancel
  (`||Σδ||/Σ||δ||=0.57`); even the best rule ≈ no-merge. See `clue-docmerge-method`.
- **LCA (ICLR'26 image-CIL SoTA) does not port to doc-IE** (we ran it): AA 41.9 ≈ naive 38.7,
  half of replay (~88); its TIES backbone-merge helps BWT but its Gaussian-feature classifier
  alignment over-regularizes per-token features. See `clue-lca-paper`.
- **Prompt/frozen-backbone methods cap at AA ~30** on `dil` — head drift, routing can't fix it.
- **ER wins because replay re-grounds the head with real old data** — but it stores a buffer.

The unifying diagnostic (already measured, **already cross-backbone**): catastrophic forgetting
is **head-localized** — head Fisher-weighted displacement ~1e-4 vs backbone ~1e-8, reproduced on
LayoutLMv3, BERT, LiLT, BROS (4 architectures). This is an *established* phenomenon
(BiC/LUCIR/Lesort'21/Davari'22/Liu&Huang'23/Lapacz'25) — we cite it as corroboration, never claim
discovery; our confirmation across multimodal document encoders is the domain contribution.

**The idea.** Two literatures have never been combined: (a) **positive-BWT via gradient-subspace
transfer** — exists only in CUBER (NeurIPS'22); every other buffer-free gradient-projection method
(GPM, OGD, Adam-NSCL, TRGP, SGP, …) is whole-network and protection-only (BWT≈0); and (b) the
**head-localization diagnostic**. We apply gradient-subspace transfer **surgically to the
classifier head**, motivated by the diagnostic. The result is buffer-free, positive-BWT, and —
because the head-locus phenomenon is general — generalizes across backbones like ER does.

### Novelty (lit-checked — `research-validator`, brutal pass)

| Claim | Status |
|---|---|
| Head-localized gradient-subspace projection | **NOVEL** (every prior gradient-proj method is whole-network) |
| Diagnostic-motivated head localization | **NOVEL framing** |
| Buffer-free + positive-BWT + backbone-general, beating ER | **NOVEL combined claim** |
| Head-gradient-subspace on a frozen pretrained backbone | **NOVEL** (RanPAC/SLCA/EASE line is all BWT≈0, none use it) |
| "Forgetting is head-localized" | **ESTABLISHED — cite, don't claim** (BiC/LUCIR/Lesort/Davari/Lapacz) |
| Positive-BWT via gradients | **CUBER owns it → MANDATORY baseline** |

**Mandatory baseline: CUBER (arXiv 2211.00789).** Distinctions (all defensible, all in the
ablation): (1) diagnostic-motivated, not ad-hoc; (2) cost O(head_dim × n_tasks), independent of
backbone depth (CUBER's per-layer subspace is backbone-specific); (3) frozen-pretrained-backbone
compatible (CUBER's whole-network updates are forbidden there); (4) cheaper (one head SVD vs
per-layer SVD across all layers). **Watch:** ETCL (2601.05623) — head-only BKT but *sparse-mask*
mechanism, no diagnostic; cite + distinguish.

## The method (HGT)

Head `W ∈ ℝ^{C×d}` (the classifier `nn.Linear`; `d`=hidden, `C`=labels, grows in CIL). Backbone
`f`. For task `t` with frozen-or-trainable `f`:

1. **`train_task`**: standard CE training of the head (and backbone if `backbone_trainable`). No
   change to the loss.
2. **`after_task(t)` — store the head's old-task subspace.** Accumulate the **uncentered feature
   second-moment** of task `t` over its scored tokens:
   `M_t = Σ_x fᵀf  ∈ ℝ^{d×d}` (features `f = backbone(x)` at label≠-100 tokens). Take its
   top-`k` eigenvectors `U_t ∈ ℝ^{d×k}` (the directions in which task `t`'s head logits are
   sensitive to feature changes — equivalently the input-subspace GPM/Adam-NSCL store, but only
   for the head's input). Keep `{U_t}` per task. Cost: one `d×d` eigendecomposition per task,
   `k·d` floats stored — tiny, backbone-depth-independent.
3. **Gradient steering during `train_task(t)` (t>0) — the transfer rule.** For each head-gradient
   `G = ∂L_t/∂W ∈ ℝ^{C×d}` decompose its *input-space* footprint against each old task's `U_τ`:
   - `G_proj^τ = G U_τ U_τᵀ` (component acting on task-τ's feature subspace),
   - `G_orth = G (I − Σ_τ U_τ U_τᵀ)` (acts on no old task — free to update).
   The **CUBER-style positive-transfer step** (the key): for the part of `G` lying in `U_τ`, keep
   it **only if it also decreases task-τ's loss** — measured *without τ's data* via the stored
   second moment as a metric: a head update `ΔW` changes τ's logits by `ΔW U_τ`; the update is
   "τ-helpful" when its alignment with τ's own descent direction (estimated from `U_τ` + the
   frozen old head row geometry) is positive. Concretely (per CUBER): split each `G_proj^τ` into
   its component aligned with τ's stored gradient-subspace sign and the conflicting component;
   **let the aligned component through (→ τ improves, positive BWT), zero the conflicting one
   (→ τ protected).** `G_used = G_orth + Σ_τ α · aligned(G_proj^τ)`. `α` is the transfer
   strength (ablation; `α=0` recovers pure GPM-style protection → BWT≈0 control).
4. **`evaluate`**: standard forward (HGT is a `_STD_FORWARD` method — no special inference).

**Config axes (the experiments live here):**
- `backbone_trainable: {false, true}` — the deferred frozen-vs-trainable decision; the first
  experiment settles it. Frozen → clean head-locus story + cheap + CUBER-differentiated; trainable
  → higher AA, closer to ER's setting.
- `transfer_alpha: {0.0, 0.5, 1.0}` — `0` = protection-only (GPM-like, BWT≈0 control); >0 = the
  positive-BWT transfer. **The `α=0` vs `α>0` contrast is the headline ablation: does head-gradient
  transfer actually produce positive BWT?**
- `subspace_k` — top-eigenvector count per task (memory/fidelity knob).

### Why positive BWT can happen here (the load-bearing hypothesis — to be tested)

Old-task features are fixed (frozen backbone) or slow (trainable). "Improving task τ" = a head
update that lowers τ's loss on τ's feature distribution. We don't store τ's data, but `U_t` +
the (frozen) old head rows encode τ's logit geometry. The transfer step keeps exactly the
new-task gradient component that *co-descends* τ. **The risk (explicit):** if different tasks'
head-gradients live in orthogonal feature-subspaces, the aligned component is ~0 → BWT≈0 → we
degrade to GPM. The first experiment's `α=0`-vs-`α>0` contrast tests this directly; a flat result
is itself a finding (and the analysis-paper fallback).

## Files

**New — `doccl/methods/hgt.py`** (`HGT(NaiveFineTune)`):
- `after_task`: `_accumulate_feature_moment(loader)` (uses `LayoutLMv3Wrapper.token_features` —
  already added for LCA) → eigvecs `U_t`; store per task.
- `train_task`: standard loop + a **gradient hook on the head** (`W.register_hook`) that rewrites
  the head gradient to `G_used` (steps 3) before the optimizer sees it. Backbone params follow
  `backbone_trainable`. Reuse `make_early_stopper` + AMP helpers.
- `evaluate`: standard forward (inherited from `NaiveFineTune`).
- A `doccl/methods/grad_subspace.py` helper: `feature_moment`, `top_eigvecs`, `project_and_steer`
  (the math — the only net-new numerics; unit-tested on synthetic tensors).

**New — `doccl/methods/cuber.py`** (`CUBER(NaiveFineTune)`): faithful port of CUBER (whole-network
gradient-subspace, positive BWT) as the mandatory baseline. Stores per-layer feature moments,
steers every layer's gradient. Reuses `grad_subspace.py`. (If a reference repo exists, port
verbatim; else implement from arXiv 2211.00789 §3.)

**New configs** `configs/method/hgt.yaml`, `configs/method/cuber.yaml`.
**Edit** `scripts/train.py`: register `hgt`, `cuber` in `METHOD_REGISTRY` + `_STD_FORWARD`.
**New tests** `tests/methods/test_hgt.py` (synthetic: projection/steering math — orthogonal
gradients → `G_used==G`; aligned gradient → transfer component non-zero; `α=0` → pure orthogonal
projection == GPM; subspace eigvec shapes) + register `hgt`/`cuber` in
`tests/methods/test_all_methods_e2e.py`.

### Reuse (do not reimplement)
- `LayoutLMv3Wrapper.token_features` (token features pre-head) + `model.model.classifier` —
  already present.
- `fisher.py` (head-locus diagnostic — re-validate the premise), `param_groups_by_depth`.
- `ContinualMethod`/`NaiveFineTune` lifecycle, `CLMetricsTracker` (BWT sign convention already
  represents positive BWT), `make_early_stopper`, AMP helpers.

## Verification

1. **Unit/lint:** `pytest tests/methods/test_hgt.py`; `ruff`/`black` clean. Key invariants:
   `α=0` ⇒ steered gradient is orthogonal-projection only (GPM-equivalent, BWT-protection);
   orthogonal old-subspaces ⇒ `G_used==G`; transfer component differentiable + finite.
2. **E2E:** `pytest tests/methods/test_all_methods_e2e.py -k "hgt or cuber"` — full CIL+DIL
   lifecycle on real LayoutLMv3, head growth survived.
3. **Core experiment (the go/no-go) — `dil`, seed 42, the local 2060 (VRAM-safe:
   `batch_size=2 gradient_checkpointing=true num_workers=0 wandb.mode=offline`), ONE dataset
   builder at a time:**
   - **HGT `α∈{0, 0.5, 1.0}` × `backbone_trainable∈{false,true}`** + **CUBER** + **ER** (buffer
     reference) + **naive** (floor). `analyze_results.py --source local`.
   - **Headline test:** does `α>0` give **BWT > 0** while `α=0` gives BWT≈0? (proves head-gradient
     *transfer*, not just protection). Does HGT match ER's AA **buffer-free**? Does HGT beat/match
     CUBER while being cheaper + frozen-compatible?
   - A flat `α` result (no positive BWT) → the head-gradient-transfer hypothesis is falsified on
     doc-IE → fall back to the analysis-paper framing (still strong: the measured-failure story).
4. **Generality (after `dil` is positive):** repeat the `α=0`-vs-`α>0` + CUBER + ER comparison on
   **BERT, LiLT, BROS** (all wired) → the "document-general, generalizes like ER" claim. Heavy →
   rented GPU per `RUNBOOK.md`, not the laptop.

## Notes / risks
- **Frozen-vs-trainable is deferred to experiment** (config axis), per decision. Frozen is the
  cleaner/cheaper/CUBER-differentiated story but caps AA; trainable competes with CUBER on its turf.
- **The positive-BWT-on-the-head hypothesis is unproven** (step-3 risk above) — the `α` ablation is
  designed to test it cheaply; honest negative → analysis paper.
- **CUBER port fidelity** is essential (it's the headline baseline); port verbatim if a repo exists.
- Local box limits (RTX 2060, <5GB VRAM / <14GB RAM); heavy/multi-backbone runs on rented GPU.
- This spec keeps the backbone regime + transfer strength as ablations so the *first run answers
  the open questions* rather than us guessing — explicitly chosen after two prior build-first burns.
