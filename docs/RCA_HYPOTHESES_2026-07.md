# Pre-registered hypotheses: root causes of forgetting in multimodal doc-IE baselines

**Status:** PRE-REGISTERED 2026-07-16, **before** any Tier B data exists
(`results/rca/dil_*_seed42_rca.json` — zero files on disk at time of writing; the
`run_rca_baselines.sh` chain is queued behind the seeds-conservation chain).
Written from Tier A artifacts only. Tier C (`docs/RCA_FORGETTING_BASELINES_2026-07.md`)
must adjudicate each hypothesis against the decision rules below **as written** —
post-hoc rule changes must be flagged as exploratory.

## Question

Baselines are multimodal (LayoutLMv3: text + 2D layout + vision). Does forgetting
correlate with modality — i.e., is the root cause modality-asymmetric drift — or is it
modality-independent (head/label-space interference)? Tier A established *what* forgetting
looks like; these hypotheses compete to explain *why*.

## Verified anchors (Tier A + priors, copy-checked on disk 2026-07-16)

- **Class-asymmetric extinction** (`results/rca/a1_summary.md`): in failing methods the
  sparse classes die completely while VALUE survives — naive KEY=0.0, HEADER=0.0, VALUE=46.9
  (AA 40.3); lwf KEY=0.0, HEADER=0.0, VALUE=51.4 (AA 43.7). ER is NOT degenerate
  (KEY 87.2 ≈ AA 86.8). Degeneracy flag also on: spectral_memory, coreset_memory,
  aglr_replay, lexmem_ctrl/v2, fisher_mask, lexslot_fm.
- **One-boundary collapse** (`results/rca/a2_family_signature.csv`): naive immediate drop
  −75.6, gradual +7.9 (partial recovery), immediate_share 1.12. Family bands: raw/logit
  replay 0.31–0.49, prompt 0.44–0.58, penalty (ewc) 1.16 but small total (−16.8),
  distill (lwf) 1.14 with large total (−69.1).
- **Head-refit oracle** (`results/head_refit_oracle.json`, dil seed42): naive final row
  funsd 17.7 / sroie 3.7 / cord 97.3, AA 39.6. Frozen-trunk linear head refit recovers
  pooled AA 55.3 (per-task probes: funsd 41.9, sroie 66.0, cord 97.2; joint ref 88.7)
  ⇒ features retain substantial signal the trained head no longer reads out.
- **Priors** (`docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`): F1 forgetting head-localized +
  architecture-general; F2 locus migrates under constraint; F3 only whole-doc replay grounds
  the head.
- **Architecture note:** LayoutLMv3 is single-stream — text and visual tokens share Q/K/V;
  there is **no separable fusion module** (`doccl/models/param_grouping.py` deliberately has
  no `fusion` group). Modality-specific parameters exist only at the input embeddings:
  `text_word_embed`, `layout_2d_pos_embed`, `image_patch_embed`.

## Adjudication data (Tier B JSON schema, from `scripts/rca_baselines.py`)

Per method ∈ {naive, ewc, lwf, er, der_pp, colar}, `results/rca/dil_<m>_seed42_rca.json`:

- `boundaries[b].eval[t][mask]` for mask ∈ {`full`, `text_only`, `text_layout`,
  `image_layout`} → `{f1, per_class, confusion}` (confusion: 9×9 token counts, gold=row,
  BIO tags of HEADER/KEY/VALUE/OTHER + O; DIL = funsd→sroie→cord, fixed head).
- `boundaries[b].displacement_by_group` (component vocab incl. the three modality embeds,
  `attn_qkv`, `ffn`, `classifier`, …) and `displacement_by_depth`
  ({input, early, mid, late, head}), b ≥ 1, Fisher-weighted vs previous boundary.

**Definitions.**
- At-learning F1 of task t under mask m: `boundaries[t].eval[t][m].f1`.
- Drop(t, m) = at-learning − `boundaries[-1].eval[t][m].f1` (F1 points; old tasks t < 2).
- **Mask-validity guard (pre-registered):** a mask is interpretable for task t only if its
  at-learning F1 ≥ 20; below that, retention under it is noise and is excluded (expected
  risk: `image_layout`, where zeroed text may floor performance).
- **Method strata:** high-forgetting stratum = methods with FULL-mask mean Drop ≥ 10
  (expected: naive, lwf, ewc); replay stratum (er, der_pp, colar) acts as control —
  predictions of extinction/flow must be ABSENT there.

---

## H1 — Modality-asymmetric drift (forgetting correlates with modality)

**Mechanism.** The three input pathways receive asymmetric gradient traffic during
sequential fine-tuning; the pathway that moves most carries the forgetting. Two directional
sub-hypotheses:
- **H1a (text-drift):** token-classification loss is dominated by text-token gradients →
  text pathway overwritten most → old-task F1 under `text_only` drops MOST;
  `displacement_by_group.text_word_embed` ≫ `layout_2d_pos_embed`, `image_patch_embed`.
- **H1b (layout/vision-drift):** under-trained non-text pathways decay/detune →
  `image_layout` drops most, text-including masks track FULL.

**Prediction (eval-space).** In the high-forgetting stratum, mean Drop over valid old-task
masks differs by ≥ 10 F1 points between the max- and min-drop mask, consistently in the same
direction for ≥ 2 of the 3 high-forgetting methods.
**Prediction (weight-space).** The same direction shows in `displacement_by_group`: the
implicated modality embed group's displacement exceeds the other two embed groups
(order-of-magnitude comparison only — group sizes differ; not a calibrated effect size).
**Falsification.** Mask spread < 10 points (masks degrade in lockstep) in the
high-forgetting stratum ⇒ forgetting is modality-independent → supports H2.

## H2 — Head/label-space interference (modality-independent; schema/frequency-driven)

**Mechanism.** Extinction of KEY/HEADER is a classifier-readout phenomenon: the new task's
label distribution overwrites logit geometry for classes it lacks (CORD emits no KEY spans),
regardless of which modality produced the features. Per F1/F2 priors, damage lives in the
head, so ablating input streams changes little.

**Prediction.** (i) KEY/HEADER extinction is mask-uniform: in naive/lwf at the final
boundary, KEY F1 < 5 under EVERY valid mask for funsd/sroie. (ii) Confusion rows for gold
B/I-KEY and B/I-HEADER lose > 50% of their mass to VALUE-tag columns and O — with VALUE
capture exceeding chance — rather than recovering under any single mask. (iii) Replay
stratum shows none of this (KEY row mass stays on-diagonal).
**Falsification.** KEY/HEADER survives (F1 ≥ 20) under at least one mask while dead under
FULL ⇒ the loss is modality-tied (H1), not head-generic.

## H3 — Late-layer cross-modal integration drift

**Mechanism.** Cross-modal integration in a single-stream encoder happens in the late
blocks (CKA prior: drift monotone with depth). Continual updates concentrate displacement
there, degrading integrated (multimodal) readout more than any single-stream readout.

**Prediction.** (i) `displacement_by_depth`: late > mid > early for every high-forgetting
method at every boundary (head excluded — priors already say head dominates; H3 is about
the trunk profile). (ii) Masks that need integration most (`image_layout`, which must bind
layout+vision without lexical shortcut) show larger Drop than `text_only`, where valid.
**Falsification.** Depth profile flat or front-loaded (early ≥ late) in ≥ 2 high-forgetting
methods, or `image_layout` Drop ≤ `text_only` Drop.
**Caveat (pre-registered):** colar's backbone freezes after task 0 — its
`displacement_by_depth` is head-only from boundary 1→2 and is excluded from trunk-profile
claims (absent keys = not trainable, per the JSON's own `caveats`).

## H4 — Recency/logit bias (one-boundary collapse is a readout shift, not feature loss)

**Mechanism.** Training task t+1 snaps the shared head's logits toward t+1's label
distribution; old-task features survive (head-refit oracle: 39.6 → 55.3 pooled) but tokens
get read out as the new task's dominant labels. Explains immediate_share ≈ 1 for
naive/lwf/ewc and the partial recovery (+7.9) when later tasks re-cover old label modes.

**Prediction.** (i) At boundary b, old-task gold tokens' off-diagonal confusion mass under
FULL concentrates in the label columns dominant in task b's training data (not scattered:
top-2 new-task-dominant columns absorb > 50% of off-diagonal mass). (ii) The effect appears
at the first boundary after training (matching immediate_share ≈ 1) — same confusion
structure under all masks (it is a head effect). (iii) Replay stratum: no such concentration.
**Falsification.** Off-diagonal mass spread ~uniformly across wrong labels, or concentrated
in O only with no dominant-label capture, or head-refit-style recovery absent.

### H2 vs H4 disambiguation (pre-registered — they overlap in DIL)

In DIL the new task's dominant label often IS VALUE, so both predict KEY→VALUE flow.
Disambiguate by: (a) **HEADER at boundary 1** — sroie's distribution vs the schema-wide
VALUE prior differ enough to separate "new-task-majority capture" (H4) from "global
frequency capture" (H2); (b) **the gradual term** — H4 predicts partial recovery of
old-task F1 when a later task re-exercises the lost label (a2 gradual = +7.9 for naive);
H2 predicts monotone loss for classes absent from every later task; (c) **O-capture share**
— H2 tolerates mass going to O; H4 requires it going to *labels trained last*.
H2 and H4 are NOT mutually exclusive — both can hold at different loci (H2: which classes
die; H4: where their mass goes).

## Secondary question QA — domain-similarity interference (exploratory, not a hypothesis)

Naive's final row is funsd 17.7 / **sroie 3.7** / cord 97.3 — the MIDDLE task is most
forgotten, violating pure recency. SROIE and CORD are both receipts; candidate explanation:
domain overlap makes CORD's updates overwrite SROIE's readout hardest. Check: at boundary 2,
is sroie's off-diagonal confusion more CORD-shaped than funsd's? Report descriptively in
Tier C; no pre-registered threshold.

## Adjudication protocol

1. Wait for all 6 JSONs (`run_rca_baselines.sh` chain; smoke check first: naive
   final-boundary FULL AA must land ≈ 40 ± 3, cf. head_refit_oracle 39.56 — else stop and
   debug before adjudicating).
2. `scripts/rca_synthesize.py` (Tier C, to be written) computes: Drop(t, m) table per
   method (`c_modality_deltas.csv`), confusion-flow decomposition, depth-displacement
   profiles joined vs a2 `immediate_share` (`c_displacement_vs_signature.csv`).
3. Verdict per hypothesis: SUPPORTED (all pre-registered predictions pass) / PARTIAL
   (direction holds, thresholds miss) / REFUTED (falsification criterion met). Recorded in
   `docs/RCA_FORGETTING_BASELINES_2026-07.md` with the actual numbers.

## Caveats

- n=1 seed (42), one scenario (dil), one backbone (LayoutLMv3) — all verdicts provisional;
  multi-seed confirmation only if a verdict is load-bearing for the paper.
- Prompt (l2p/dualprompt/coda) and LoRA families are OUT of Tier B: the uniform
  `eval_under_mask` probe is dishonest for prompt methods (frozen backbone;
  `forward_with_prompts` has no `modality_mask`; prompt selection would see full-modality
  queries). o_lora WOULD be drop-in compatible (PEFT wraps the inner model under the plain
  forward path) — noted for a possible extension, not scoped now. Prompt-family inference
  rests on Tier A signatures only (immediate_share 0.44–0.58, ongoing leak).
- Fisher-weighted displacement across groups of different parameter counts is compared as
  profile shape, not calibrated magnitude.
