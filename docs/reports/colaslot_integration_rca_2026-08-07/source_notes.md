# CoLaR + LexSlot integration RCA source notes

## Decision frame

- Question: why did full-capacity CoLaSlot fail to improve CoLaR, and what is the smallest
  integration that directly addresses the verified failure modes?
- Audience: technical research/implementation.
- Controlling comparison: fresh matched seed-42 DIL runs at k4/d50/r128, maximum 10 epochs,
  identical data, batch size 1, gradient checkpointing, and early stopping.
- Primary decision metric: final average accuracy (AA). Guardrails: final domain F1 and average
  forgetting (AF).

## Source authority and conflicts

- The fresh paired metrics control causal interpretation because they share code, seed, data, and
  runtime. The historical CoLaR 87.58 row remains benchmark context but is not used to estimate the
  slot effect because the fresh CoLaR control is 86.52.
- Saved `metrics.json` and `per_class_f1.json` own reported scores. Live code owns implementation
  behavior. `STATE.md`, `ROADMAP.md`, and the July KT note provide historical intent only.

## Recomputed checks

- AA delta: 86.4820005 - 86.5239595 = -0.0419590 points.
- Final domain deltas: FUNSD -0.8812730; SROIE +1.6990675; CORD -0.9436715.
- Forgetting: CoLaR [0.4106926, 6.2692840, 0.0]; CoLaSlot [2.9118196, 4.3730383, 0.0].
- Slot ownership at each task follows `n_slots_head=48`, `n_slots_late=18`, `n_tasks=6`.
  `LexSlot.before_task` claims 8 head and 3 late slots per task, but computes the mask while all
  remaining slots still have owner -1. `slot_trainable_mask` assigns owner -1 a mask of 1.0.
  Therefore task 0 trains 48/48 head and 18/18 late slots, not only its claimed block.
- The raw and masked routing probes use every DIL train and eval document. Both use the same
  task-signature aggregation and top-1 cosine rule; only padding/special-token masking changes.

## 2026-08-08 repaired-integration evidence

- Cheap matched d5/r64/5-epoch CoLaR: AA 60.7895952, AF 41.0471861, final
  [44.3218852, 41.3393964, 96.7075038].
- Cheap matched CoLaSlot-R: AA 74.8140276, AF 20.5990209, final
  [64.3088553, 64.2679901, 95.8652374]. Delta versus CoLaR: AA +14.0244324,
  AF -20.4481652, final [+19.9869701, +22.9285936, -0.8422665].
- Acquisition deltas are [+0.2408348, +1.7783985, -0.8422665]. Subtracting those from
  final deltas attributes [19.7461353, 21.1501951, 0.0] points to reduced post-learning
  loss. This is 97.2% of the summed final-domain gain.
- CORD has zero forgetting in both methods. Its -0.8423 regression therefore enters at
  acquisition, not retention. The same last-task penalty exists with routing disabled:
  -0.5758 in the cheap ungated run and -0.9437 in the full ungated run. This makes slot/base
  co-training or direct current-slot contribution the primary locus; routing errors are a
  secondary amplifier, not a sufficient root cause.
- CoLaSlot-R versus cheap ungated CoLaSlot changes the final row by
  [-3.1518919, +12.0434138, -0.2665129]. Hard routing reallocates the retention benefit
  toward SROIE rather than uniformly improving it.
- Every saved `metrics.json` matrix was checked against its sibling `matrix.npy`; all five
  matched runs agree, including NaN positions.

## Training-only routing calibration

- Five-fold out-of-fold calibration covers 1,575 training documents and 2,350 routing
  decisions across the two-domain and three-domain stages.
- At margin 0.05, 2,173 decisions are accepted (92.4681% coverage) with 17 errors
  (99.2177% accepted accuracy). CORD contributes 16 of those errors among 698 accepted
  decisions, or 97.7077% accepted accuracy.
- The zero-error margin 0.326 accepts only 421 CORD decisions and no FUNSD or SROIE
  decisions. No threshold on the 0.001 grid satisfies both at least 50% coverage and at
  least 99% accepted accuracy for every stage/domain.
- This calibration uses training documents only and is not a downstream F1 estimate. It
  closes the global scalar-margin follow-up; it does not rule out a different router or a
  design that makes wrong routes non-destructive.

## 2026-08-14 post-task refit experiments

- Both successors use the matched DIL/LayoutLMv3 seed-42 k4/d5/r64/5-epoch cell and save
  slots-on plus same-state slots-off evaluation after every task. Their slots-off matrices
  are identical: stage 0 [87.702902], stage 1 [63.196126, 81.639566], and stage 2
  [67.649424, 52.670441, 96.939556]. This directly isolates the residual effect.
- CoLaSlot-RF slots-on: AA 72.419807, AF 24.511302, final
  [67.649424, 52.670441, 96.939556]. It gives a temporary +0.121 FUNSD change at stage 1
  and exactly zero final change. Positive refit losses are approximately 1e-5: the replay
  hard labels are already satisfied, so the residual receives no useful drift signal.
- CoLaSlot-RA stores acquisition-time replay logits and anchors the owner residual to them.
  Slots-on: AA 71.993560, AF 25.150673, final [67.362784, 51.678339, 96.939556].
  Relative to the same-state fallback, deltas are [-0.286639, -0.992102, 0.000000],
  AA -0.426247, and old-domain mean -0.639371. Positive anchor losses of 2.823--5.703
  confirm that RA fixes RF's missing-gradient failure.
- The preregistered gate requires AA >= +0.5, old-domain mean >= +1.0, and every domain
  >= -0.5. RF fails the two improvement clauses; RA fails all three because SROIE is
  -0.992. RA memory is 3,404,280 bytes versus RF 3,266,040 (+138,240, +4.23%).
- Per-class comparison localizes RA's final SROIE loss to false positives: micro precision
  changes -1.186 while recall is identical; KEY and VALUE F1 change -0.125 and -1.028.
  FUNSD also changes precision and recall in opposing class-specific directions. The anchor
  therefore has signal but overgeneralizes beyond five stored documents.

## Updated successor decision

- CoLaSlot-RF, CoLaSlot-RA, and CoLaSlot-RO are completed no-gos, not recommendations.
  Together they rule out hard-label head residuals at post-task and online timings, while
  dense acquisition anchors provide unsupported false-positive corrections.
- A further residual is permitted only as a support-bounded falsification test: validate
  entity/class corrections, require non-negative precision, and explicitly drive current,
  foreign-owner, and non-entity support toward zero. This gate must pass before any full
  d50/r128 or multi-seed run. Adding rank, router, or anchor-weight sweeps is not justified.

## 2026-08-15 online hard-label experiment

- CoLaSlot-RO uses the same d5/r64/5-epoch seed-42 cell and the exact RF/RA slot-free
  trajectory. Slots-on AA is 72.410336 versus same-state AA 72.419807 (-0.009471);
  AF is 24.525508 versus 24.511302. Final domain deltas are
  [-0.054351, +0.025940, 0.000000], and old-domain mean delta is -0.014206.
- The only earlier-stage effect is -0.197335 FUNSD after task 1. At final evaluation,
  recall is unchanged on every domain; micro precision moves -0.128 FUNSD and +0.031
  SROIE. The residual is functionally negligible rather than a hidden redistribution.
- Owner-specific online CE averages are 9.20e-4 (task 1/owner 0), 4.47e-5
  (task 2/owner 0), and 2.19e-3 (task 2/owner 1). RO and RF both use 3,266,040 bytes and
  2,424.84 MB peak GPU memory; RO adds 1,615 seconds (26.9 minutes). JSON and matrix.npy
  agree. The strict gate fails both improvement clauses.

## 2026-08-15 primary-literature update

- The closest published document benchmark found is *Universal Graph Continual Learning*
  (TMLR 2023), which converts CORD/SROIE/WildReceipt into graph-unit node classification.
  Its graph model, class-incremental task construction, labels, and average-performance metric are
  not comparable to this LayoutLMv3 domain-incremental token-classification protocol. Any SoTA
  statement here must therefore be explicitly protocol-specific and supported by matched seeds.
- *Mixture of LoRA Experts for Continual Information Extraction with LLMs* (Findings of EMNLP
  2025, https://aclanthology.org/2025.findings-emnlp.718/) reports that token-level expert
  selection and distillation of router/key distributions outperform sentence-level selection in
  continual IE. LexSlot already has token-dependent slot activations, but chooses the owner block
  from one document-level OCR cosine; changing that owner router is a separate hypothesis, not a
  remedy for an unsupported residual.
- *PASs-MoE* (ACL 2026, https://aclanthology.org/2026.acl-long.1474/) identifies router/expert
  misaligned co-drift and derives routing weights from each low-rank pathway's own activation
  energy. `LogitSlots.values` provides the analogous input-side pathway directions, so a
  pathway-energy router is mechanically available without adding parameters. It is secondary:
  the matched RA result already shows harm under owner-forced training and localizes it to false
  positives, while the historical CORD penalty survives routing removal.
- *Layerwise Proximal Replay* (ICML 2024,
  https://proceedings.mlr.press/v235/yoo24a.html) stabilizes replay optimization by constraining
  changes to past hidden activations. This is not the next test: CoLaR's slot-free trajectory is
  already the matched control, and a transformer-wide optimizer/preconditioner would change the
  base method rather than isolate the LexSlot integration.
- *Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation*
  (ICML 2026 workshop, https://arxiv.org/abs/2606.18024) predicts old-task output drift from
  cross-task kernels and finds that its energy concentrates in a small number of output-space
  modes. The frozen-linear-head result is exact and the nonlinear case is a local approximation.
  This directly motivates a conditional successor to hard-label RO: measure each base update's
  replay-logit drift and train the owner slot to cancel that one-step functional change, while
  keeping the residual zero on current and foreign support. It does not justify more slot rank.
- Replay-selection proposals are also deprioritized. The project has already observed
  redistribution rather than a reproducible AA gain from k-center/class-balanced/reweighted
  replay, so another selector does not address the measured residual false-positive mechanism.

### CoLaSlot-FD preregistration

Protocol correction before adjudication:

- The first FD launch was stopped during task 1 and is audit-only. Random CoLaSlot initialization
  and the LexSlot pass over a shuffled DataLoader both advanced the global Torch RNG, changing
  subsequent batch/dropout order relative to CoLaR. The partial run task-0 F1 (87.7029) therefore
  differed from the matched CoLaR control (86.9223).
- CoLaSlot now restores CPU Torch RNG after both auxiliary operations. A regression test uses a
  real DataLoader with shuffle enabled and pins initialization plus prepass neutrality. The corrected
  run must reproduce the CoLaR task-0 trajectory before its slots-on/off result is eligible for the
  gate; thresholds and permitted follow-ups remain unchanged.

1. Capture each sampled owner's combined centered logits in eval/no-grad mode immediately before
   the normal CoLaR optimizer step. After the step, freeze CoLaR and train only that owner's
   existing head-slot rows to cancel the measured decision-logit drift on entity tokens.
2. Give same-owner O tokens, the current batch, and foreign-owner replay features a zero-residual
   target. Reuse RO's optimizer setting, hard lexical route, and memory; add no rank, slots,
   teacher store, or sweep.
3. The matched cheap gate remains AA >= +0.5, old-domain mean >= +1.0, and every domain >= -0.5.
   Failure closes the functional residual family. A full d50/r128 run and seeds 7/123 remain
   forbidden until this gate passes.
4. Pathway-energy self-routing remains an inference-side ablation only after FD passes; it cannot
   rescue an unsupported or inert residual.

## Historical successor preregistration

- Recommended concept: **CoLaSlot-RF**, a retention-only, post-task head-slot refit.
  CoLaR trains with slot reads disabled, so its main optimization path and current-domain
  acquisition are unchanged. After each task, CoLaR is frozen and owner-specific head
  residuals are fit on current-base replay features. At evaluation, only slots owned by
  prior tasks may contribute; a route to the current task is a zero-residual CoLaR fallback.
- Refit old residuals against their owner replay documents and constrain their delta toward
  zero on current/foreign training documents. This moves safety from an infeasible global
  confidence threshold into the residual itself and re-anchors the slot after shared-head
  drift.
- Existing code provides all required seams: zero-init head residuals in
  `lexslot_memory.py`, current-state replay features in CoLaR, replay-owner tracking in
  `colar_cb.py`, and the proven train-mode refit pattern in `lexslot_fm.py`. No new router
  family or storage object is needed for the first gate.

## Chart map

- `entity_delta_chart`: comparison/ranking, horizontal signed bar; six entity-domain F1 deltas;
  proves the aggregate is redistribution. Hard two-root palette with signed labels; semantic table
  fallback retained.
- `routing_accuracy_chart`: grouped comparison bar; three domains under raw versus masked token
  signatures; proves padding dominates the current router and that the minimal correction is viable.
- `scalar_calibration_chart`: grouped comparison bar; five stage/domain cells under raw
  accuracy, margin-0.05 coverage, and accepted accuracy; shows why the scalar gate cannot
  satisfy coverage and reliability together. Relaxed three-category palette; exact table retained.

## Required technical-report structure mapping

- Title: `title` block.
- Technical summary: `summary` block.
- Key findings with visual evidence: redistribution and router sections plus two charts and two
  exact tables.
- Scope/data/metric definitions: `scope` block.
- Methodology/model specification: `method` and `design` blocks.
- Limitations/robustness: `limitations` block.
- Recommended next steps: `experiment` block.
- Further questions: `questions` block.

## Validation assessment

- Ready to share as an RCA and experiment recommendation, with caveats.
- Calculations were independently recomputed from both saved metric files.
- Cheap/full `metrics.json` matrices were reconciled to all sibling `matrix.npy` files, and
  calibration counts were recomputed from sample sizes and coverage.
- The routing probe is descriptive and training-free; it validates separability, not downstream AA.
- Only seed 42 has been run for CoLaSlot and CoLaSlot-R. The repaired run uses the cheap
  d5/r64 gate, while the full paired run covers only ungated CoLaSlot; cross-budget comparisons
  are treated as triangulation, not a matched effect estimate.
- No slots-on versus slots-off evaluation was saved from the repaired checkpoint. The evidence
  localizes CORD damage to acquisition and shows it survives router removal, but cannot yet
  separate direct slot readout damage from training-time shared-weight co-adaptation.
- CoLaSlot-RF and CoLaSlot-RA have run and failed the preregistered cheap gate; see the
  2026-08-14 experiment section above.
- No trend chart was used because the evidence consists of discrete method/domain comparisons.
