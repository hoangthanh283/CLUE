# Senior-reviewer read of the CVPR draft, verified against the result files (2026-11-10, v2)

Supersedes the v1 read of the same date, which was done from the PDF text alone. This pass re-read the
source, recomputed the paper's arithmetic, and checked claims against `results/` and the code.

**Recommendation.** CVPR: borderline, leaning weak reject (3/6). NeurIPS/ICML: 4/10, soundness 2,
presentation 2, contribution 3. The readout finding and SBR-Recall are accept-quality; the paper around
them over-claims in ways a careful reviewer will find, and four of those are factual, not stylistic.

## Summary of the submission
Diagnosis-first study of forgetting in continual key-information extraction (LayoutLMv3/LiLT/BROS/BERT;
FUNSD, SROIE, CORD). Claims: (1) forgetting is head/late-layer localised and architecture-general, and
protecting the locus relocates it; (2) six classical remedies order as predicted; (3) five buffer-free
families fail for one cause; (4) CoLaR, per-document SVD latent replay, matches replay on DIL;
(5) class-incremental "floor" is a scoring artefact plus a shared-background contradiction, fixed by a
session-blocked readout (SBR 88.0) and per-session LoRA recall (SBR-Recall 92.8).

## What I recomputed and found correct
EWC sign-flip deltas on Mixed (−11.9/−2.6/+3.5/+5.2, 17-point range); joint-to-naive gap spread 2.5;
68/72 cells and the four missing cells; LwF +3.9/+5.0 on BROS/BERT; +27 single-variable control
(63.8 − 36.7); 2.7× (163/60 MB); Table 3 means against `results/rescore`. Reporting of nulls and
seed counts is unusually candid.

## Critical issues (each threatens a stated conclusion)

**C1. Finding 1's significance claim tests the quantity the paper says must not be read as forgetting.**
"A per-component test rejects the null of uniform displacement with the head dominant (p < 0.01)."
The test in `doccl/pilot/analyze.py::component_hypothesis_test` runs on `metric == "fisher"`, the
per-group Fisher *importance*, which the Setup paragraph says "peaks at the head by construction and is
never read as forgetting". In `results/pilot/findings_summary.md` (n = 6, Bonferroni α′ = 0.0042) the
classifier is significant against 3 of 12 rival groups (ffn, image_patch_embed, text_word_embed,
p = 0.0022) and indistinguishable from layernorm and "other" (p = 1.00), cls_pooler (0.71), text_attn
(0.55). The verdict is "reject" because *any* rival rejects. This does not support "head dominant".

**C1b. The architecture-generality test has no power to fail.** The location test reports cosine
distance 0.000 between every pair of L1-normalised profiles. Because D_g is Fisher-weighted and Fisher is
40× larger at the head by construction (0.22 vs 0.005), every profile is near one-hot at the head, so a
permutation test cannot reject (p = 0.38, 0.51 is the expected outcome under any backbone). "Fails to
reject a shared location" is an argument from ignorance. What survives is descriptive: the CKA depth
gradient on seven conditions. Fix: test the encoder-depth profile with the head excluded, use unweighted
or parameter-count-normalised displacement, and report effect sizes.

**C2. The interventional support for the locus is confounded by replay.** "The same budget concentrated
on the head/late locus recovers 84.7, head-only 86.1, late-only 86.2" comes from the DocCL
`target_depth` ablation, whose config has `use_replay: true, buffer_size: 200`. Plain ER with the same
200 documents scores 87.9. So the probe shows that a uniform penalty hurts (42.6), not that a
locus-targeted penalty recovers anything; and as written it contradicts the paper's own claim that
parameter-space protection cannot work. The buffer must be disclosed and a no-replay arm reported.

**C3. The two universal claims are contradicted by the paper's own best result.** Abstract: "a
buffer-free fix cannot work". Section 4: "under architectural constraint the total forgetting is
redistributed, not removed". SBR-Recall on CIL-CORD uses no bank, freezes the base, and reaches 92.8
with BWT ≈ 0: a buffer-free, architecture-constrained method with no forgetting. (The image-side SLCA
result is a second counter-example.) The claims hold for methods that keep updating a *shared* trunk and
head without added capacity; they must be scoped to that, and Recall on DIL is the decisive missing test.

**C4. Table 3's comparison is confounded, and some baseline cells are numerical artefacts.**
- Precision. Baselines ran with fp16 + GradScaler (the log shows the scaler; the caption said bf16),
  SBR rows fp32. Five runs collapse to exactly 0.0 with the model dead from the collapse point on
  (LwF CIL-CORD s123, LwF Mixed s7, IS3 Mixed s42, CPFD CIL-CORD s7, CPFD Mixed s42). In the fp32
  legacy runs LwF never collapses (CIL-CORD 18.7/19.0/18.8, Mixed 36.5/35.1/34.1) and IS3 Mixed is
  30.3/28.8/31.2. Three of the five zeros are therefore mixed-precision failures of distillation
  losses, reported as method results (LwF 12.3 ± 10.7).
- Noise floor. The same method and seed differ between precisions by −2.4 (SBR s42: 87.6 fp32 vs
  85.2 AMP) and +5.1 (s7: 86.3 vs 91.4). Run-to-run variation is as large as the reported seed s.d.,
  so "+1.7 from LexSlot" and "Recall vs joint" are inside noise.
- Budget. Baselines use a 100-epoch early-stop ceiling, the method rows `method.epochs=10`. Setup says
  100 for everything; the Table 2 caption says 10 for everything. Both cannot be right.
- Memory. ER/DER++ hold 200 documents in total; CoLaR/SBR hold 50 per session, i.e. 250 on CIL-CORD
  and 300 on Mixed.
- Seeds. Two for most baselines, three for the method.

**C5. Model selection on the test split, for every dataset.** `scripts/train.py` passes the current
task's *test* loader as the early-stopping signal unless `training.val_frac > 0`; CORD's validation
split is never used. The Setup paragraph says validation F1 is used with FUNSD and SROIE as the
exceptions, which is inaccurate; Limitations states it correctly. The headline cells need the
`val_frac` re-run in the paper, not promised.

**C6. The method claim lacks its two obvious controls and its closest prior art.** (i) SBR's readout on
raw replay (ER + SBR): without it the paper cannot say whether CoLaR matters to SBR. (ii) An isolation
baseline with task inference for Recall. (iii) SBR is multi-head learning with a learned per-task
reject logit, the "within-task prediction × task-identity/OOD detection" decomposition of Kim et al.
(NeurIPS 2022) and its descendants (MORE, CLOM); Recall is adapter-per-task with task inference (EASE,
InfLoRA, HiDe-Prompt, MoE-Adapters). The genuine insight, that token-level IE supplies true negatives
for every block for free, should be stated against that line. (Citations named from memory; verify
before adding.)

## Important issues
- **"At the oracle."** Abstract and contribution 4 say SBR is at the oracle on CIL-CORD: 88.0 ± 1.9 vs
  91.9 ± 1.3 is a 3.9-point gap. On Mixed SBR *exceeds* the oracle by 2, and Recall ties it; an "upper
  bound" that methods exceed is an under-trained joint baseline (patience-2 early stopping; the LiLT
  footnote already concedes this). Rename it and tune it.
- **Restricted scoring needs its argument and a companion metric.** All CIL-CORD sessions are scored on
  the same 100 receipts under different masks, so each error is charged exactly once across sessions;
  the paper never says so. The cleanest number is absent: final-model entity-F1 on the unmasked,
  full-label CORD test set, which is comparable with the literature.
- **Two "Independent" numbers** for LayoutLMv3/DIL: 88.2 ± 1.0 (Table 1) vs 88.79 ± 0.55 (Table 2).
- **SBR on DIL** is said to reduce to CoLaR but scores 86.2 ± 1.3 against CoLaR's 87.75 ± 0.48.
- **Recall's costs and scope.** Base frozen after session 0, so session order matters and is untested;
  one upper-trunk pass per session at inference (≈ 3.7× compute at five sessions); 7.1M parameters per
  session; Mixed is single-seed and loses to SBR.
- **Privacy asserted, storage ambiguous.** Layer-4 latents plus boxes and labels are not private by
  construction; "2.7× less storage" is against raw *latent* replay, not ER's raw documents.
- **Framing.** Title and Fig. 1 present CoLaR and depict a component the caption calls a null. The
  method that carries the paper has no figure, no loss equation and no algorithm.
- **Sample size of the benchmark.** Test sets of 50 (FUNSD), 100 (CORD), 347 (SROIE) documents; no
  confidence intervals; 1–2 point differences on FUNSD are within sampling noise.

## Minor
Legacy-scored CIL columns measure what the paper calls an artefact. In-house names (LexSlot, LexMem,
HGT, CUBER, CoLaR-R) are undefined in the body. "Registered" needs a registry or should read "specified
in advance". "Exact retention" against BWT +0.6/+0.4/−0.1 should read "near-exact". Mean ± s.d. with
n = 2 is not informative; print both values. Code "on acceptance".

## What would change my score, in order
1. Text only: scope the universal claims (C3); replace the p < 0.01 sentence with what the test shows
   (C1); disclose the replay buffer in the consolidation probe (C2); correct the early-stopping
   description (C5); "at the oracle" only where true; reconcile the Independent numbers.
2. Cheap GPU (about three days locally): fp32 re-run of the Table 3 baselines that matter (joint, ER,
   DER++, LwF, CL-LoRA; three seeds); ER + SBR control; `val_frac = 0.1` re-run of SBR, Recall, ER and
   joint; Recall on DIL and on Mixed seeds 7/123; one Recall session-order permutation; full-label
   final F1.
3. Larger: one isolation baseline with task inference; an inversion probe or a softened privacy claim;
   a method figure and loss for SBR/Recall replacing the teaser.

With 1 and 2 done this is an accept at CVPR. Without them, C1–C4 are each sufficient for a reject from
a reviewer who checks.

## Corrections applied during this review (errors introduced in earlier edits)
- "SBR-Recall ... above the joint oracle on every seed" was false. Per seed (Recall vs joint, same
  seed): 92.69 vs 92.84, 93.37 vs 90.48, 92.44 vs 92.43. Paper and thesis now say "at the oracle".
- Table 3 caption said baselines use bf16 and SBR rows fp32. The baselines use fp16 + GradScaler, and
  SBR's Mixed cell is from the same mixed-precision runs. Caption corrected in paper and thesis.
