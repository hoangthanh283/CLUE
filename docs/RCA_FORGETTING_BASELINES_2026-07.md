# RCA: root causes of forgetting in multimodal doc-IE baselines (Tier C verdicts)

**Date:** 2026-07-17. **Adjudicates** the pre-registered hypotheses in
`docs/RCA_HYPOTHESES_2026-07.md` (registered 2026-07-16 BEFORE any Tier B data) against
Tier A artifacts (`results/rca/a1_*`, `a2_*`, 3 seeds) and Tier B instrumented runs
(`results/rca/dil_{naive,ewc,lwf,er,der_pp,colar}_seed42_rca.json`; synthesis
`results/rca/c_*.csv`). Decision rules applied **as written**; every number below is in a
named artifact.

**Smoke check passed:** naive final FULL AA 37.9 (pre-registered bar 40 ± 3, cf.
head-refit oracle 39.6). Strata (FULL mean old-task Drop, `c_modality_deltas.csv`):
high-forgetting = naive 76.4, lwf 73.9, ewc 43.1, colar 28.9; replay control = er 2.2,
der_pp 1.2. ⚠ colar ran at its CANONICAL config (k8/d5/r64), not the headline
k4/d50/r128 — it lands in the high stratum here; do not compare to the 87.6 grid row.

## Verdicts

| Hypothesis | Verdict | One-line reason |
|---|---|---|
| H1 modality-asymmetric drift | **PARTIAL** | Eval-space spread passes (≥10 pts, consistent), but BOTH directional mechanisms fail in weight space |
| H2 head/label-space interference | **SUPPORTED** | All three predictions pass |
| H3 late-layer integration drift | **REFUTED** | Trunk displacement is front-loaded (early ≫ late) in naive/lwf/der_pp/er |
| H4 recency/logit bias | **SUPPORTED** | Off-diagonal mass tracks the LAST-trained task's label distribution; replay stratum clean |

### H1 — PARTIAL (formal pass, mechanism fails)

Eval space (`c_modality_deltas.csv`, valid old-task masks): spread between max- and
min-drop mask = naive 52.3, lwf 51.3, ewc 33.8, colar 29.6 — all ≥ 10, same direction
(text-including masks `full`/`text_layout` drop 43–76 pts; `image_layout` drops only
12–24). Formally the pre-registered threshold passes.

Weight space (`c_displacement_vs_signature.csv`) kills both sub-mechanisms:
- **H1a (text-drift): REFUTED.** `text_word_embed` displacement is 2–4 orders of
  magnitude SMALLER than the layout/image embeds (naive b1: text 1.9e-10 vs layout
  7.0e-8, image 6.6e-8), and the head dwarfs all embeds (2.3e-5, ~300× the largest embed).
- **H1b (layout/vision decay): REFUTED.** predicted `image_layout` drops most — it drops
  LEAST everywhere.

Caveat (floor effect): `image_layout` at-learning F1 is only ~20–25, so its small
absolute drop partly reflects less room to fall. Honest synthesis: the modality
correlation is real but lives in WHAT the head loses (text-carried evidence — the
dominant F1 mass), not WHERE parameters drift. H1-as-mechanism is superseded by H2+H4.

### H2 — SUPPORTED

1. KEY extinction is mask-uniform (per-mask `per_class` in the Tier B JSONs): naive and
   lwf final-boundary KEY F1 = **0.0 under every valid mask** on funsd AND sroie. The H1
   escape hatch (survival ≥ 20 under some mask) never fires.
2. Confusion flow (`c_confusion_flow.csv`, final boundary, gold KEY/HEADER, old tasks):
   naive/lwf off-diagonal mass → O+VALUE = **1.00** (top-2 columns absorb 1.00);
   ewc 0.91.
3. Replay stratum clean: er retains 0.83, der_pp 0.89 of gold KEY/HEADER mass
   on-diagonal; er's final KEY F1 85.8 (funsd) / 91.3 (sroie).

### H3 — REFUTED

Trunk `displacement_by_depth` (head excluded, colar excluded post-freeze per
pre-registered caveat): naive early 0.70/0.64 vs late 0.03/0.03 (b1/b2); lwf 0.69/0.56
vs 0.05/0.09 — **front-loaded**, meeting the falsification criterion (≥2 high-forgetting
methods early ≥ late). ewc is the one late-heavy method (late 0.48/0.35 — its Fisher
penalty anchors early weights). Tension worth a paper sentence: weight displacement is
front-loaded while the CKA prior says functional drift grows with depth — small early
weight motion amplifies through depth; "where weights move" ≠ "where function changes",
and neither is where the damage is read out (the head).

### H4 — SUPPORTED

1. Boundary 1 (just trained SROIE, ~99% O): old-task (funsd) gold KEY off-diagonal mass
   → **O at 0.96–0.98 top-1** for naive/lwf/ewc — O IS SROIE's dominant label, i.e. the
   flow tracks the last-trained distribution, not a static prior.
2. Boundary 2 (just trained CORD, VALUE-rich): flow rotates to VALUE capture — e.g.
   colar sroie-KEY → I-VALUE 0.71 + B-VALUE 0.23 (to_VALUE = 0.93); naive/lwf
   O+VALUE = 1.00.
3. Replay stratum: no concentration collapse (retained 0.83–0.89).
4. Disambiguators (pre-registered): (b) a2 gradual term = **+7.9 recovery** for naive
   when CORD re-exercises VALUE, while KEY (absent from CORD) stays at 0.0 — monotone
   loss exactly for never-re-exercised classes; (c) captured mass goes to labels trained
   LAST, rotating per boundary (O → VALUE) — the H4 signature, on top of H2's
   schema-frequency substrate. H2 and H4 co-hold at different loci, as pre-registered.

## Root-cause statement (n=1 seed, dil, LayoutLMv3 — provisional)

**Forgetting in these baselines is a readout-recency phenomenon on a class-asymmetric
substrate.** The shared head's logit geometry snaps to each new task's label
distribution (H4); classes the later tasks never re-exercise (KEY, HEADER) are
extinguished mask-uniformly (H2, Tier A: extinction to F1=0.0 with VALUE surviving);
trunk weight drift is front-loaded but functionally minor (H3 refuted), and no modality
pathway is the culprit (H1 mechanisms refuted) — the "multimodal correlation" reduces to
text being the head's dominant evidence stream. This is fully consistent with F1
(head-localized), the head-refit oracle (features survive: pooled 55.3 vs trained-head
39.6), and F3 (only replay grounds the head).

**Falsification tests for the statement:** (i) a frozen-trunk naive run must reproduce
near-identical extinction (head-only cause) — cheap, not yet run; (ii) logit-bias
correction alone (recalibrate head priors per task, no feature change) must recover a
large fraction of the one-boundary drop; (iii) multi-seed repeat of Tier B for the
load-bearing numbers.

## Method-design implications (input to the next brainstorm)

1. **Attack the readout, not the trunk.** Consistent with the queued read-side direction
   (colar_knn kNN head is extinction-immune by construction: banked KEY tokens cannot be
   overwritten by logit drift). The RCA independently re-derives that design's premise.
2. **Class re-exercise is the active ingredient of replay** — er's KEY survival (85.8)
   with only 200 exemplars suggests targeted minority-class replay (KEY/HEADER-rich doc
   selection) could match full replay at a fraction of the bytes; ties to CoLaR-Bal's
   soft-CE result (SROIE +1.8 over 3 seeds).
3. **Cheap logit-prior correction** (H4): per-task label-frequency recalibration of the
   head at eval — a near-zero-cost baseline every fancy method must beat; if it recovers
   most of naive's drop, it is a paper finding on its own.
4. **Don't invest in trunk-protection mechanisms** (penalty/metaplastic on trunk): H3
   refuted + ewc's late-heavy displacement still forgets 43 pts — colar_meta (M1) is
   predicted to fail; treat its queued run as the falsification test.

## Caveats

- Tier B is n=1 seed / dil / LayoutLMv3; Tier A per-class findings are 3-seed.
- colar row = canonical k8/d5/r64, NOT the headline recipe.
- colar displacement is head-only after task 0 (backbone frozen) — excluded from H3.
- `text_only` mask mostly invalid (at-learning < 20) — text-drift eval test relies on
  `text_layout` vs `image_layout` contrast.
- Prompt/LoRA families not instrumented (out of Tier B scope).
