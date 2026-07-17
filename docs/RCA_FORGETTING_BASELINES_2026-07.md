# RCA: root causes of forgetting in multimodal doc-IE baselines (Tier C verdicts)

**Date:** 2026-07-17 (amended same day after adversarial verification — 16-agent
adjudication/verification pass; every load-bearing number independently recomputed from the
raw JSONs). **Adjudicates** the pre-registered hypotheses in `docs/RCA_HYPOTHESES_2026-07.md`
(registered 2026-07-16 BEFORE any Tier B data) against Tier A artifacts (`results/rca/a1_*`,
`a2_*`, 3 seeds) and Tier B instrumented runs
(`results/rca/dil_{naive,ewc,lwf,er,der_pp,colar}_seed42_rca.json`; synthesis
`results/rca/c_*.csv`). Decision rules applied **as written**; deviations flagged inline.

**Smoke check passed:** naive final FULL AA 37.9 (pre-registered bar 40 ± 3, cf. head-refit
oracle 39.6). Strata (FULL mean old-task Drop, `c_modality_deltas.csv`): high-forgetting =
naive 76.4, lwf 73.9, ewc 43.1, colar 28.9; replay control = er 2.2, der_pp 1.2. ⚠ colar ran
at its CANONICAL config (k8/d5/r64), not the headline k4/d50/r128 — it lands in the high
stratum here; do not compare to the 87.6 grid row.

**Schema fact (affects several rules below):** SROIE has ZERO HEADER gold tokens (row totals
0 at every boundary) — HEADER exists only in FUNSD; CORD has neither KEY nor HEADER and is
99.8% VALUE / 0.2% O at token level; SROIE is 84.8% O / 11.5% VALUE / 3.7% KEY.

## Verdicts

| Hypothesis | Verdict | One-line reason |
|---|---|---|
| H1 modality-asymmetric drift | **REFUTED (as mechanism)** | Both directional sub-mechanisms fail in eval AND weight space; the raw-scale "spread" that formally passes is a baseline-floor artifact |
| H2 head/label-space interference | **SUPPORTED** | All three predictions pass (verified; scope notes below) |
| H3 late-layer integration drift | **REFUTED** | Trunk displacement is front-loaded (early ≫ late) in naive/lwf; prediction (ii) untestable under the validity guard |
| H4 recency/logit bias | **SUPPORTED as amended (H4′)** | As-registered falsifier was drafted with a flaw (O both dominant label and falsifier target); the amended marginal-snap test is decisive |
| QA sroie-vs-funsd (exploratory) | **NOT SUPPORTED** | No receipts-overlap signature; the asymmetry is explained by the marginal snap + at-learning differences |

### H1 — REFUTED as mechanism

Eval space (`c_modality_deltas.csv`, valid old-task masks): raw spread between max- and
min-drop mask = naive 52.3, lwf 51.3, ewc 33.8 — formally ≥ 10. But the direction matches
NEITHER sub-mechanism: `image_layout` is the *minimum*-drop mask in 3/3 (H1b predicted
maximum), and `text_only` is guard-invalid in 2/3 methods (at-learning 1.7–15.4) so H1a's
eval test is information-starved. The raw spread is a floor artifact: `image_layout`
at-learning is only ~20–26, so it has little room to fall. **Normalized retention
(final/at-learning) collapses the story**: naive spread 4.7pp (image_layout 6.1%, full
10.1%, text_layout 10.8% retained — lockstep annihilation), lwf 11.3pp, ewc 20.7pp with the
worst mask *reversing* (text_layout for ewc vs image_layout for naive/lwf) — no coherent
single-modality direction on either scale.

Weight space kills both sub-mechanisms directly (`c_displacement_vs_signature.csv`):
- **H1a (text-drift):** `text_word_embed` is the SMALLEST of the three modality embeds in
  5/6 method×boundary checks and never largest (naive b1: text 1.9e-10 vs layout 7.0e-8,
  image 6.6e-8).
- **H1b (layout/vision decay):** predicted `image_layout` drops most — it drops LEAST
  everywhere.
- One more nail: the trunk group that moves most is **layernorm** — 2.4–20.9× the largest
  modality embed and 262–10,055× `text_word_embed` — i.e. what actually moves is not any
  modality pathway at all.

Note on rule semantics: the pre-registered falsifier ("spread < 10 = lockstep") is not
literally met on the raw scale, but the prediction clause requires a *consistent direction
matching H1a or H1b*, which fails everywhere; the normalized supplement shows even the raw
spread is confounded. Honest synthesis: the modality correlation lives in WHAT the head
loses (text-carried evidence = the dominant F1 mass), not WHERE parameters drift.

### H2 — SUPPORTED

1. KEY extinction is mask-uniform (per-mask `per_class` in the Tier B JSONs): naive and lwf
   final-boundary KEY F1 = **0.0 under every valid mask** on funsd AND sroie. The escape
   hatch (survival ≥ 20 under some mask while dead under FULL) never fires — checked across
   all methods and masks. HEADER is testable only on funsd (SROIE schema has none): 0.0
   under every valid mask for naive/lwf.
2. Confusion flow (`c_confusion_flow.csv`, final boundary, gold KEY/HEADER, old tasks):
   naive/lwf off-diagonal mass → VALUE-tags = **1.00** (O share 0.00 at the final boundary);
   8 of 9 method×entity cells pass the >50% VALUE-capture threshold (sole failure:
   ewc-HEADER 32.7%).
3. Replay stratum clean: er retains 0.79–0.99 of gold KEY mass on-diagonal (final KEY F1
   85.8 funsd / 91.3 sroie), der_pp 0.87–0.99; their small residuals go to O (ordinary
   errors, not capture).

Scope note: predictions (i)/(ii) were registered for naive/lwf only; ewc (also
high-forgetting, Drop 43.1) behaves intermediately (funsd KEY 24.9, HEADER 38.4 at final —
partial survival), so extinction is not homogeneous within the stratum.

### H3 — REFUTED

Trunk `displacement_by_depth` (head excluded, colar excluded post-freeze per pre-registered
caveat): naive early 0.70/0.64 vs late 0.03/0.03 (b1/b2); lwf 0.69/0.56 vs 0.05/0.09 —
**front-loaded**, meeting the falsification criterion (≥2 high-forgetting methods
early ≥ late). ewc is the one late-leaning method (late 0.48/0.35, though strict
late>mid>early fails even there — its Fisher penalty anchors what task 0 used). Prediction
(ii) (`image_layout` vs `text_only` drops) is **untestable as registered**: the validity
guard eliminates all 6 comparisons. Tension worth a paper sentence: weight displacement is
front-loaded while the CKA prior says functional drift grows with depth — "where weights
move" ≠ "where function changes", and neither is where damage is read out (the head).

### H4 — SUPPORTED as amended (H4′); as-registered rule was flawed

**Transparency first.** The pre-registered falsifier said "concentrated in O only with no
dominant-label capture" falsifies H4 — drafted assuming O ≠ the new task's dominant label.
At boundary 1 the just-trained task (SROIE) is **84.8% O**, and old-task off-diagonal mass
goes to O at 91–98% (naive/lwf/ewc). Read literally, the falsifier fires; read through
prediction (i) ("labels dominant in task b's training data"), O IS the dominant label and
the same data supports H4. The drafting flaw makes the as-registered verdict ambiguous, so
we adjudicate the amended, sharper test below and mark this deviation openly.

**H4′ (amended, decisive): the head's output marginal on old-task inputs snaps to the
just-trained task's gold label marginal** (`c_marginal_snap.csv`; cos = cosine similarity
of marginals under FULL; selected cells):

| method | cell | cos→just-trained | cos→own gold | O-row acc |
|---|---|---|---|---|
| naive | b1 funsd | **0.997** | 0.599 | 0.96 |
| naive | b2 funsd | **0.865** | 0.670 | 0.00 |
| naive | b2 sroie | **0.937** | 0.105 | 0.00 |
| lwf | b1 funsd | **0.998** | 0.616 | 0.96 |
| lwf | b2 sroie | **0.925** | 0.105 | 0.00 |
| ewc | b1 funsd | **0.979** | 0.786 | 0.87 |
| ewc | b2 funsd | 0.712 | **0.921** | 0.72 |
| ewc | b2 sroie | **0.986** | 0.129 | 0.02 |
| er | b1 funsd | 0.701 | **0.998** | 0.81 |
| er | b2 sroie | 0.096 | **1.000** | 0.99 |
| der_pp | b2 sroie | 0.093 | **1.000** | 0.99 |
| colar | b2 sroie | 0.166 | **0.997** | 0.90 |

Failing methods track the last-trained marginal in every cell (naive predicts 95% O on
funsd at b1, then 100% VALUE at b2); replay methods stay locked to the old task's own gold
everywhere. The snap direction *rotates with the schedule* (O after SROIE → VALUE after
CORD) — recency, not a static prior.

**The snap explains three otherwise-puzzling observations:**
1. **O-collapse (invisible to seqeval per-class F1):** naive/lwf O-row accuracy on old
   tasks is exactly **0.00%** at the final boundary (every true-O token predicted as a
   VALUE tag), while er/der_pp/colar keep 76–99%. O dies at b2, not b1 — because at b1 the
   snap direction WAS O (funsd O-acc 80.5 → 96.2 → 0.0 for naive). Entity extinction (b1)
   and O-collapse (b2) are the same mechanism pointed at different marginals, resolving the
   apparent two-schedule puzzle.
2. **a2's "+7.9 gradual recovery" for naive is not healing** — it is VALUE F1 partially
   returning (funsd VALUE 7.1 → 18.0) because the b2 snap direction happens to overlap
   funsd's own VALUE mass; KEY/HEADER (absent from CORD) stay at 0.0. Monotone loss exactly
   for never-re-exercised classes, as pre-registered in disambiguator (b).
3. **EWC's protection is task-0-biased:** it resists the snap for funsd (b2 cos→own 0.921)
   but snaps completely on sroie (0.986 vs 0.129, O-acc 2.4%) — its Fisher anchor was
   estimated when task 0 was the whole world, and its sroie at-learning was already
   crippled (41.9 FULL vs 82.5 for naive/er: the quadratic penalty impaired *acquisition*,
   a distinct stability-plasticity failure worth its own sentence).

**Weight-space anchor (from `displacement_by_group`):** classifier displacement exceeds the
entire summed trunk by 8.3×/16.5× (naive b1/b2), 24.2×/497.9× (lwf) — the "damage lives in
the head" claim is direct, not inferred. ewc is the exception (0.9–1.5×: the penalty pins
the head, displacement migrates trunk-ward — Finding-2 migration — and it still forgets
43 pts).

**H2 vs H4′:** both hold at different loci, as pre-registered. H2 names WHICH classes die
permanently (those the later schedule under-exercises — schema exclusion, e.g. CORD's zero
KEY, is the terminal case; SROIE's 3.7% KEY share already failed to keep funsd-KEY alive at
b1, so heavy skew suffices). H4′ names WHERE the mass goes (the last marginal) and WHY the
collapse is one-boundary-shaped.

### QA (pre-registered exploratory) — receipts-overlap NOT supported

"sroie forgotten more than funsd" is robust only in absolute final F1 (all 5 methods);
under the Drop framing it reverses for ewc (funsd 48.7 > sroie 37.4) because ewc's sroie
at-learning was already 41.9. Two operationalizations of "sroie's confusion is more
CORD-shaped than funsd's" both fail in 3/5 methods (naive/lwf are tied at cos ≈ 1.0 — total
collapse is generic, carrying no domain signature). The asymmetry needs no
domain-similarity story: sroie is 84.8% O, and the b2 snap (100% VALUE) destroys O-heavy
tasks hardest.

## Root-cause statement (n=1 seed, dil, LayoutLMv3 — provisional)

**Forgetting in these baselines is a readout-marginal snap on a class-asymmetric
substrate.** After each task, the shared head's output marginal on old inputs realigns to
the just-trained task's label marginal (cos 0.87–1.00 for failing methods vs 0.99–1.00 to
own-gold for replay); classes the later schedule under-exercises (KEY, HEADER — and O once
CORD's marginal excludes it) are extinguished mask-uniformly; trunk weight drift is
front-loaded but functionally minor (head/trunk displacement 8–500×), and no modality
pathway is the culprit — the "multimodal correlation" reduces to text being the head's
dominant evidence stream. Fully consistent with F1 (head-localized), the head-refit oracle
(features survive: pooled 55.3 vs trained-head 39.6), and F3 (only replay grounds the head).

**Falsification tests for the statement:** (i) frozen-trunk naive must reproduce
near-identical extinction (cheap, not yet run); (ii) logit/marginal recalibration alone (no
feature change) must recover a large fraction of the one-boundary drop — the snap is a
marginal shift, so a per-task label-prior correction attacks it directly; (iii) multi-seed
repeat of Tier B for the load-bearing numbers.



## Kill-test partial result: eval-time readout repair is dead (2026-07-17 pm)

Readout-suite retrain passed the smoke gate (`results/rca/killtests/train_meta.json`):
final FULL row FUNSD 18.6 / SROIE 3.9 / CORD 96.8, pooled AA 39.78 (registered gate
37.9 ± 2). The correction stage then failed the preregistered support bar: best pooled AA
was 39.78 uncorrected, with marginal-match 38.58, prior-ratio 39.70, and per-doc EM
39.60; old-task AA stayed ~11 and never recovered. CORD non-regression held, so the
negative is not from sacrificing the new task.

Saturation split decides the mechanism: uncorrected old-task KEY/HEADER probability mass is
~4e-6--7e-6 for KEY and <7e-7 for HEADER, far below the 1e-3 live-logit threshold. So the
verdict is not "calibration algorithm weak"; the final head has already annihilated the
recoverable mass. Test-time marginal correction is closed. Any positive method must keep
old classes alive during training or re-exercise them with task-consistent evidence.

Precision notes (adversarial verification, 5 independent recomputation agents — all
numbers above reproduce exactly): (a) per the amended prereg's own branch wording, the
sub-1e-3 saturation routes to "information destroyed pre-correction — family verdict
deferred, mechanism untested": the tested eval-time mechanisms are dead for THIS artifact,
but the rule reserves judgment on eval-time correction applied to a head whose mass
survives (e.g. after a training-time keep-alive). (b) Scope qualifier on
"irrecoverable": funsd-KEY shows a real partial recovery under marginal_match (0.0 → 16.4
F1) — masked in the pooled number because the same forced marginal degrades VALUE/O
elsewhere (sroie collapses to all-O: O-acc 100%, F1 0.0). HEADER (both tasks) and
sroie-KEY recover nothing under any variant.


## Kill-test partial result: frozen-trunk naive is acquisition-invalid (2026-07-17 pm)

`dil_naive_frozen_seed42_rca.json` repeats the old-class extinction pattern (final FUNSD
HEADER/KEY = 0.0, SROIE KEY = 0.0), but fails the preregistered comparability bar: final
FULL row is FUNSD 11.9 / SROIE 4.7 / CORD 68.4, AA 28.3, versus full naive 13.1 / 4.4 /
96.1, AA 37.8. The miss is not less forgetting; it is poor acquisition under a fully
frozen pretrained trunk (at-learning 41.4 / 47.7 / 68.4). Therefore this test does not
confirm pure-head causality. It still supports the practical lesson that old-class
extinction can occur with the trunk fixed, but it cannot be used as a clean effect-size
match to the standard naive baseline.

## Kill-test results: training-time marginal objectives are nulls too (2026-07-17 eve)

Both training-time anti-snap objectives ran at the registered settings (no tuning) and
changed essentially nothing — numbers adversarially verified against the raw JSONs:

**marginal_kl** (CE + KL anchoring the batch output marginal to the PRIOR-task mixture,
λ=1): at-learning 88.0/79.0/96.3; final row 17.8/3.2/96.3, AA 39.14 ≈ naive. KEY/HEADER
final F1 = 0.0 — the pre-registered H2-terminal-case reading HOLDS (the >20 refutation
never fires). The H4′-fixability side moves in the predicted direction but negligibly:
funsd VALUE 18.0 → 26.5 (+8.5), funsd O-recall 0% → 3.1%. AA landed below the 45–55
sanity window; per the amended rule this gates a mechanism-level adjudication, so the
verdict is **provisional NULL**: one candidate explanation is resolved (the anchor is
correctly inactive at task 0 and excludes the current task — verified in code), two remain
open (KL-vs-CE gradient scale at λ=1; batch-size-2 Monte-Carlo noise in the batch
marginal). A λ-sweep would be an unregistered follow-up, not a re-adjudication.

**logit_adjust** (balanced-softmax CE under the cumulative seen prior, τ=1): at-learning
86.0/82.9/96.0 — within 5 pts of Tier B naive's actual diagonal 87.8/82.5/96.1 on every
task, so the mandatory acquisition guard PASSES (no EWC-style plasticity damage; note the
prereg's hard-coded reference "88.5/84.0/97.6" was stale — corrected here, verdict
unaffected under either reference). Survival bar FAILS: KEY/HEADER final F1 = 0.0 vs the
>20 threshold; final AA 36.9. Balanced-softmax at τ=1 does not prevent the snap. Verdict:
**no buffer-free signal at the registered setting** ("family closed" is deliberately NOT
claimed — τ=1 is one point on the family's curve, and rule 4 registered only the binary
signal/no-signal outcome).

**Consolidated kill-test verdict (n=1 seed, provisional):** the readout-marginal snap is a
*symptom*, not an invertible mechanism. The trained head collapses predicted probability
mass on under-exercised classes to ~1e-5–1e-6 (a prediction-mass statistic; weight-space
geometry was not measured), which (i) no eval-time diagonal reweighting recovers in
aggregate, (ii) marginal-level training objectives at registered strength do not prevent,
and (iii) old-class extinction reproduces with the trunk frozen — while the head-refit
oracle (55.3, a FRESH probe on frozen features) shows the representation-level information
survives. The damage lives in what the existing head does with surviving features, and the
only registered intervention that prevents it remains task-consistent re-exercise
(replay stratum: er/der_pp/colar clean everywhere). This closes the "did you try cheap
recalibration" reviewer hole with a registered null.

## Deep-dive artifact pass (implemented 2026-07-17 pm)

`scripts/rca_deep_dive.py` is the CPU-only consolidation pass for this RCA. It reads the
Tier-B JSONs and emits `results/rca/d_class_extinction.csv`, `d_task_mask_drops.csv`,
`d_multimodal_summary.csv`, `d_marginal_snap_extended.csv`, `d_locus_summary.csv`, and
`d_summary.md`. No training or new instrumentation is hidden in this step; it is a
paper-facing repackaging of the already adjudicated Tier-B probes.

Current seed-42 summary: final old-task extinctions under the FULL valid task mask are
concentrated in failing methods (naive: FUNSD HEADER/KEY and SROIE KEY/VALUE; lwf: FUNSD
HEADER/KEY and SROIE KEY; ewc: SROIE VALUE). The strongest snap cells remain the old-task
inputs whose predicted marginal matches the just-trained task far more than their own gold
marginal (ewc b2/sroie 0.986 vs 0.129; naive b2/sroie 0.937 vs 0.105; lwf b2/sroie 0.925
vs 0.105). The multimodal summary keeps the same at-learning mask guard; it is for
quantifying mask sensitivity, not changing the H1 verdict.

## Method-design implications (input to the next brainstorm)

1. **Attack the readout, not the trunk — but not by blending or recalibrating.**
   ADJUDICATED 2026-07-17: the read-side blend is a no-signal (colar_knn λ=0.3 inert,
   λ=1.0 destructive), and every marginal correction/objective is a null (kill-tests
   above). What remains open on the readout side is *replacing* the damaged readout with
   one whose old-class evidence cannot be overwritten (the head-refit oracle's 55.3 is
   the existence proof) — any such method must store or reconstruct per-class evidence,
   which is replay-adjacent by construction.
2. **Cheap marginal-recalibration baseline: ADJUDICATED — registered null.** No cheap fix
   exists (best pooled recovery −0.1 AA); replay's value is NOT replicable by prior
   correction. This is now a paper finding (closes the standard reviewer hole), not a
   pending baseline.
3. **Class re-exercise is the active ingredient of replay** — er's KEY survival (85.8) with
   200 exemplars suggests targeted minority-class replay (KEY/HEADER-rich selection) could
   match full replay at a fraction of the bytes; but note SROIE's 3.7% KEY share was NOT
   enough for cross-domain KEY retention — re-exercise must hit the old task's
   distribution, not just the label.
4. **Don't invest in trunk-protection mechanisms:** H3 refuted + ewc's trunk-ward
   displacement still forgets 43 pts AND impairs acquisition (sroie 41.9) — colar_meta (M1)
   is predicted to fail; treat its queued run as the falsification test.

## Caveats

- Tier B is n=1 seed / dil / LayoutLMv3; Tier A per-class findings are 3-seed. Verdicts
  provisional until the load-bearing numbers repeat.
- colar row = canonical k8/d5/r64, NOT the headline recipe; colar displacement is head-only
  after task 0 (backbone frozen) — excluded from H3.
- `text_only` mostly guard-invalid (at-learning < 20); H1a's eval test rests on the
  weight-space evidence. Replay methods show small nonzero `image_layout` at-learning on
  sroie (0.2–1.3) where naive/ewc/lwf sit at exactly 0.0 — an acquisition-side curiosity
  outside the pre-registered scope.
- `rel_pos_bias` displacement is exactly 0.0 in every method/boundary — instrumentation
  note (likely non-trainable or unused in this checkpoint config), not a finding.
- Confusion-flow/marginal analyses use the FULL mask (per `rca_synthesize.py`); per-mask
  extinction claims come from the per-mask `per_class` fields.
- Prompt/LoRA families not instrumented (out of Tier B scope; the uniform probe is
  dishonest for prompt methods — see hypotheses doc).
- Verification provenance: adjudicated by 5 independent agents, each verdict checked by 2
  adversarial verifiers (numeric reproduction + methodology), plus a completeness critic;
  corrections folded in (H1 verdict tightened PARTIAL→REFUTED-as-mechanism, H4
  as-registered ambiguity disclosed, SROIE-HEADER schema fact, O-collapse, head/trunk
  ratios, ewc acquisition failure, SROIE gold O share corrected 99%→84.8%).
