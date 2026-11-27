# Mass-constrained label completion: preregistration (2026-11-26)

Follow-up direction after CVPR 2027 #35275. Written before any Gate 0 run. Plan of record:
`~/.claude/plans/regarding-the-research-for-ancient-pelican.md`. Rule agreed with the author:
**OT stays only where it beats its non-OT control.**

## Hypothesis

H: pseudo-label repairs of the shifted background failed on the *amount* of mass they assigned,
not on the teacher's ranking. A background token is a partial-label instance whose candidate
set {O} ∪ {classes its annotator could not see} is known from the session masks. Assigning
these tokens under class-frequency column marginals fixes the mass by construction.

Evidence that motivated H (all exploratory, older protocol): the entropy rule wrote 0 to 84
tokens into the memory per run (F10c); with sliced replay 49 to 186 per pass (E1, 2026-11-02);
backward completion at tau 0.9 relabelled about 27k current tokens per epoch and the new
session was not learned (E2, BiLC, 44.1).

Correction to the plan text: the oracle relabel counts quoted beside F10c (3231 / 2792 / 0 /
305) are from the **Mixed** stream, not CIL-CORD. The probe therefore reports its own hidden-
entity counts on CIL-CORD, and the "reproduce the ledger counts" sanity check is replaced by an
internal one (hidden-entity count = what `oracle_refresh` would change on the same tokens).

## Gate L verdict (literature, read 2026-11-26)

| Paper | What OT does there | Overlap |
|---|---|---|
| SoLar (NeurIPS 2022) | Sinkhorn pseudo-labels under candidate sets and a class prior (moving average), image partial-label learning | The formalism. Not continual, no background, no sequence labelling. |
| MProto (EMNLP 2023) | O tokens to prototypes, entity quota uniform and set by beta; O tokens assigned to an entity prototype are dropped from the loss | Closest mechanism. Distant supervision, not continual; masks, never relabels; no class-frequency prior. |
| Confident Sinkhorn Allocation | Class bounds from label frequency, tabular semi-supervised | Prior-bounded assignment exists outside CL. |
| Lifelong Event Detection via OT (arXiv 2410.08905) | OT loss aligning the classifier output with the LM head | Not a label assignment; the NA class is only re-weighted. |
| OT latent-space cultivation (arXiv 2211.16780) | Wasserstein fit of a per-class Gaussian mixture, online CIL | No background, no assignment. |
| COIL (ACM MM 2021) | Transport of classifier weights between old and new classes | No assignment. |
| MOTR (dual-OT continual NER) | Not found; known from one secondary citation | **Open. Must be read before any novelty claim.** |

No kill. Positioning consequence: the objective is SoLar's; a contribution can only be the
diagnosis (mass against ranking), the cross-session source of the marginals, and the two-sided
continual use. "SoLar applied to continual tagging" is the expected reviewer objection.

## A priori note on the OT-necessity control

For one hidden class against background, the zero-entropy transport solution **is** per-class
top-k. With several classes, exact OT differs from independent top-k only in how tokens
claimed by two classes are resolved. So the prior (the mass budget) may be the whole effect and
the coupling may add nothing. `quota_assign` uses the same prior without a coupling and is
the control that decides whether "OT" survives. Also measured before the run on synthetic
data: SoLar's kernel sharpness (lam = 3) leaves the plan too smooth for a hard label, so the
primary setting is lam = 10 (sensitivity: 3, 30).

## Gate 0: offline ranking and mass (CIL-CORD, seed 42, exploratory)

Run: `method=colar_otc_probe` (double-sliced CoLaR, k 4, 50 documents, rank 128; no label
repair and no true label in any loss), clean-protocol flags, output `results/gates/ot_gate0`.
Analysis: `scripts/rca/ot_gate0_mass.py`.

Two sides, at every session boundary:
- **current side**: O-labelled tokens of the new session, scored by the model before the
  session over {O} ∪ old classes; truth = hidden old-class entities;
- **stored side**: O-labelled tokens of documents banked earlier, scored by the model after the
  session over {O} ∪ the session's new classes; truth = hidden new-class entities.

Measures: macro average precision of the class scores (ranking); tokens assigned against
hidden entities (mass); micro precision / recall / F1 of the completed labels, pooled over
boundaries.

Rules compared:
1. threshold (argmax with confidence gate), tau in {0.5, 0.7, 0.9, 0.99}; the **best tau is
   picked on the truth**, which favours this control;
2. quota (per-class top-k), oracle marginals and estimated marginals;
3. Sinkhorn balanced, lam = 10, oracle and estimated marginals;
4. Sinkhorn with relaxed columns (exponent 0.9 and 0.5), estimated marginals.

Estimated marginal of class c = (share of valid tokens labelled c in the session that annotated
c) × (valid tokens on the side). On CIL-CORD each session's training set is a different subset
of the receipts, so this estimate is misspecified by construction; the oracle-marginal rows
bound what a perfect prior would give.

Bars (fixed now):
- **KILL-rank**: macro AP < 0.70 on either side. No assignment rule repairs a broken ranking.
- **KILL-mass**: best threshold rule within 3 F1 points of the best prior-based rule with
  estimated marginals on both sides. Then mass was not the problem.
- **PASS**: a prior-based rule with estimated marginals reaches completed-label F1 ≥ 85 on both
  sides and beats the best threshold rule by ≥ 10 points on at least one side.
- **OT verdict** (separate from pass/kill): if Sinkhorn (best of rules 3 and 4) is within
  3 F1 points of quota at equal marginals on both sides, the coupling is not needed; the
  direction continues as prior-constrained completion with the quota rule, and it is reported
  to the author as such before Gate 1.
- Anything between KILL and PASS is "inconclusive": report, do not proceed to training without
  the author's decision.

Known limits of this gate: one seed; the overlapped protocol (the same receipts return in every
session, so the teacher has trained on these pages) inflates current-side ranking, to be
controlled on a disjoint stream at Gate 2; boundary 1 is the only boundary whose teacher does
not depend on the training regime; label F1 is not downstream F1.

## Later gates (bars from the plan, unchanged)

- Gate 1a: two-sided true-label upper bound under a shared softmax; kill if < 88 full-label F1
  on CIL-CORD.
- Gate 1b: two-sided completion; pass at ≥ 70 on CIL-CORD and within 3 of the MiB-style loss on
  CIL-FUNSD. References: CoLaR 30.7, MiB-style 60.0 / 87.3, CoLaR-SBR 89.1 / 87.2.
- Gate 2: 5, 10 and 0 stored documents; pass at ≥ +10 over CoLaR-SBR at 10 documents (68.1),
  with the gain surviving on a disjoint stream.
- Claims need three runs with sign agreement; seed-42 gates are exploratory.

## Gate 0 verdict (2026-11-26, seed 42, exploratory)

Run `results/gates/ot_gate0/cil_cord_colar_otc_probe_seed42` (the probe run itself, double-
sliced CoLaR with no repair: full-label F1 54.4; last row of the session matrix 28.7 / 63.3 /
73.5 / 77.1 / 91.1). Table:
`ot_gate0_report.json` in the run directory. Completed-label F1, pooled over four boundaries:

| Rule | Current side | Stored side |
|---|---|---|
| threshold, best tau (0.5) | 81.4 (P 97.8, R 69.7) | 90.9 |
| quota, estimated / oracle marginals | 90.9 / 91.6 | 90.5 / 91.9 |
| Sinkhorn lam 10, estimated / oracle | 91.8 / 94.1 | 92.4 / 93.2 |
| Sinkhorn, relaxed columns 0.9 / 0.5 (est.) | 91.2 / 90.8 | 91.9 / 90.8 |
| *post hoc*: one scalar background shift, tuned on the truth | 90.2 (t 0.001) | 90.9 (t 0.3) |
| *post hoc*: total budget only, estimated / oracle | 89.4 / 91.6 | 83.0 / 91.9 |

Hidden entities: 38,431 of 46,369 O-labelled current tokens; 1,492 of 8,278 stored ones.
Macro AP per boundary: current 93.2 / 77.4 / 73.8 / 74.4; stored 98.5 / 96.6 / 89.1 / 59.8.

Against the bars:
- **KILL-rank: not triggered, with one cell below the bar.** The bar did not say whether it
  applies per boundary or per side (my omission). Per side (mean) it holds: 79.7 and 86.0. The
  one cell below 0.70 (stored, boundary 4) comes from two classes with 3 and 5 hidden tokens
  (AP 2.8 and 44.7); the two classes with 193 and 227 tokens have AP 96. With classes of at
  least 10 hidden tokens every boundary is at 79.9 or above.
- **KILL-mass: not triggered.** Current side +10.4 over the best threshold; stored side +1.5
  (the bar needed both sides within 3).
- **PASS: met on the letter, by 0.4 points.** Sinkhorn with estimated marginals: 91.8 and 92.4
  (both ≥ 85), +10.4 on the current side. The quota rule is at +9.5 and misses the +10 bar by
  0.5. One run; a margin of this size is not evidence that the two rules differ.
- **OT verdict: the coupling is not needed.** Sinkhorn minus quota at equal marginals: +0.9 and
  +1.9 (estimated), +2.5 and +1.3 (oracle). Ahead in all four cells, inside the 3-point bar in
  all four. By the agreed rule the direction continues as prior-constrained completion with
  the quota rule.

What the run says about H:
1. Mass is wrong on the **current side, and in the direction opposite to the one H assumed**.
   83% of the O-labelled current tokens are hidden old entities, and the threshold rule misses
   30% of them at 98% precision. The "27k tokens per epoch" of E2 was therefore not a flood: it
   is below the number of hidden entities. The failure of E2 (44.1) has another cause, which
   Gate 1a isolates.
2. Mass is **not** the problem on the stored side once replay is sliced: the plain threshold
   reaches 90.9. The near-empty refreshes of F10c belong to a teacher trained on unsliced
   replay, not to the assignment rule.
3. Post hoc controls (added after the first reading, so exploratory): a single scalar shift of
   the background score, tuned on the truth, recovers 90.2 on the current side, and a total
   budget with oracle count ties the per-class quota (91.6). What remains for the prior is that
   it sets the operating point without labels: the best scalar differs by a factor of 300
   between the two sides, and the total budget with an estimated count falls to 83.0 on the
   stored side where the per-class quota holds 90.5.
4. The estimated marginals overshoot (current side +9% to +21%, stored side up to 2.7×; a
   session's training set is selected on its own classes). The cost is 0.7 to 2.3 points.
5. Ranking on the current side falls from 93.2 at boundary 1 to about 74 afterwards, and the
   teacher of boundary 1 is the only one that does not depend on the training regime.

Limits unchanged: one seed, overlapped protocol, label F1 is not downstream F1.

## Gate 1a design (written 2026-11-26, before the run)

`method=colar_otc_oracle_both`: stored documents carry the true label of every class in the
head before each session (`refresh: oracle`, `refresh_before_task: true`), and the current-side
loss reads the same for the session's pages (`current_oracle: true`). No slicing. Seed 42,
clean-protocol flags, output `results/gates/ot_gate1a`.

Amendment to the plan: CIL-FUNSD is added to this gate. The stored-side oracle alone already
reaches 92.1 on CIL-CORD and 52.2 on CIL-FUNSD, so CIL-FUNSD is where a two-sided bound is
informative.

Bars: kill if full-label F1 < 88 on CIL-CORD (from the plan). On CIL-FUNSD, kill if < 84.3,
the Gate 1b pass bar there (within 3 of the MiB-style loss, 87.3): an upper bound below the
bar that the method must reach closes the dataset.

## Gate R: transport readout (written 2026-11-26 21:55, before any result)

Author's direction after Gate 0: find the place for OT in this problem. Candidate: the
**readout**. A shared softmax trained under the background conflict is miscalibrated between
sessions, and no label-based calibration is available because every stored label set is
incomplete. Matching the predicted class marginal to the class-frequency table needs no label:
the column potentials of the transport plan are a per-class logit correction, and the decoded
label is argmax(log p + g). Same operation as Gate 0's current side, moved to test time.
Signal that motivated it (exploratory, token-level split of the Gate 0 dumps): potentials
fitted on half the tokens decode the other half at 91.8 to 94.0, argmax 81.1, top-k
thresholds 90.7 to 91.1, one-step prior adjustment 88.7 to 89.9.

Data: the saved test-set outputs of finished runs (`full_label_dump.pt`, three runs each), no
training. Script `scripts/rca/ot_readout.py`. Metric: full-label F1 (seqeval), per document
sequence; the argmax row must reproduce the stored `full_label_f1`.

Primary models (CIL-CORD, shared softmax, CoLaR k4 d50 r128): plain CoLaR (`colar_dump`),
double-sliced (`colar_r_sliced2`), MiB-style loss (`colar_mib`).

Rules:
1. argmax (as stored);
2. transport readout, lam 1 (plain marginal matching, no sharpness to tune), over all test
   tokens; sensitivity lam 10 and relaxed columns 0.5;
3. the same with potentials fitted on one half of the test documents and applied to the other
   half (two folds), to separate the correction from transduction;
4. same-prior controls without a coupling: per-class top-k; one-step prior adjustment
   (log prior minus log mean predicted probability);
5. control without a prior, tuned on the truth (favours the control): one scalar shift of the
   background logit.

Prior: estimated = the class-frequency table of the Gate 0 run (share of valid tokens of class
c in the session that annotated c; one table, used for the three runs); background share =
max(1 − sum, smallest class share), then normalised. Oracle = test counts (upper bound, also
reported on CIL-FUNSD and CIL-WildReceipt, whose estimated tables need a dataset build).

Bars:
- **PLACE**: rule 2 with the estimated prior raises mean full-label F1 by ≥ 10 over argmax on
  at least two of the three models, same sign in all three runs, and rule 3 keeps at least
  half of that gain.
- **OT against its control**: rule 2 exceeds the best rule-4 control by ≥ 3 points in the mean
  with the same sign in all three runs, on at least two of the three models. Otherwise the
  simpler rule is reported as equivalent, as at Gate 0.
- **KILL**: gain over argmax below 3 on all three models.

Known limits: 100 test receipts; rule 2 is transductive; a readout cannot restore a class the
features no longer separate.

## Gate R verdict (2026-11-26, three runs per model, saved outputs)

Report: `results/gates/ot_gate0/cil_cord_colar_otc_probe_seed42/ot_readout_report.json`. The
argmax row reproduces the stored full-label F1 of every run. CIL-CORD, estimated prior, mean
of three runs (per run in the report):

| Model (shared softmax) | argmax | transport readout | held-out potentials | top-k | one-step | background shift tuned on truth |
|---|---|---|---|---|---|---|
| plain CoLaR | 29.6 | 74.6 | 72.8 | 52.5 | 52.4 | 62.7 |
| double-sliced CoLaR | 46.7 | 83.7 | 83.3 | 69.8 | 77.6 | 72.9 |
| MiB-style loss | 60.0 | 82.2 | 81.4 | 71.4 | 76.5 | 79.5 |
| current-sliced CoLaR (secondary) | 30.6 | 72.9 | 72.7 | 53.3 | 57.1 | 62.9 |
| ER, 250 pages (secondary) | 21.3 | 79.2 | 78.5 | 64.9 | 66.2 | 73.6 |

- **PLACE: met on the three primary models.** Gains +45.0, +37.0, +22.2, positive in every
  run; held-out potentials keep 43.2, 36.6, 21.4.
- **OT against its control: met on the three primary models.** Over the best same-prior
  control: +22.1, +6.1, +5.7, positive in every run (on the MiB-style model two of the three
  runs are within 2.5 points).
- **KILL: not triggered.**

What limits the result (all from the same report):
1. It does not reach the blocked readout: CoLaR-SBR is at 89.1 on CIL-CORD.
2. Sensitivities are not flat: lam 10 costs 4 to 6 points with the estimated prior, relaxed
   columns (0.5) cost 7 to 17. The balanced plan at lam 1 is the only setting that passes.
3. On models that are already calibrated the estimated prior costs: joint 91.7 to 90.9, true
   stored labels 92.1 to 89.8 (with the oracle prior: 92.8 and 92.5).
4. Where the features are gone the readout does nothing: naive 13.2 to 20.5, LwF 13.0 to 30.6
   with one run unchanged.
5. Other datasets, oracle prior only (upper bound of the readout; estimated tables pending):
   CIL-FUNSD plain CoLaR 52.2 to 64.4, double-sliced 83.8 to 86.3 (top-k 86.2), MiB-style 87.3
   to 85.1 (worse), joint 84.3 to 83.1 (worse); CIL-WildReceipt plain CoLaR 32.5 to 42.6
   (runs 36 / 57 / 35), MiB-style 66.5 to 70.1 (top-k 69.7), ER 30.8 to 44.5, joint 83.7 to
   84.0. CoLaR-SBR there: 87.2 and 79.5. The large effect is specific to CIL-CORD so far.
6. 100 test receipts; the prior comes from the same distribution as the test set.

Open before any claim: literature on prior matching at test time in class-incremental
learning and label-shift correction (Sinkhorn / iterative proportional fitting, bias
correction layers, logit adjustment); estimated tables for the other two datasets; a stream
whose test prior differs from the training prior.

## Gate L2: literature for the readout (read 2026-11-26)

| Paper | What it does | Overlap |
|---|---|---|
| OTTER (Shin, Zhao, Cromp, Vishwakarma, Sala; arXiv 2404.08461) | Exact transport (linear program) between a batch of zero-shot predictions (cost −log p) and a given label distribution; label = argmax of the plan row. R-OTTER learns a per-class reweighting from the OTTER labels for online use. Theorem: Bayes-optimal under label shift when the given distribution is the true one. Beats Prior Matching on 17 of 21 datasets. | **The mechanism.** Not continual, no background class, no sequence labelling; the label distribution is given (true, or estimated from 10 labels per class). |
| Prior Matching (the baseline of OTTER) | Per-class weights on the probabilities so that the mean prediction equals the prior | **Equal to our primary setting**: the entropic plan at lam 1 is this matrix scaling, solved by Sinkhorn iterations. |
| ARC (Chen, Goldblum, Wu, Jiang; arXiv 2405.14318) | Test-time correction of the recency bias in class-incremental image classification with a task-wise softmax score and pseudo-label updates | Same goal, no prior, no transport, images only. |
| AdaPrior (CVPR 2026), Prior2Posterior | Post-hoc prior correction for long-tailed (continual) classification, prior estimated from the model | Not read in full. Same family of corrections. |
| BiC, WA, IL2M | Bias correction layers or statistics for old against new classes | Need a labelled validation set or per-class statistics from labelled data. |
| MProto, CPFD, PLOP, BACS, "Mitigating Background Shift" (ECCV 2024) | Training-time handling of the background (pseudo-labels, distillation) | None corrects the readout with a class prior. |

Four searches and two papers read (both pages truncated at about half; the rest from a model
summary). No paper found that matches a predicted class marginal to class frequencies in
continual sequence labelling or under background shift.

Consequences:
1. The readout is **OTTER / Prior Matching applied to a continual stream**; the mechanism is
   not new and both must be named and compared. An exact-plan row (as OTTER) is added to
   `ot_readout.py` after this reading.
2. What the stream adds: the label distribution needs no labelled calibration set, it
   accumulates as a frequency table while sessions arrive; and the finding that most of the
   background-conflict loss of a shared softmax on CIL-CORD is recoverable by that table alone.
3. A paper needs more than the readout: the correction inside training (next gate), and a
   case where the table is misspecified.

## Gate 1b design (written 2026-11-26 22:15, before any run)

`method=colar_otc_{transport,quota,threshold}`: two-sided completion under one shared softmax
(`doccl/methods/colar_otc.py`, `completion`). Current side: the model before the session scores
the session's O-labelled tokens over {O} ∪ old classes, the rule assigns them once for the whole
training set, plain CE on the completed labels. Stored side: after the session the trained
model scores the O-labelled tokens of the banked documents over {O} ∪ the session's classes
(background = everything else), the rule writes the labels; until then the document is sliced.
Masses from the frequency table; `transport` = entropic plan at lam 1 (the Gate R setting),
`quota` = per-class top-k with the same masses, `threshold` = argmax with tau 0.5.

Runs: seed 42 first, CIL-FUNSD and CIL-CORD, clean-protocol flags plus the test-output dump,
output `results/gates/ot_gate1b`. Each run is scored twice: argmax, and the transport readout
with the same table (Gate R rule 2).

Bars (unchanged from above): pass at ≥ 70 on CIL-CORD and ≥ 84.3 on CIL-FUNSD (argmax).
OT against its control: `transport` exceeds the better of `quota` and `threshold` by ≥ 3 on
CIL-CORD; seed 42 alone is exploratory, a claim needs the three runs. References: double-sliced
CoLaR 46.7 (83.7 with the readout), MiB-style 60.0 / 87.3, CoLaR-SBR 89.1 / 87.2, two-sided
true-label bound from Gate 1a.

## Gate 1a verdict (2026-11-26, seed 42)

Two-sided true-label bound under one shared softmax: **CIL-CORD 94.9, CIL-FUNSD 88.6**
full-label F1 (`results/gates/ot_gate1a`). Both above their kill bars (88 and 84.3). For
scale: joint 93.1 and 84.3 (three-run means), stored-side oracle alone 92.1 and 52.2,
CoLaR-SBR 89.1 and 87.2. A shared softmax is sufficient when both sides are labelled; on
CIL-FUNSD the room above the methods of the paper is about one point.

## Gate R, estimated tables on the other two datasets (2026-11-26)

Tables from `scripts/rca/class_freq_table.py` (training sets only). Mean of three runs,
argmax -> transport readout with the estimated table [best same-prior control]:

| Model | CIL-FUNSD | CIL-WildReceipt |
|---|---|---|
| plain CoLaR | 52.2 -> 71.0 [56.7] | 32.5 -> 42.7 [38.4], runs 36.5 / 56.1 / 35.5 |
| ER | 51.4 -> 69.7 [62.7] | 30.8 -> 44.5 [38.6] |
| current-sliced CoLaR | 78.0 -> 85.6 [84.9] | not run |
| double-sliced CoLaR | 83.8 -> 85.2 [84.6] | not run |
| MiB-style loss | 87.3 -> 85.3 [87.3] | 66.5 -> 69.9 [69.3] |
| joint | 84.3 -> 82.9 [84.3] | 83.7 -> 83.5 [83.8] |

Reading: the readout repairs models that the background conflict left badly miscalibrated
(+10 to +19 on plain CoLaR and ER on every dataset, held-out potentials within 1.1), and it
is neutral to slightly harmful (−2.0 at worst) on models that are already calibrated. It beats
its controls by 3 or more only on the badly miscalibrated models. On CIL-WildReceipt it
recovers a small part of the gap (CoLaR-SBR 79.5). On CIL-FUNSD the estimated table does
better than the test counts on plain CoLaR (71.0 against 64.4), which the method does not
explain; noted, not interpreted.

Exact plan (the OTTER rule), first timing: one linear program takes 213 s on CIL-CORD beside
a training job, and gives 75.2 on the double-sliced run of seed 42 where the entropic plan at
lam 1 gives 83.1. The three primary models are being run; rows will be added to the report.

## Gate R, exact plan (the OTTER rule) on the three primary models (2026-11-26)

CIL-CORD, mean of three runs, added after the literature check (exploratory):

| Model | argmax | entropic lam 1, est. / oracle prior | exact plan, est. / oracle prior |
|---|---|---|---|
| plain CoLaR | 29.6 | 74.6 / 74.1 | 68.5 / 75.0 |
| double-sliced CoLaR | 46.7 | 83.7 / 85.0 | 75.5 / 83.3 |
| MiB-style loss | 60.0 | 82.2 / 82.4 | 74.7 / 81.8 |

With the true class counts the two plans agree within 1.7. With the estimated table the exact
plan loses 6.1 to 8.2 points in every run and the entropic plan loses at most 1.3. The exact
plan enforces the table as hard quotas; the scaling at lam 1 only moves one bias per class.
This matches the lam 10 row of the verdict above (sharper plan, worse with the estimated
table) and is the one place where our setting differs in outcome from the published one.

## Gate 1b verdict, v1 (2026-11-27, seed 42, exploratory)

Full-label F1, `results/gates/ot_gate1b`; second number = with the transport readout
(estimated table) on the saved test outputs:

| Completion rule | CIL-FUNSD | CIL-CORD |
|---|---|---|
| transport | 88.5 (86.3) | 72.8 (84.1) |
| per-class top-k | 82.8 (84.8) | 62.4 (84.6) |
| threshold 0.5 | 88.3 (86.4) | 57.1 (86.5) |

References: two-sided true labels 88.6 / 94.9; CoLaR-SBR 87.2 / 89.1; MiB-style 87.3 / 60.0;
double-sliced CoLaR with the readout 85.2 / 83.7.

- **Pass bars: met by the transport rule** (72.8 ≥ 70 on CIL-CORD, 88.5 ≥ 84.3 on CIL-FUNSD).
- **OT against its control (argmax): met on CIL-CORD**, +10.4 over top-k and +15.7 over the
  threshold. On CIL-FUNSD transport and threshold tie (88.5, 88.3).
- **With the readout on top the three rules are within 2.4 points on CIL-CORD and the order
  reverses** (threshold 86.5). The gain of the transport rule is a calibration gain that the
  readout also delivers. On CIL-FUNSD the readout costs about 2 points on these models.
- No run reaches CoLaR-SBR on CIL-CORD. On CIL-FUNSD transport and threshold are at the
  true-label bound and 1.3 above CoLaR-SBR, one run against a three-run mean.

What is wrong with the v1 models (test outputs): every error is a class predicted as O. The
final models predict O on 714 / 978 / 1123 of 2,356 test tokens; 4 are O. Under the transport
rule the last session is not learned (its own F1 15.3; 82% of its tokens predicted O), under
the threshold rule it is (91.0) while session 1 is lost (36.1).

Flaw found in the implementation, not in a rule: a page enters the bank with the labels of
the loader, which hide every OLDER class under O, and the stored-side pass only writes NEWER
classes. From session 2 on, each bank therefore replays old entities as O for the rest of the
stream. The design text above did not say which labels are banked; this was an omission.

## Gate 1b v2 and timing diagnostic (written 2026-11-27 04:40, before the runs)

`bank_completed: true`: a page is banked with the labels its session trained on (completed,
or true under `current_oracle`). Configs `colar_otc_{transport,quota,threshold}_bank`. CIL-CORD
only (on CIL-FUNSD the first bank hides no older class, so v2 equals v1 there). Same bars as
Gate 1b; exploratory, seed 42.

`colar_otc_oracle_late`: true labels on both sides with the timing of the completion runs (a
stored page receives a session's classes only after that session; sliced until then). Reading
fixed now: within 3 points of the 94.9 of Gate 1a means the timing is harmless and the
remaining gap of v2 is label error; a larger drop means the stored side must be completed
during the session, not after it.

Order: transport, threshold, oracle_late, top-k. Output `results/gates/ot_gate1b_v2`.

## Gate 1b v2 verdict (2026-11-27, CIL-CORD, seed 42, exploratory)

`results/gates/ot_gate1b_v2`. The machine went down at about 10:18 during the fourth run
(top-k, session 1); cause not established; that run is not repeated here.

| Run | argmax | + transport readout (est. table) | + one-step adjustment |
|---|---|---|---|
| transport completion, banked | 83.2 | 85.5 | 84.5 |
| threshold completion, banked | 83.2 | **90.0** | 91.2 |
| true labels, completion timing (`oracle_late`) | 93.6 | 92.6 | 94.4 |

- Banking the completed labels is worth +10.4 (transport) and +26.1 (threshold) over v1.
- Timing diagnostic: 93.6 against 94.9 with labels before the session, inside the 3 points
  fixed above. The late stored side is harmless; what remains is label error.
- **The transport rule no longer beats the threshold inside training** (83.2 both), and after
  the readout it is 4.5 below. At boundaries 3 and 4 the table overshoots the number of O
  tokens, the balanced plan assigns every O token to an old class (11,939 of 11,939 with
  10,493 hidden), and the session's own classes are learned badly (own F1 9.1 at session 4).
- **Best single-softmax pipeline so far: threshold completion + banked labels + transport
  readout = 90.0** (held-out potentials 90.0), against CoLaR-SBR 89.1 (three-run mean). One run.
- On these better-calibrated models the one-step adjustment matches or beats the transport
  readout (91.2 against 90.0). The readout's advantage over its controls holds on the badly
  miscalibrated models of Gate R, not here.

## Feasibility batch (written 2026-11-27 15:00, before the runs)

Question: does "one shared softmax + label completion + class-prior readout" hold beyond one
run and beyond a 50-page memory? Configs `colar_otc_{threshold,transport}_bank`, clean-protocol
flags, test outputs saved; every run scored by argmax, transport readout and one-step
adjustment with the estimated table.

1. CIL-FUNSD, seeds 42 / 7 / 123, both rules (`results/gates/ot_feas`).
2. CIL-CORD, 10 stored pages per session (`method.docs_per_task=10`), seed 42, both rules
   (`results/gates/ot_feas_d10`). Reference CoLaR-SBR at 10 pages: 68.1.
3. CIL-CORD, 50 pages, seeds 7 and 123, threshold rule then transport rule.

Bars, on completion + transport readout:
- FEASIBLE at 50 pages: three-run mean on CIL-CORD ≥ 88.1 (within 1 of CoLaR-SBR) and on
  CIL-FUNSD ≥ 84.2 (within 3 of CoLaR-SBR).
- FEASIBLE at 10 pages: ≥ 78.1 (CoLaR-SBR + 10), seed 42.
- OT component inside training: transport rule ≥ threshold rule + 3 in the three-run mean,
  else the threshold rule is kept. OT component at readout: transport readout ≥ one-step + 3
  in the three-run mean, else the simpler adjustment is reported as equivalent.
