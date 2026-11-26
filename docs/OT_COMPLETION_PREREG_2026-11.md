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
