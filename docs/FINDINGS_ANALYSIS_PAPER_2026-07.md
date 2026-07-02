# Findings: The Anatomy of Forgetting in Multimodal Document IE

**Date:** 2026-07-02
**Status:** Consolidated articulation of the full research program (diagnostic pilot →
7 proposed/ported methods → falsification chain). This is the argument spine of the
analysis paper (the direction chosen on 2026-07-02: analysis paper + LexMem-v3b spine).

All numbers are from ground-truth artifacts: `STATE.md`, `results/ledger_gate*.json`,
`thesis/generated/table_main.tex`, `results/dil_lexmem*_seed42/metrics.json`.

---

## Central claim

Catastrophic forgetting in multimodal document information extraction is
**head/output-localized and architecture-general**; under naive protection the
forgetting locus **migrates** to whatever remains plastic; and **no buffer-free
consolidation mechanism** — weight merging, merge + generative realignment, gradient
subspace transfer, or input-anchored lexical memory — recovers the lost competence.
**Only replay** (real gradients from past-task data) grounds the shared classifier head
across conflicting domain optima.

This is not a string of failed methods. It is a convergent, mechanistically explained
negative result, established by testing the "memory can substitute for replay"
hypothesis at three successively weaker levels and falsifying all three.

---

## Finding 1 — Forgetting is head-localized and architecture-general

**Evidence (diagnostic pilot, FUNSD→CORD→SROIE, 4 mask conditions × seeds):**

- Fisher-weighted parameter displacement concentrates in the classifier head/output
  block on **all four backbones**: LayoutLMv3, BERT, LiLT, BROS.
- CKA representation drift is monotone with depth — early layers barely move; the
  penultimate representation and head move most.
- Permutation tests across backbones (p = 0.38 / 0.51) fail to reject a **shared
  locus**: the *location* of forgetting is backbone-independent, even though the
  *amount* differs (n = 3 seeds supports shared location, not equal magnitude — this
  caveat is logged and must be stated in the paper).

**Interpretation.** In doc IE the encoder's multimodal features remain largely intact
across tasks; what is destroyed is the *readout* — the mapping from features to the
label space. This is why regularization on the trunk (EWC 41.2, LwF 43.7 vs naive 41.3
AA on DIL) buys almost nothing: it protects the part that wasn't the problem.

---

## Finding 2 — The locus migrates to whatever stays plastic

**Evidence (LexMem pilot arms, DIL seed 42):**

| arm | AA | BWT | task-0 drop | mechanism observed |
|-----|----|-----|-------------|--------------------|
| v1 freeze-all | 41.1 | −0.0 | 0.0 | pure stability, no plasticity (new-task 3.4) |
| v2 freeze head+late, plastic early/mid | 38.8 | −76.8 | −84 | **early/mid CKA collapses 1.0 → 0.19** |
| ctrl (same freeze map, no memory) | 38.9 | −75.6 | −83 | identical — drift, not memory, is the bottleneck |

Freezing the diagnosed locus (head + late layers) does **not** preserve old-task
competence: the optimization pressure *redirects* into the early/mid trunk, which was
stable under naive training (CKA ≈ 1.0) and now drifts catastrophically (CKA 0.19).
The v2-vs-ctrl comparison is the key control: adding the memory module changes nothing
(38.8 vs 38.9), proving the failure is drift-driven, not memory-capacity-driven.

**Interpretation.** "Where forgetting lives" is not a fixed property of the
architecture — it is a property of the *optimization under constraint*. The diagnosis
(Finding 1) describes the unconstrained failure mode; any intervention that blocks that
pathway reroutes the damage rather than preventing it. This is the sharpest novel
finding of the program and explains a priori why locus-targeted buffer-free methods
asymptote (Finding 3).

---

## Finding 3 — Only replay grounds the head; every buffer-free mechanism fails

**Reference operating points (DIL, LayoutLMv3, 3 seeds, AA):**

- Floor: naive 41.3 · EWC 41.2 · LwF 43.7
- Replay: ER 87.9 · DER++ 88.0 · ER-C-Flat++ 88.4
- Oracle: joint 88.7

Replay methods sit ~0.5 pt below the joint oracle. Everything else proposed or ported
in this program lands at or near the floor:

| method | mechanism | outcome | decisive evidence |
|--------|-----------|---------|-------------------|
| HRP (ours) | sparse-OCR prompt routing + head replay | routing works; lift comes from replay | routing acc 0.29→0.92; AA 20.9→31.7 and BWT −10.1→**+5.7** only when head-replay added |
| DocMERGE (ours) | diagnosis-guided head merge (plain/TIES/Fisher) + KV memory | NO-GO | best merge 21.6 ≈ no-merge 21.9; merged-head cos(d0) = 0.99 → merge is a near-no-op on conflicting rows |
| LCA (ICLR'26 port) | TIES merge + Gaussian classifier realignment | NO-GO on doc IE | AA 41.9 ≈ naive; published image-CIL SoTA does not transfer |
| HGT / CUBER (ours) | head / whole-net gradient-subspace transfer | NO-GO | α is a pure stability↔plasticity dial; no configuration yields positive BWT |
| LexSlot / LexSlot-FM (ours) | residual lexical slot memories on a plastic base | DROPPED (RCA) | standalone ≈ naive (AA ~42); the 87.3 figure was the DocCL hybrid (replay+KD+Fisher), not the standalone method |
| LexMem v3b (ours, SMF port) | frozen-after-task-0 base + sparse KV logit memory + trunk EWC | partial — "EWC++" | AA 66.0, task-0 drop −4.4 (bar passed), but mid-task SROIE 68 < 75; asymptote ~22 pts below replay |
| Lexical Ledger (ours) | input-anchored (lexeme × layout-cell) append-only memory, zero-training probe | FALSIFIED | own-ledger CORD 76.1, but **cumulative** ledger (what sequential CL actually produces) collapses: FUNSD 21 / SROIE 23 ≪ 60 bar; key collisions cost −18 on CORD |

**The falsification chain.** The hypothesis "an explicit memory can substitute for
replay" was tested at three successively weaker (more drift-immune) levels:

1. **Parametric slot memory on a plastic base** (LexSlot) — falsified: the base drifts
   under the slots, and residual corrections written in an old feature space are
   meaningless in the new one.
2. **Feature-space memory on a drift-controlled base** (LexMem v1–v3b) — partially
   falsified: freezing induces locus migration (Finding 2); adding trunk EWC recovers
   to AA 66 but the mid-task residual lives in the trunk (slot Jaccard 0.0, CKA 0.99 —
   i.e. *not* a memory problem), so the line asymptotes.
3. **Input-anchored memory, drift-immune by construction** (Lexical Ledger — keys are
   raw lexeme × layout cells, no learned features at all) — falsified: without a
   shared feature space, task-conditional retrieval fails; merged ledgers collide and
   cumulative accuracy collapses even though per-task ledgers are strong (76.1 own).

**Interpretation / mechanism.** The shared classifier head must place N tasks'
conflicting label geometries into one weight matrix. Merging cancels conflicting rows;
regularization freezes the wrong subspace; subspace transfer only interpolates the
stability–plasticity trade-off; and input-keyed memory dodges drift but loses the
shared representation needed to route queries. Replay is unique because it supplies
*real past-task gradients* that re-carve the joint optimum every session — it is the
only mechanism among those tested that optimizes the actual multi-task objective
rather than a proxy for it.

---

## Secondary findings

- **Lexical routing is a real signal but scenario-bound.** TF-IDF/OCR-vocab routing
  identifies the source task at 0.92 on DIL (distinct document families) but collapses
  to 0.22 on same-dataset CIL splits. Any lexically-gated method inherits this ceiling.
- **The single positive-BWT result in the program is head-replay** (HRP: +5.7),
  consistent with the central claim.
- **FWT as currently computed is invalid for CIL** (zero-shot on unseen labels ≈ −85
  for every method). Forward-transfer claims need a learning-curve/few-shot metric.
  (`docs/FWT_NOTE.md`)
- **Underfit-floor confound.** Several NO-GO verdicts were reached at a low operating
  point (3-epoch, frozen-backbone, at-learning F1 ≈ 21 vs replay 88). RCAs argue the
  mechanisms fail independently of the operating point (merge is a near-no-op even on
  strong heads; ledger own-vs-cumulative gap is training-free), but this is the paper's
  main attack surface and must be addressed explicitly.

---

## What the paper is, and is not

**Is:** a diagnostic + falsification paper. (1) A multi-backbone anatomy of forgetting
in multimodal doc IE (localized, general, migrating). (2) A controlled demonstration
that four distinct buffer-free consolidation families fail *for identified mechanistic
reasons*, with LexMem-v3b as the buffer-free positive control showing what partial
recovery looks like (AA 66) and precisely where it asymptotes (mid-task trunk drift
under task-0-dominated online Fisher). (3) The implication: in this regime the research
question is not "how to avoid the buffer" but "how small can the buffer be / what must
it contain" — replay's gradients are doing something no tested surrogate replicates.

**Is not:** a SoTA-method paper. No proposed method beats replay; the thesis
"LexSlot (ours) 87.3" row is the DocCL hybrid and must be relabeled before submission
(tracked, deferred).

**Gating work before submission:**
1. LexMem-v3b multi-seed (7, 123) — the spine's error bars. Currently **stalled**: the
   watcher is deadlocked behind opencode's repeated `lexslot_fm` launches (decision on
   killing/relaunching pending).
2. Multi-backbone generalization grid (rented GPU) — required for "architecture-general"
   to cover the *method comparison*, not only the diagnostic.
3. Fix the thesis LexSlot row label; rebuild FWT for any transfer claims.
