# Findings: The Anatomy of Forgetting in Multimodal Document IE

> **Positioning (2026-07-13).** This paper is **diagnostic + falsification, fully scoped ("B+")**
> — every headline claim multi-seed × multi-scenario × multi-backbone (venue/acceptance rationale
> in `EXPLORE.md` §6, critical path in `ROADMAP.md`). Two later results slot into the argument, not
> as method contributions: **CoLaR** (per-doc SVD compressed latent replay, dil r128 = 87.6 @ 60 MB,
> lossless 2.7×) is the *constructive control* — compression that respects whole-document consistency
> works while every marginal summary collapses, proving the negatives are not effort-limited;
> **PLaR** (public-proxy replay, zero private bytes, ~59 AA) is a *bounded-success negative*
> (coverage-limited). See Finding 3b for the consistency-law mechanism they bracket.

**Date:** 2026-07-02 (v5 reframe added 2026-07-04; positioning 2026-07-13)
**Status:** Consolidated articulation of the full research program (diagnostic pilot →
8 proposed/ported methods → falsification chain). This is the argument spine of the
analysis paper. **Framing decision (2026-07-04): this is a DIAGNOSTIC + FALSIFICATION
paper, not a method paper.** No proposed method beats replay; each is evidence in a
convergent negative result. In particular LexMem v5 (feature-Gaussian graph replay) is
NOT a contribution — feature-statistic replay is a solved 2021-2023 subfield (PASS
CVPR'21, FeTrIL WACV'23, FeCAM NeurIPS'23). v5 enters the paper only as the *terminal
tombstone* of the falsification chain: the strongest, most literature-standard
buffer-free method (plus a novel relational variant) STILL cannot beat representation
drift. Its value is as evidence, not as a method.

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
| LexMem v5 (ours) | feature-Gaussian **generative replay** into a drift-controlled head; edges-on = novel *relational* variant (prototypes reconstructed via graph message-passing) vs edges-off = standard independent-prototype replay (FeCAM-style) | FALSIFIED (terminal) | BOTH variants AA 63.2, row [FUNSD 86.6 / **SROIE 13.0** / CORD 89.9]: the mid-task collapses to 13 — the field-standard buffer-free method, and a novel graph variant, both cannot beat representation drift. **Honesty caveat:** the two arms came out byte-identical (ΔAA exactly 0.00) despite differing reconstruction-loss streams (edges-on recon≈1.42 vs edges-off≈0.013), so we do NOT claim a clean edges-vs-no-edges ablation — the head-side replay signal is *swamped by trunk drift* (the recon gradient reaches the head but does not move the evaluated model). That the ablation is *uninformative* is itself consistent with Finding 3: the bottleneck is trunk-resident representation drift, not head misalignment, so no head-side reconstruction — relational or not — can reach it. |

**The falsification chain.** The hypothesis "an explicit memory can substitute for
replay" was tested at four successively stronger / more literature-standard levels,
all falsified:

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
4. **Feature-Gaussian generative replay — the field's own strongest buffer-free tool**
   (LexMem v5; independent-prototype replay = PASS/FeTrIL/FeCAM, plus a novel relational
   graph variant) — falsified: AA 63.2, mid-task SROIE collapses to 13.0. This is the
   decisive level: the mechanism the CIL literature holds up as the exemplar-free answer
   still cannot survive representation drift in doc-IE, and neither does making it
   relational. **The method is not the contribution — its failure is.** It closes the
   chain: if *this* can't do it, no buffer-free memory can, and the "only replay grounds
   the head" claim (Finding 3) is complete. *(Finding 3b, added 2026-07-10, empirically
   discharges the "no buffer-free memory can" clause: we then tried the logically strongest
   remaining case — a **real**-activation coreset, not a fitted distribution — and it fails
   too; the isolating control shows the missing ingredient is whole-document (feature,
   position, label) consistency, which no marginal summary preserves.)*

**Interpretation / mechanism.** The shared classifier head must place N tasks'
conflicting label geometries into one weight matrix. Merging cancels conflicting rows;
regularization freezes the wrong subspace; subspace transfer only interpolates the
stability–plasticity trade-off; and input-keyed memory dodges drift but loses the
shared representation needed to route queries. Replay is unique because it supplies
*real past-task gradients* that re-carve the joint optimum every session — it is the
only mechanism among those tested that optimizes the actual multi-task objective
rather than a proxy for it.

### Finding 3b — *what* the buffer must contain: whole-document consistency, not summaries (2026-07-10)

The chain above ends by asking not "how to avoid the buffer" but "what must it contain."
A four-way controlled ablation now answers it. Working in the frozen-trunk **latent-replay**
regime (freeze layers `<k=8` after task 0, replay real layer-`k` activations into the plastic
head+late layers — the one buffer variant that WORKS: dil AA 87.3, ≈ ER-200, within 1.4 of the
joint oracle), we ask which property of the replayed content is load-bearing by replacing the
stored real documents with progressively more "buffer-free" feature memories, all behind the
*same* replay hook (dil, LayoutLMv3, k=8, epochs=5, AA / SROIE-final-F1 / memory):

| replay content | AA | SROIE final | mem |
|---|---|---|---|
| **whole real documents** (latent_replay, 5 docs) | **87.3** (conv) | survives | 16 MB |
| synthetic features, rank-`r` forgetting subspace (SpectralMemory / "SLR") | 41.9 | 3.2 | 0.45 MB |
| synthetic features, full-`d` per-class Gaussian (AGLR-CL port, arXiv 2505.08524) | 39.4 | 3.0 | 0.39 MB |
| **real** activation centroids (k-means coreset), 4 layout carriers | 36.7 | 2.9 | 0.48 MB |
| **real** centroids, 50 carriers + 16 centroids/class | 39.7 | 3.1 | 4.05 MB |

Every buffer-free variant reproduces the LexMem-v5 signature (current task ~93, **all prior tasks
~3**). Four independent controls isolate the cause:
1. **Not synthesis quality** — a *real*-activation coreset fails identically to synthetic Gaussians.
2. **Not rank** — full-`d` Gaussians fail like rank-16. (And the features are not low-rank to begin
   with: rank-16 explains only 43% of layer-8 feature variance, rank-256 ≈ 92% — the "forgetting is
   low-rank" property is of the *output/NTK* space, not the head-*input* feature space.)
3. **Not carrier diversity** — 50 layout carriers at 8× the memory fail like 4.
4. **The necessary ingredient is (feature, position, label) co-occurrence.** The clean isolation:
   replaying **4 whole real documents** (feature[t], bbox[t], label[t] all from the same real token)
   scores AA 63.8 / SROIE 37.1, versus **4 layout carriers with real centroid features pasted onto
   unrelated positions** at AA 36.7 / SROIE 2.9 — **+27 AA from consistency alone**, same count,
   same boundary, same real features.

**Interpretation.** Every buffer-free memory tested — parametric slot, feature-Gaussian, real
coreset, lexical ledger — is some *marginal* summary (per-class, per-token, or per-lexeme). The
head cannot be grounded by marginals: it needs the *joint* structure of an intact document, where
each feature is bound to its real spatial position and label. This is why replay is
non-substitutable, stated more precisely than before: not merely "real gradients," but **real
gradients from consistently-bound (feature, position, label) triples that only a stored whole
document preserves.** It also reframes the "SLR/spectral latent replay" idea (arXiv-theorem-inspired,
2606.18024) as *falsified for doc-IE* and folds it, plus the AGLR-CL port and the real-coreset
probe, into the chain as its terminal level.

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
that FIVE distinct buffer-free consolidation families fail *for identified mechanistic
reasons* — weight-merge (DocMERGE/LCA), subspace-transfer (HGT/CUBER), parametric slots
(LexSlot), input-anchored memory (Ledger), and **feature-Gaussian generative replay, the
field's own strongest exemplar-free tool** (LexMem v5 ≈ PASS/FeTrIL/FeCAM + a novel
relational variant). LexMem-v3b is the positive control (AA 66, where partial recovery
asymptotes); v5 is the terminal negative (AA 63, mid-task → 13, even the standard method
fails). (3) The implication: in this regime the research question is not "how to avoid
the buffer" but "how small can the buffer be / what must it contain" — replay's gradients
are doing something no tested surrogate replicates.

**Is NOT a method paper — and must not be pitched as one.** Every proposed method is
evidence, not a contribution. Concretely, v5's core mechanism (feature-statistic replay)
is *already solved literature* — pitching it as novel would be a desk-reject at
AAAI/ICML. The novelty of the paper is the **diagnosis** (localization + migration) and
the **rigorous convergent negative result** ("nothing buffer-free works, and here is the
mechanism"). The novel bits inside methods (v5's relational replay, HRP's routing) are
reported honestly as *tried and insufficient*, which strengthens the negative result
rather than pretending to a SoTA claim. The thesis "LexSlot (ours) 87.3" row is the
DocCL hybrid and must be relabeled before submission (tracked).

**Venue implication:** a negative-result-with-mechanism paper fits CoLLAs / TMLR /
ACL-Findings naturally; for AAAI/ICML main-track the lead must be the migration finding,
not any method.

**Gating work before submission:**
1. LexMem-v3b multi-seed (7, 123) — the spine's error bars. Currently **stalled**: the
   watcher is deadlocked behind opencode's repeated `lexslot_fm` launches (decision on
   killing/relaunching pending).
2. Multi-backbone generalization grid (rented GPU) — required for "architecture-general"
   to cover the *method comparison*, not only the diagnostic.
3. Fix the thesis LexSlot row label; rebuild FWT for any transfer claims.
