# Research Plan — Next Mechanism Beyond CoLaR / LexSlot

Date: 2026-10-14. Owner: this session. Related: `STATE.md`, `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`,
`docs/KT_2026-07-14_colar_lexslot.md`, `docs/CACOLAR_T1_T2_2026-08-20.md`.

## 0''. TOURNAMENT VERDICT (2026-10-15 00:30) — idea selection closed

Two-round tournament (5 lenses × ≥12 candidates = 67, red-team + recombination; files
`sources/brainstorm_r1_*.md`, `sources/brainstorm_r2_redteam.md`). Converged; no further rounds.

**New measured fact.** CIL-CORD re-shows 100 % of its documents (every later session is a
subset of session 0's 800 receipts); Mixed re-shows FUNSD fully (revisit) and CORD 798/800.
Hence the "oracle" label refresh is implementable from the stream as a ≤ 1 MB per-document
label ledger — not privileged information. Caveat for the paper: disjoint-slice CNER
protocols (CFNER/ExtendNER) get nothing from a ledger; only the label-free variants apply there.

| Rank | Package | Headline claim (falsifiable) | GPU | Kill |
|---|---|---|---|---|
| **P1** | **Live labels for a drift-free memory** | Replay is below naive on class-incremental document streams because buffered labels are stale; keeping them live — exactly via the label ledger (= `colar_r_oracle`), or label-free via pseudo-refresh / sliced-softmax replay CE (`colar_r_sliced`, new) — recovers ≥ 4 AA of the 14-pt CIL-CORD gap and ≥ 1 on Mixed, three seeds. Controls: CoLaR-plain, CPFD, IS3, ER+ledger, stale-label audit (GPU-free), head-row gradient-conflict hook. | ~110 h queued + ~50 h | ledger ≤ plain + 1 |
| P2 | Evaluation methodology for document CL | Joint−Independent headroom must be certified before ranking (≤ 0.5; negative on LiLT/BROS); head-statistics methods invert rank across domains (win on image CIL, null on token-level documents) because the readout needs binding. SLCA-token port + 3 DIL runs. | ~8 h | SLCA-token ≥ CoLaR − 1 on DIL |
| P3 | Where forgetting lives inside the sequence | Drift is text-position-localised (visual positions CKA ≈ 1.0); visual positions are structural replay carriers (silent-modality probe); conservation reproduces on ViT. | ~15 h | visual-pos CKA within 0.05 of text; ViT early/mid CKA > 0.9 under freeze |

Convergence: #1 = P1, stable across rounds and strengthened by the re-show measurement.
**If the running CoLaR-R cells are null, #1 becomes P2** and P1 reduces to a ledger row plus
the stale-label audit; P3 unaffected. Killed/demoted after red-team: depth-targeted replay
gradients, keep-correctable cap, abstention refresh, modality-asymmetric ranking.

Queue consequence: R2 (30 cells) runs to ~Oct 19–20 at the observed ~9 min/epoch; cap seeds
7/123 of CPFD/IS3 on Mixed at `method.epochs=5` if Gate 1 slips. `scripts/run_r2b_queue.sh`
appends the sliced-softmax control (6 cells) after R2.

## 0'. REVISION (2026-10-14 evening) — supersedes §§2–3a below

**Retraction.** Re-reading `docs/IMAGE_SCOPE_PREREG.md` Amendment 9 shows two of the three
directions below are already falsified on documents: **A** (Gaussian head re-alignment) is the
B2 cell — drift-compensated per-class Gaussian head alignment at d=50 = 86.2 vs uniform 86.4,
null; **B** (token-level coreset) is predicted dead by D1/D2 — dropping context tokens collapses
SROIE 81→59→12. **D** (NC-ETF head) has no evidence either way but targets a scenario (DIL)
with 0.5–1.1 AA of headroom. Nothing on DIL can "beat" anything: Independent 88.2, ER 87.9,
Joint 88.7.

**Where the headroom actually is (measured, LayoutLMv3, 3 seeds):**

| Scenario | Naive | ER | DER++ | Best any | Joint | Gap |
|---|---|---|---|---|---|---|
| CIL-CORD | 19.0 | **16.0** | 18.2 | 19.0 (naive!) | 33.3 | **14** |
| Mixed | 33.9 | 60.3 | 60.8 | 62.2 (ER+C-Flat) | 67.5 | **5** |
| DIL | 41.3 | 87.9 | 88.0 | 88.4 | 88.7 | 0.5 |

Replay *hurts* on CIL-CORD (ER < naive). Mechanism, from the code: every buffer (ER, DER++,
CoLaR) stores labels in the **capture-session head space** (`CIL_LabelRemapper` masks
not-yet-introduced entities to O). At session t the replayed document teaches "these tokens
are O" for every class introduced after capture. The paper names this ("replayed documents
carry the *old* labelling of the new classes") and lists the fix as untested. CoLaR, CPFD and
IS3 have **never been run** on CIL-CORD or Mixed (CPFD/IS3 are registered with configs; zero
cells in `all_runs.csv`).

**New primary direction — Label-Refreshed Latent Replay (working name CoLaR-R).**
The stored latents sit below the frozen boundary, so they are *exactly* what the current model
would compute on the original page; the upper stack can therefore re-decode the memory and
re-annotate its stale O tokens with fresh pseudo-labels for classes learned since capture —
without a single raw byte. Features never go stale in CoLaR; only labels do. Two variants:

- **Oracle refresh** (upper bound on the mechanism): store the document's full native labels
  at capture (`labels_full`, int8, ~0.5 KB/doc); at session t translate into the current head
  space. Cheap plumbing in `cil_remapping.py` + `latent_replay.py`. Measures the *entire*
  stale-label cost.
- **Pseudo refresh** (deployable): before each session, forward every stored doc through the
  current upper stack; O tokens whose max-prob class is a post-capture class and whose
  entropy is below a per-class threshold (CPFD rule, one pass) get relabelled. Teacher-free.

Controls that make it a paper result, not a trick: (i) ER + oracle refresh (is it CoLaR-specific
or buffer-general?), (ii) CPFD and IS3 as the continual-NER baselines on CIL-CORD and Mixed,
(iii) CoLaR without refresh (the 18-ish number we expect).

**Pre-registered bars.** CIL-CORD: CoLaR-R (pseudo) ≥ naive +4 AA and ≥ CPFD/IS3, three
seeds, all seeds same sign. Mixed: ≥ ER+C-Flat++ 62.2 by ≥ 1.0. Oracle must exceed pseudo
(otherwise the pseudo-labeller is noise). Kill: pseudo ≤ CoLaR-plain +1.

**Cost.** CIL-CORD runs are 3–15 min/session × 5; Mixed 3–7 min × 6. Full programme
(CoLaR, CoLaR-R-oracle, CoLaR-R-pseudo, ER-oracle, CPFD, IS3) × 2 scenarios × 3 seeds =
36 runs ≈ 12–20 GPU-h on the local box. Fits in the Oct 18–26 windows of §3a; §3a's A/D
slots are replaced by this. Secondary backbones only for the winner.

**CVPR framing.** Adds the one thing the paper lacks: a method result on the hard scenarios,
tied to the diagnosis (the readout-marginal snap, Finding 2) — "the memory must hold whole
documents *and* live labels". Also closes the stated limitation and gives CVPR reviewers the
continual-NER comparators. Optional vision hook if time permits (§0'b).

**§0'b — deferred probes (only after the above lands):**
- *Silent-modality probe.* Amendment 9 D2: dropping the 197 visual positions from replay
  collapses SROIE to 12 although the visual stream neither drifts (patch CKA 1.00) nor forgets
  (C3 ≈ C4). One control (keep positions, replace content with task-mean visual latent)
  decides whether the visual tokens carry *information* or only *structure* for replay.
  High CVPR value if information; 2 runs, seed 42.
- *Frozen-PTM baselines on documents* (SimpleCIL/RanPAC exist for images only, `frozen_ptm.py`
  uses `pixel_values`). Token-level port ≈ 1 day; expected to fail (features need fine-tuning
  for BIO). Reviewer-preemption value only.

## 0. One-paragraph framing (original, 2026-10-14 afternoon)

CoLaR (87.6 AA) and LexSlot (87.3) sit ~1 point below Joint (88.7) and ~0.5 below the
Independent per-dataset upper bound (88.2). Three NO-GOs (CoLaSlot d50/r128; CoLaSlot d5/r64;
lexical replay routing) show that **neither more parametric slots on top of CoLaR nor a smarter
sampler over its memory** moves the needle. The headroom to close is 0.5–1.1 AA on DIL. The
diagnostic says forgetting is **head-localized** and migrates when the head is naively protected.
Two levers that we have *not* probed, each surfaced by the surveys and each falsifiable in
≤2 days, are:

1. **Head-side correction** (analytic / Gaussian re-aligned classifier) stacked on an existing
   replay backbone — orthogonal to replay, cheap, matches our diagnostic.
2. **Per-token coreset replay** (byte-matched to CoLaR's 60 MB) — tests whether CoLaR's
   whole-document unit is the ceiling or just the floor.

Everything else surveyed is lower-leverage against our setup: MoE / LoRA-expert routing
(unlearned token-level routers risk collapse on 3 tasks), closed-form ridge-only methods (need
a frozen backbone — kills late-layer plasticity we already rely on), prompt-family methods
(capped at AA 30 on DIL in our matrix — head-drift memory).

## 1. Research question

*Can a head-centric correction, bolted onto CoLaR or ER, close the 0.5–1.1 AA gap to Joint on
DIL (FUNSD→SROIE→CORD), on LayoutLMv3, under the 60 MB per-task buffer ceiling and three-seed
multi-backbone verification?*

Secondary: *Does per-token coreset replay (same bytes) match or beat per-document SVD?* If yes,
our Finding 3b (whole-doc consistency) weakens and the paper's constructive-control story
changes. Either outcome is publishable.

## 2. What the surveys actually recommend (filtered)

Deduplicated and ranked against *our* constraints (6 GB VRAM, token-level BIO, three-seed cost,
head-locus prior):

| # | Direction | Lever | Falsifier (one day) | Why it fits |
|---|-----------|-------|---------------------|-------------|
| **A** | **SLCA++ head on ER / CoLaR** | Replace plastic classifier with per-class (μ, Σ)-aligned Gaussian head; store statistics only. ~60 kB total, essentially free. | If logistic refit on buffer logits ≈ Gaussian head within 0.3 AA, the Gaussian prior adds nothing. | Direct head-locus intervention, orthogonal to replay, byte-cheap, no new routing. |
| **B** | **Core-Tokensets replay at 60 MB** | Token-granularity coreset (gradient-matched or random control) in place of 50 docs × SVD. 60 MB ≈ 20 k tokens. | If random 60 MB tokens matches gradient-matched within 0.3 AA, selection ≠ signal; just volume. If 20 k tokens < 50 docs by ≥0.5 AA, Finding 3b holds. | Tests whole-doc-consistency claim directly. Cheap: no new modeling. |
| C | ACIL / DS-AL analytic head over **frozen late layers** | Freeze layers k+ after task 1; closed-form ridge on layer-k. | Compare frozen-k alone vs frozen-k + analytic head. If freezing alone within 0.5, the analytic head isn't the lift. | Backbone-freeze may cost ≥ 1 AA on DIL (we have plastic late layers for a reason). Still worth one seed. |
| D | **Progressive Neural Collapse head on CoLaR** | ETF-pinned classifier geometry each step; replay perturbs encoder only. | One loss term, no new memory. If no gain, kill in 2 h. | Head intervention, no cost. |
| E | PQ-codec CoLaR (REMIND lineage) | Product quantization instead of per-doc SVD at matched 60 MB. | Amendment 9 killed *learned* AE at matched bytes; PQ is non-learned. One-day falsifier. | Byte-matched follow-up to Amendment 9. Negative result still publishable. |

**Dropped from the surveys and why:**
- MoLE-CIE / LD-MoLE / D-MoLE / TreeLoRA — LLM-tuning or three-task routing collapse risk; porting cost not amortized by thesis timeline.
- VQ-Prompt — prompt family capped at AA 30 on our DIL.
- Influence-guided + CGM — needs cached per-task gradients (10–50 MB) and reopens the sampler question we just killed; defer.
- RanPAC / F-OAL / SPARCL — frozen-backbone assumption we can't honor without separate ablation.
- Boost-PFT, KMR — multi-view ensemble / rumination are additive to backbone and don't address head-locus directly.

## 3a. Deadline constraint (overrides §3)

Target venue: **CVPR 2027**. Deadlines (cvpr.thecvf.com, read 2026-10-14): **registration
2026-11-10**, **paper 2026-11-16 AoE**, supplementary 2026-11-23. That is **33 days** from this plan;
the registration deadline fixes title/abstract six days earlier. Operating loop: `docs/RESEARCH_OPS_LOOP.md`. The paper
already exists (`paper_cvpr/main.tex`, 8 pp body, Overleaf synced) with CoLaR as constructive
control, so the research chain only has to produce *at most one* new row — or a certified null.

Compressed schedule (GPU slot frees ~2026-10-14 23:00 when the Independent grid finishes):

| Window | GPU work (one slot, 6 GB) | Writing work (parallel, no GPU) |
|--------|---------------------------|---------------------------------|
| Oct 15–17 | Implement A (Gaussian head) + D (NC-ETF head); unit tests on synthetic tensors; seed-42 smoke on LayoutLMv3 DIL. | Fill Table 1 `\tbd` cells from finished grid; reconcile CoLaR headline 86.8 vs 86.62. |
| Oct 18–21 | Three-seed A, D, plus A's logistic-refit control (9 DIL runs ≈ 9 × 2.5 h ≈ 1 day GPU each batch). **Gate 1 (Oct 21):** pass ⇒ go to secondary backbones; fail ⇒ B. | Draft §4 method subsection for A and D speculatively (one page, swappable). |
| Oct 22–26 | **If Gate 1 passed:** winner on LiLT/BROS/BERT DIL × 3 seeds (9 runs). **If failed:** implement B (token coreset) + random-token control; three-seed primary (6 runs). | Draft falsification-ledger row; §7 discussion reconciliation paragraph. |
| Oct 27–30 | **Gate 2 (Oct 27).** Remaining ablation controls for whichever passed; or, if nothing passed, run B's CORD-only diagnostic to certify Finding 3b. | Integrate numbers; rebuild; 8-page budget check; Overleaf sync. |
| Oct 31–Nov 6 | Buffer for failed/re-run seeds. No new mechanisms after Oct 30. | Full read-through at Q1-reviewer bar; supp PDF; abstract/contributions final. |
| Nov 7–12 | — | Freeze, proofread, co-author pass, submit ≥24 h early. |

Explicit scope cuts vs §3: **C (frozen-k analytic head) and E (PQ codec) are dropped** from the
CVPR cycle — both reopen questions (backbone freeze; Amendment 9) the paper already closes, and
neither fits before Oct 30. They move to `ROADMAP.md` Deferred for the thesis / a workshop paper.
Secondary backbones run **only for the one winner**, never for controls.

Fallback already in hand: if Gates 1 and 2 both fail, the submission is the current diagnostic
paper plus a third three-seed null row (head-side correction) — the falsification ledger grows
and the "headroom is small" framing (§3, Week 3) becomes the discussion. No new writing risk.

## 3. Plan: three-week falsification chain (reference; §3a is the operative schedule)

**Pre-registration contract.** All runs are three seeds (42, 7, 123). Primary: LayoutLMv3 DIL
(FUNSD→SROIE→CORD). Secondary: LiLT, BROS, BERT only if primary passes. Verdict rule: a cell
*passes* if seed-mean AA > CoLaR 87.6 by ≥ 0.5 AND sign-agrees on all three seeds AND fits
≤ 60 MB per task. Anything that depends on frozen-k reports an auxiliary row for the frozen-k
*alone* control.

### Week 1 — Head-centric probes (A, D in parallel)

Days 1–2: implement **A (CaCoLaR-Head)**:
- During training, maintain per-class running (μ_c, Σ_c) over the final hidden state at every seen token.
- After each task, refit the classifier W = argmin via a shrinkage ridge on these statistics (closed-form).
- Store (μ, Σ) per class, not per sample. 25 classes × 768 dim × 4 bytes ≈ 77 kB total.
- Falsifier control: W = logistic refit on current buffer logits (ER-only, no CoLaR). If equal, kill Gaussian.

Days 3–4: implement **D (NC-Head on CoLaR)**:
- Fix classifier to a 25-way ETF simplex (W = sqrt(25/24) × (I - 1/25 · 1 1ᵀ)); encoder only.
- No new memory. One extra loss term weight.

Day 5: run three-seed primary. Decide.

### Week 2 — Byte-matched replay unit (B, E in parallel if A failed; else B alone)

Days 1–2: implement **B (TokenCoLaR)**:
- Per-task coreset of 20 k tokens (layer-k embedding + label + box + layout signature). Three selection modes: random (control), herding on layer-k, gradient-matched via GCR.
- Replay = sample a mini-batch of stored tokens as auxiliary token-classification loss.
- Memory check: 20 k × (768·4 + 4·4) ≈ 60 MB. Clip or sub-sample to the exact CoLaR byte budget.

Days 3–4: implement **E (PQ-CoLaR)** only if A/D both failed. Reuses CoLaR io path.

Day 5: three-seed primary. Decide.

### Week 3 — Secondary backbones + ablations

If any of A / D / B / E cleared the pre-reg bar on LayoutLMv3:
- Run the winner on LiLT / BROS / BERT (three seeds × 3 datasets × 3 backbones = 27 cells, reuses `scripts/run_independent_grid.sh` scaffolding).
- Add the mandatory ablations: controls listed next to each falsifier above; the paper has to show that the *mechanism*, not just the training set-up, is the lift.

If **none** cleared the bar:
- Finding gets reframed: the Independent / Joint / CoLaR triangle is tight because the task
  structure leaves ≤ 1 AA of true transfer. Paper's framing becomes **"the headroom is small
  and we certify why"** — a diagnostic refinement. This is still a submission-ready story at
  the B+ scope.

## 4. Paper structure (if a winner emerges)

Minimum delta over the current CVPR draft: add one method section + one results subsection +
one row in Table 1. Preserve the diagnostic-led spine.

| Section | What changes |
|---------|--------------|
| Abstract | Add one sentence naming the mechanism (head-align or token-coreset). |
| §1 Introduction | One contribution bullet; replace softened pre-registered-null language if a mechanism holds. |
| §4 Method | New subsection — the winner only. 1 page. |
| §6 Experiments | New row in main table; add the falsification-control row the pre-reg demands. |
| §7 Discussion | Reconcile with CoLaR / LexSlot NO-GOs. Explain why the *locus* mattered more than the *carrier*. |
| §8 Limitations | Keep CoLaSlot NO-GO as evidence; add any new negative. |

## 5. Task graph

```
A-impl  ─┐
D-impl  ─┼─> three-seed primary ─> (pass?) ──YES──> Week-3 secondary backbones ─> paper edit
B-impl  ─┤                                    └──NO──> B/E stage ─> three-seed primary ─> ...
E-impl  ─┘                                                           (pass?)
```

Parallelizable: A ∥ D (both head-side, no shared code). B ∥ E (both memory-side, both reuse
CoLaR io). Not parallelizable with the grid currently running (independent-grid is saturating
the VRAM slot).

## 6. Risks and pre-registered kill criteria

| Risk | Kill criterion |
|------|----------------|
| Gaussian head overfits to O-class on token-level BIO | If post-hoc logistic ≥ Gaussian within 0.3 AA, drop Gaussian. |
| Token coreset loses whole-doc layout context → regresses on CORD | If TokenCoLaR < CoLaR on CORD by ≥ 1 AA while matching on FUNSD/SROIE, Finding 3b *holds* — paper angle; not a research dead end. |
| NC-ETF head hurts token classification (class imbalance O vs entities) | Known failure mode in NC literature on long-tail. If seed-42 AA < 85, abort before full three-seed. |
| Three-week budget bleeds past deadline | Weeks 1 and 2 are each one week of GPU time + one week of implementation. Hard stop at 3 weeks; if nothing cleared, pivot to the "headroom is small" reframing. |

## 7. Datasets, metrics, baselines (unchanged)

- Datasets: FUNSD, SROIE, CORD (DIL); LayoutLMv3 primary, LiLT / BROS / BERT secondary.
- Metrics: Average Accuracy (AA), Backward Transfer (BWT), seqeval token-F1 per class.
- Baselines in Table 1 already: Naive, Indep., Joint, EWC, LwF, ER, DER++, ER+C-Flat++, O-LoRA, CL-LoRA, DocCL (legacy), LexSlot, CoLaR.
- No new baselines — this plan tests head-side and token-granularity mechanisms, not another method family.

## 8. Decisions taken / open

Resolved 2026-10-14 (CVPR 2027 target, ~30 days):
- Schedule is §3a. A + D first, B only on Gate-1 failure. C and E deferred.
- Independent grid runs to completion tonight (3 cells left); A/D implementation starts on the freed slot.

Still open for the user:
1. If B's token-coreset result contradicts Finding 3b (whole-doc consistency), the paper's
   constructive-control story changes — flag before writing it up.
2. Confirm the deadline date once the CVPR 2027 CFP is published; the Oct 30 "no new mechanisms"
   cut moves with it.
