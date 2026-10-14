# Research Synthesis: DocCL — everything measured so far, and the CVPR 2027 story

**Method:** pre-registered falsification programme (diagnose → predict → kill-test → construct)
**Participants:** ~430 training runs (`results/`), 4 backbones × 3 scenarios × 3 seeds for the
grid (68/72 cells), plus seed-42 diagnostics with three-seed confirmation at the stated points
**Date:** 2026-06 → 2026-10-14 | **Target:** CVPR 2027 (deadline ≈ 2026-11-13, 30 days)
**Companion docs:** `docs/RESEARCH_PLAN_2026-10_next_mechanism.md` (§0' = active plan),
`docs/IMAGE_SCOPE_PREREG.md` (Amendments 1–10), `docs/PILOT_STUDY_REPORT.md`,
`paper_cvpr/main.tex` (8 pp body, Overleaf-synced)

---

## Executive Summary

Forgetting in continual key-information extraction is output-localised (head + late layers),
architecture-general (LayoutLMv3, LiLT, BROS, BERT, and ViT), and *conserved*: protecting the
locus relocates the drift instead of removing it. That single fact predicts the whole remedy
ordering we measure — replay reaches the oracle, every buffer-free family (five of them, plus
five registered kill-tests) fails for one reason, and the only admissible compressor is one
that keeps each document's (feature, position, label) binding intact (CoLaR: 87.6 AA at 60 MB,
lossless vs raw replay). Every attempt to add intelligence *on top* of that memory — lexical
routing, parametric slots, read-side blends, drift-compensated head statistics, learned codecs —
is a clean null across three seeds. The reason is now quantified: on DIL the independent
per-dataset deployment (88.2) is within 0.5 of pooled training (88.7) and *above* it on two
backbones, so there is no transfer headroom left to win. The headroom is on **CIL-CORD (gap 14,
where replay is below naive)** and **Mixed (gap 5)**, and the code shows why: every buffer
stores labels frozen in the capture-session head space. The one mechanism we have not tested
is the one the paper lists as its limitation — refreshing those labels. That is the proposed
method addition for CVPR.

---

## Key Findings

### F1 — Forgetting is output-localised and architecture-general
**Prevalence:** 4/4 document backbones × 4 modality masks × 3 seeds; reproduced on ViT (3/3 seeds)
**Evidence:**
- Per-token CKA between consecutive checkpoints falls monotonically with depth: 1.00
  (text/layout/patch embeddings) → 0.78 (early) → 0.34 (mid) → 0.25 (last layer) → 0.16 (head).
- Old-task-Fisher-weighted displacement is 1–4 orders of magnitude larger at the head than at
  any encoder bucket; per-component test rejects uniform displacement, p < 0.01.
- Permutation tests fail to reject a shared location for LiLT (p = 0.38) and BROS (p = 0.51).
- ViT image CIL: head share of displacement 97–100 % on 3/3 seeds (`results/pilot/cv_vit_*`).
- Modality masks: C4 full / C3 text+layout / C1 text-only forget comparably (BWT −81/−88/−79).
  Removing text collapses training itself (c1_bert 0/3 seeds, c2_no_text 1/3) — text is the
  load-bearing modality; vision neither drifts (patch CKA 1.00) nor forgets.
**Implication:** the paper's spine. Any remedy that does not place an old-task gradient on the
head is predicted to fail; we then test that prediction.

### F2 — Protecting the locus relocates the drift (conservation)
**Prevalence:** DIL/LayoutLMv3 seed 42 for freeze arms; 3 seeds for the consolidation probe;
extended to 3 task orders and a long CIL stream (`fig:conservation`)
**Evidence:**
- Freeze everything after task 1: AA 41.1, BWT −0.0 (nothing forgotten, nothing learned).
- Freeze only head + late layers: AA 38.8, BWT −76.8; early/mid CKA collapses 1.0 → 0.19.
- Frozen locus + memory, no protection: 38.9 / −75.6 — indistinguishable; drift is the bottleneck.
- Uniform depth-scaled consolidation: 42.6. Same budget concentrated on head/late: 84.7;
  head-only 86.1; late-only 86.2 (3 seeds).
**Implication:** a buffer-free fix at the head alone cannot work. This is a *prediction* the
rest of the paper confirms, which is what makes it a diagnosis rather than an observation.

### F3 — Replay re-grounds the head and DIL is saturated
**Prevalence:** 4/4 backbones, 3 seeds; Independent upper bound now 4/4 backbones
**Evidence (DIL AA):**

| Backbone | Naive | Indep. (per-dataset) | Joint | ER | DER++ |
|---|---|---|---|---|---|
| LayoutLMv3 | 41.3 | 88.2 ± 1.0 | 88.7 | 87.9 | 88.0 |
| LiLT | 40.1 | **85.3 ± 0.7** | 83.8 | 85.7 | 84.4 |
| BROS | 40.3 | **88.8 ± 0.1** | 87.5 | 87.8 | 87.6 |
| BERT | 39.4 | ≈77.5 (2 cells finishing) | 77.2 | 77.1 | 77.4 |

- Independent ≥ Joint on LiLT and BROS: pooling three datasets buys **no** transfer on DIL.
  (Joint on LiLT is under-fitted by the shared early-stopping rule — footnoted in the paper.)
- Memory-side levers obey a conservation law: balanced losses 87.1, coverage selection 86.8,
  entity-weighted compression 87.1 — each trades FUNSD for SROIE ≈ 1:1, AA unchanged;
  three-seed check: base 87.8 = balanced 87.8 > k-centre 87.0.
**Implication:** "beating ER on DIL" is not a research goal — the task structure leaves ≤ 0.5 AA.
The Independent column makes that explicit and converts the question into *cost*: one model +
60 MB vs three models.

### F4 — A remedy's benefit is backbone-dependent; the phenomenon is not
**Prevalence:** 4 backbones × 3 seeds on Mixed and CIL-CORD
**Evidence:** naive-to-joint gap agrees to 2.5 points across backbones on Mixed (+33.7 / +31.1 /
+31.4 / +31.6) while EWC's benefit over naive spans 17 points with a sign flip: −11.8
(LayoutLMv3), −2.6 (LiLT), +3.5 (BROS), +5.3 (BERT). LwF helps CIL-CORD on BROS/BERT (+4/+5),
nothing on LayoutLMv3.
**Implication:** single-backbone comparisons of buffer-free remedies can land on either side of
zero; a multi-backbone protocol is a methodological requirement, not a nicety. (This is the
B+ scope decision of 2026-07-13 paying off.)

### F5 — Five buffer-free families fail for one mechanism
**Prevalence:** 5 families, 7 instances, 5 kill-tests; DIL/LayoutLMv3

| Family | Instance | Best AA | Why it fails |
|---|---|---|---|
| Weight merging | TIES / Fisher head merge | 21.6 (no-merge 21.9) | head vectors cancel; near no-op |
| Merge + realignment | LCA (ICLR'26, ported) | 41.9 | Gaussian realignment kills acquisition |
| Subspace transfer | HGT / CUBER | *(cell missing — `\vnum{--}`)* | protected subspace grows; head unconstrained |
| Parametric slots | LexSlot (no buffer) | 42.2 | isolated slots cannot re-ground shared head |
| Input-anchored memory | Ledger | FUNSD 21 / SROIE 23 | own-vs-cumulative gap is training-free |
| Feature-Gaussian replay | LexMem v3b | 66.0 | partial: dense-label tasks only |
| | LexMem v5 (+relational) | 63.2; mid-task 13.0 | marginal summary breaks doc consistency |

- Root cause: the head's output marginal on old tasks snaps to the just-trained task's gold
  marginal (cosine 0.997–0.998); never-re-exercised classes (KEY, HEADER) go to exactly 0.0 F1.
- Five pre-registered corrections (marginal matching, prior-ratio, per-doc EM, KL anchor,
  balanced-softmax) recover ≤ −0.1 AA. Saturation probe: old-class mass ≈ 1e-6, 2–3 orders
  below any branch threshold — information destroyed before a correction can act.
**Implication:** the search legitimately shifts from "which regulariser" to "what minimal object
must a buffer store".

### F6 — What a document memory must hold: whole documents; CoLaR is the admissible compressor
**Prevalence:** seed 42 ladder, 3-seed headline; Amendment 9 codec/position probes
**Evidence:**
- Real banked activations of 50 docs/task: 87.3 (k = 4). Summaries of the *same* store: rank-16
  spectral 41.9, per-class Gaussians 39.4, real k-means centroids 36.7–39.7.
- Single-variable control: 4 whole docs 63.8 vs identical features on decoupled layout carriers
  36.7 → **binding is worth +27 AA** at matched budget.
- CoLaR r = 128: 87.6 (86.8 ± 0.4 three seeds; see caveat M3) at 60 MB — matches ER 87.9
  (200 raw docs) and raw latent replay 87.3 at 163 MB (2.7×). r = 64: 80.6 at 32 MB.
- Learned per-token AE codec at matched bytes is far below per-doc SVD (AE c=64 @ 56 KB = 31.5
  token-F1 vs SVD r=16 @ 39 KB = 40.9; SVD r=32 @ 77 KB = 65.6). Shared codecs discard
  per-document structure.
- Position-axis compression breaks binding: labelled-tokens-only 79.6 @ 42 MB (SROIE 81→59);
  additionally dropping the 197 visual positions 61.2 @ 23 MB (SROIE 12).
- Drift-compensated per-class Gaussian head alignment: d5 74.6 vs uniform 75.9; d50 86.2 vs
  86.4 — null at both scales.
- Public-proxy replay (PLaR, zero private bytes): 58.9, coverage-limited (FUNSD 76.1, SROIE ≈ 7).
**Implication:** CoLaR is the paper's *constructive control*: it proves the negatives are not
effort-limited. Also a byte/privacy result: no raw image or token id is stored
(`store_input_ids` defaults False — verified in `latent_replay.py`).

### F7 — Intelligence on top of the memory is a null (routing, slots, read-side)
**Prevalence:** 3 seeds for routing; 3 variants × 3 seeds × 2 scales for CoLaSlot
**Evidence:**
- Lexical routing (OCR-signature sampler, pre-registered far@d50): +1.8 / +0.4 / −0.6 over
  uniform on seeds 42 / 7 / 123, mean +0.5 pp, inside the uniform spread → **KILLED**
  (Amendment 10, 2026-10-13).
- CoLaR + LexSlot (CoLaSlot) at d50/r128: +0.19 AA for +77 % runtime. At d5/r64: +1.9 on seed
  42, −0.66 / −10.93 on seeds 7 / 123 → NO-GO.
- kNN readout blend: inert at λ = 0.3, destructive at λ = 1 (46.2). Metaplastic consolidation
  dose-dependently harmful 85.9 → 79.5.
**Implication:** consistent with F3 — there is nothing left to route *toward* on DIL. These are
reported as nulls in §7/§8 of the paper and keep the headline memory-only.

### F8 — CIL-CORD is a floor regime and Mixed is partial; replay is *below naive* on CIL-CORD
**Prevalence:** 4 backbones × 3 seeds; LayoutLMv3 numbers below

| Scenario | Naive | EWC | LwF | ER | DER++ | ER+C-Flat++ | Best | Joint | Gap |
|---|---|---|---|---|---|---|---|---|---|
| CIL-CORD | 19.0 | 0.5 | 18.8 | **16.0** | 18.2 | 15.9 | 19.0 (naive) | 33.3 | **14.3** |
| Mixed | 33.9 | 22.0 | 35.2 | 60.3 | 60.8 | 62.2 | 62.2 | 67.5 | **5.3** |

- Same picture on every backbone (CIL-CORD: all methods 16–24 vs joint 32–33).
- Prompt/LoRA families: CODA-Prompt 0.0 / 8.1, L2P 1.1 / 11.5, O-LoRA 4.7 / 13.4, CL-LoRA 8.9 /
  31.4 on CIL-CORD / Mixed — head drift, routing cannot fix (memory `clue-prompt-family-head-drift`).
- Mechanism (code-verified): `CIL_LabelRemapper` masks not-yet-introduced entities to O; ER,
  DER++ and CoLaR all bank labels in the capture-session head space, so a replayed session-1
  document teaches "O" for every class introduced later. The paper states this and lists the
  continual-NER fix as untested. CoLaR, CPFD, IS3 have **zero cells** on CIL-CORD/Mixed.
**Implication:** this is the only place a method can move numbers by more than a point, and the
mechanism is already identified. → Recommendation R1.

### F9 — The anatomy transfers to image CIL; the method ranking does not
**Prevalence:** CIFAR-100 (10×10) and ImageNet-R (10×20), 3 seeds, bs 16 local recipe
**Evidence:** joint 89.5 / 77.6; ER@2000 76.2 / 54.8; DER++@2000 76.5 / 52.8; SLCA 87.5 / 65.0;
slca_noca 38.0 / 25.8; CoLaR raw bank (500/class) 80.0 on CIFAR, trailing SLCA by 7.5.
RCA probe: RP-ridge on full-data features of the *adapted* trunk 88.6 > frozen RanPAC 85.2;
from the stored 50/class only 80.7 → root cause is readout statistics at 10 % coverage, not
trunk damage (representation gap −3.4: adaptation *improves* features).
**Implication:** on images a buffer-free Gaussian-head method beats latent replay; on documents
the same head-statistics idea is null (F6, B1/B2). The discriminating ingredient is binding
(token-level structured output vs one label per image). This 2×2 inversion is a candidate
figure/paragraph that a CVPR audience will recognise immediately.

---

## Insights → Opportunities

| Insight | Opportunity | Impact (CVPR) | Effort | Status |
|---|---|---|---|---|
| Replay < naive on CIL-CORD because buffered labels are stale (F8) | **CoLaR-R**: re-annotate the drift-free memory with the current upper stack (oracle + pseudo variants); ER + refresh control | High — the only ≥ 1-pt gain available; closes stated limitation | Med (2–3 d code, 12–20 GPU-h) | planned, §0' |
| CPFD / IS3 cited but never run (F8) | Run both on CIL-CORD + Mixed, 3 seeds | High — preempts the obvious reviewer ask | Low (configs exist; ~6 GPU-h) | queue tonight |
| Independent ≥ Joint on 2/4 backbones (F3) | Finish BERT cells; replace 3 `\tbd` in Table 1; one sentence in §6 | Med — sharpens "DIL saturated" | Trivial | grid finishing |
| Head-statistics methods win on images, null on documents (F6, F9) | 2×2 "method-rank inversion" paragraph + mini-table (SLCA vs CoLaR × images/documents) | High for a vision venue; reframes binding as the domain-discriminating ingredient | Low (data exists; SLCA on documents optional, see Q3) | writing |
| Visual positions in replay: dropping them collapses SROIE to 12 though vision neither drifts nor forgets (F1, F6) | *Silent-modality probe*: keep positions, replace content with task-mean visual latent → information vs structure | High if information (a genuine vision finding); none if structure | Low (2 runs, seed 42) | deferred, after R1 |
| Frozen-PTM baselines (SimpleCIL/RanPAC) are image-only in the repo | Token-level port; expected to fail | Low–Med (reviewer-preemption only) | Med (1 d) | optional |
| CoLaR headline 86.8 ± 0.4 vs new uniform cells 86.62 ± 0.25 (M3) | Reconcile before submission | Med (credibility) | Trivial | open |
| HGT/CUBER ledger cell `\vnum{--}`; `\todo` in app:extra | Fill or delete | Med | Low | open |

---

## Regimes (the "segments")

| Regime | Scenarios | Characteristics | What is needed | Share of paper |
|---|---|---|---|---|
| **Saturated** | DIL (all backbones) | Indep ≈ Joint ≈ replay; forgetting = pure drift; any buffer that keeps binding reaches the oracle | cost/privacy argument only (CoLaR) | §4–§7 as written |
| **Partial** | Mixed | replay closes ~85 % of gap; residual sits at CIL transitions + FUNSD revisit | label-aware replay | new §7 row |
| **Floor** | CIL-CORD | every method 16–24 vs 33; O boundary moves each session; replay hurts | label refresh + continual-NER comparators | new §7 row + ledger |
| **Image CIL** | CIFAR-100, ImageNet-R | anatomy identical; buffer-free Gaussian head wins | inversion paragraph (scope claim, not a method claim) | 1 paragraph + supp table |

---

## Recommendations (ordered)

1. **R1 — CoLaR-R on CIL-CORD and Mixed (the method addition).** Oracle refresh (store native
   full labels, ~0.5 KB/doc) bounds the stale-label cost; pseudo refresh (CPFD entropy rule
   applied to the memory, teacher-free) is the deployable variant. Controls: CoLaR-plain,
   ER + oracle refresh, CPFD, IS3. Bars pre-registered in `RESEARCH_PLAN…` §0'. Window Oct 18–26.
   *Why:* F8 — the only ≥ 1-pt headroom; mechanism code-verified; closes the paper's own
   stated limitation; diagnosis-consistent ("the memory must hold whole documents *and* live
   labels").
2. **R2 — Baselines regardless of R1's outcome:** CPFD, IS3, CoLaR-plain on CIL-CORD/Mixed,
   3 seeds (~8 GPU-h). Start tonight when the Independent grid frees the slot.
3. **R3 — Writing, no GPU:** fill Table 1 Indep. cells (LiLT 85.3, BROS 88.8, BERT pending);
   add the "Independent ≥ Joint on two backbones" sentence; reconcile M3; fill/delete HGT cell
   and app:extra `\todo`; draft the F9 inversion paragraph.
4. **R4 — Silent-modality probe** (2 runs) only after R1's three-seed verdict; include only if
   it says *information*.
5. **R5 — Do not pursue** on this cycle: more DIL mechanisms (F3), MoE/LoRA-expert routing,
   PQ codec, frozen-k analytic heads, token coresets (F6 pre-falsifies), prompt-family variants.

---

## Questions for Further Research

- **Q1.** How much of the CIL-CORD gap is stale labels vs the O-boundary shift in *live* data?
  Oracle-refresh vs CPFD separates the two (R1 answers this).
- **Q2.** Does label refresh help ER as much as CoLaR? If yes, the contribution is a protocol
  fix for all replay; if no, latent replay's drift-free substrate is doing the work (stronger).
- **Q3.** Does SLCA fail on documents as F6 predicts? One port + 3 runs would turn F9 from an
  observation into a tested inversion. Optional for CVPR; natural for the thesis.
- **Q4.** Are the 197 visual tokens information or structure for replay (R4)?
- **Q5.** Is Joint on LiLT genuinely under-fitted (Indep 85.3 > Joint 83.8), and should the
  paper's early-stopping footnote become a stated limitation?

---

## Methodology Notes and Caveats

- **M1 Protocol.** AdamW 5e-5, wd 0.01, clip 1.0; each task trained to convergence (val-F1 early
  stop, patience 2, δ 0.1, 100-epoch ceiling never binds); mean ± sd over seeds {42, 7, 123}.
  FUNSD/SROIE lack validation splits → stopping reads the eval set (mild optimistic bias shared
  by every method; stated in paper).
- **M2 Coverage.** Grid 68/72 cells; missing cells disclosed, none a DIL replay cell. Finding 2
  freeze arms and the memory ladder are seed-42 diagnostics with three-seed checks at the stated
  points.
- **M3 Headline — RESOLVED 2026-10-15.** Two recipes were conflated. Paper protocol (10 epochs,
  early-stopped): seeds 42/7/123 = 87.58/88.30/87.38 → **87.75 ± 0.47** (= the "87.8 base" of the
  conservation check). Amendment-9/10 control recipe (5 epochs): 86.38/86.62/86.87 → 86.62 ± 0.25.
  The stray "86.8 ± 0.4" matched neither and is replaced by 87.8 ± 0.5 in paper and thesis.
  Hygiene: the 10-epoch seed-42 dir was overwritten by the 5-epoch control; seeds 7/123 have
  train.log but no metrics.json → re-run queued (`colar_e10`, R2c) so the headline is disk-backed.
- **M4 Order sensitivity.** Reversing DIL moves naive BWT −81 → −75; all single-order claims
  carry the caveat (`fig:conservation` covers three orders).
- **M5 Backbone-specific.** Latent replay is lossless on LayoutLMv3/BERT/BROS; LiLT's separate
  layout stream is not stored → replay approximate there. Joint on LiLT under-fitted (M-footnote).
- **M6 Independent grid recipe.** LiLT/BROS/BERT single-task cells use 15 epochs, bs 1, grad
  checkpointing (VRAM); LayoutLMv3 cells from the cached single-task table (n = 13/3/3). Report
  as upper bounds, not as a matched-budget comparison.
- **M7 Image scope.** bs 16 local recipe, not the published bs 128; SLCA/RanPAC numbers are
  therefore below published (91.5/92.2) and quoted as such.

---

## The CVPR 2027 story (one paragraph the reviewer should be able to repeat)

*Where* forgetting lives in a visually-rich document encoder (head + late layers, on every
backbone including ViT), *what happens* when you protect it (drift relocates — a conservation
law), *what that predicts* (replay reaches the oracle; five buffer-free families and five
kill-tests fail for one readout-snap mechanism), *what a memory must therefore hold* (whole
documents — binding is worth +27 AA — compressed per document, CoLaR 87.6 @ 60 MB, lossless, no
raw bytes), *what does not help on top* (routing, slots, read-side blends: three-seed nulls,
because the independent upper bound shows DIL has no transfer headroom), and — the addition
this cycle — *the memory's labels go stale where the label space moves*, which is why replay is
below naive on class-incremental CORD, and a drift-free latent memory is exactly the object you
can re-annotate. Contributions 1–4 exist in the current draft; contribution 5 (CoLaR-R) is the
R1 programme. If R1 fails its pre-registered bar, contribution 5 becomes a sixth certified null
and the paper ships as the diagnostic it already is.
