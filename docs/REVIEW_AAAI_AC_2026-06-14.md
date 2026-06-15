# Area-Chair Review — DocCL: Per-Component Forgetting Diagnosis for Tri-Modal Document Encoders

**Reviewer role:** Senior reviewer / Area Chair (AAAI / NeurIPS standard)
**Artifacts reviewed:** Thesis draft (`thesis/chapters/chapter1–7.tex`) + accompanying code (`doccl/`) + internal reports (`docs/PILOT_STUDY_REPORT.md`, `docs/THESIS_HANDOFF_*`, `docs/DOCCL_RESEARCH_RECORD.md`)
**Date:** 2026-06-14
**Recommendation:** **Reject** (numeric **3/10**, confidence **4/5**)

> Calibration note: I am reviewing this *at top-venue standard, as requested*. This is a Master's
> thesis-in-progress with explicitly preliminary results, and the issues below are fixable. The
> review is deliberately unsparing because a diagnostic paper lives or dies on construct validity,
> and I read the code, not only the prose. Where the work is genuinely sound I say so.

---

## 1. Summary of Claims (in my words)

The thesis claims three contributions: (1) **the first per-component diagnosis** of catastrophic
forgetting in a tri-modal document encoder (LayoutLMv3), decomposing forgetting into **text,
visual, layout, and fusion** contributions via controlled modality ablation of a single backbone,
read out by per-layer CKA, per-group Fisher information, and standard CL metrics; (2) **DocCL**, a
mechanism-targeted CL remedy whose design is *derived from* the diagnosis; and (3) the **FCS-CL
benchmark** (FUNSD/CORD/SROIE under CIL/DIL/Mixed scenarios) with reproducible tooling. The
empirical core (Ch. 6) reports that naive sequential fine-tuning forgets near-totally (88→1.8 F1
in one step), that drift concentrates in the **late layers and classifier head** (not fusion), that
the cross-condition test cannot separate full-multimodal from ablated forgetting, and that the
study therefore **falls back to a characterization-only contribution**; replay (ER) is the
strongest baseline, reaching the joint oracle on DIL.

The honest one-line version: **the paper currently delivers "LayoutLMv3 forgets catastrophically,
mostly in the late layers and head, and replay helps most."** The novel parts that justify the
framing — the *multimodal* (text vs. visual vs. fusion) decomposition and the *method* — are either
unmeasured by the instrument or not yet implemented.

---

## 2. Scores (1–5, 5 = best)

| Axis | Score | One-line justification |
|---|---|---|
| **Soundness** | **2** | The headline diagnostic instrument has two demonstrable construct-validity defects and the central statistical test is mathematically incapable of rejecting its null. |
| **Contribution / Significance** | **2** | What is *delivered and valid* (catastrophic forgetting; late-layer/head drift; replay wins) is largely known/expected; the novel multimodal-decomposition claim is not supported by the measurement. |
| **Novelty** | **2** | Novel in *intent*, but the delivered finding re-confirms Ramasesh (ICLR 2021) on a new backbone; the method (the differentiating contribution) does not exist yet. |
| **Presentation** | **3** | Prose is clear, structure is good, and the preliminary status is flagged honestly — but the methodology (8 groups, text-only baseline) contradicts the implementation (7 groups, null-input baseline), and Ch. 1 advertises contributions Ch. 6 does not deliver. |

---

## 3. Strengths (specific, no padding)

1. **Intellectual honesty about preliminary status.** Ch. 6 (lines 20–30) explicitly marks
   unrun cells with em-dashes, refuses to fabricate the DocCL numbers, discloses the reduced
   compute, and reports the *negative* GATE-A outcome (fallback) rather than forcing a method.
   This is exactly the right scientific posture and is rarer than it should be.
2. **The catastrophic-forgetting result is real and well-presented.** Table 6.1 / Fig. 6.4: an
   86-point F1 collapse (88.0→1.8) in a single task step is a clean, robust motivation, stable
   across seeds, and a genuinely useful datum for the document-IE-CL community.
3. **The depth-wise CKA gradient is a clean signal.** Table 6.5 (1.000 → 0.78 → 0.34 → 0.245 at
   the first boundary) is a tidy, interpretable result *as a depth/head finding* (see Major M1 for
   the caveat about what it does and does not show).
4. **Engineering and reproducibility discipline.** The unified `ContinualMethod` interface
   (Alg. 4.1), Hydra grid, W&B tagging, deterministic flags, and the documented bug-fixes in
   `PILOT_STUDY_REPORT.md §7` (CORD bbox clamping, CKA sample alignment, `num_labels` caching) show
   careful, auditable work. The framework would support a correct study once the instrument is fixed.
5. **Order-dependence was deliberately measured** (Table 6.4 reverse order; §6.1.3) rather than
   swept under the rug — good practice for a single-order pilot.

---

## 4. Weaknesses

### CRITICAL (each, on its own, blocks acceptance at a top venue)

---

#### C1 — The "text-only" baseline (C1) is a **null-input** condition, not a text baseline. The headline "text is load-bearing" finding is built on it.

**Location.** `doccl/pilot/run_pilot.py:244–247` (`c1_bert → ModalityMask.TEXT_ONLY`);
`doccl/models/layoutlm_wrapper.py:204–224` (`_apply_mask`); `doccl/types.py:31`; surfaced in
Thesis **Table 3.1** (C1 = "text only"), **Table 6.4** (C1 row), **Fig. 3.2** (panel C1), and the
**Finding 3 / "Modality collapse"** paragraph in §6.1.3; corroborated by the author's own
`docs/PILOT_STUDY_REPORT.md §2.2` (which lists C1 as "Text ✓, Layout ✗, Image ✗").

**The defect.** For `ModalityMask.TEXT_ONLY`, `_apply_mask` executes **all three** masking
branches:
```python
if mask in (IMAGE_LAYOUT, TEXT_ONLY):   new_input_ids   = full_like(input_ids, pad_id)  # text → PAD
if mask in (TEXT_LAYOUT,  TEXT_ONLY):   new_pixel_values = zeros_like(pixel_values)      # image → 0
if mask == TEXT_ONLY:                   new_bbox         = zeros_like(bbox)              # layout → 0
```
So C1 receives **PAD tokens + zero image + zero layout** — i.e. *no usable input at all*. The
`forward` docstring even says so: `TEXT_ONLY → image AND layout masked (degenerate, sanity)`. The
condition the thesis labels "text only" has had its **text removed**. This is why C1 scored
`0.00 ± 0.00` AA and trained in `0/3` seeds (Table 6.4; report §3.2): you fed the model nothing.

**Why it matters.** This is a textbook construct-validity failure on the one external comparator
the design depends on, and it is *inverted in the prose*: §6.1.3 concludes "the text stream is
load-bearing for LayoutLMv3" from a condition that contains no text. The report's own Table (§2.2,
"Text ✓" for C1) contradicts the code — meaning the author is unaware that C1's text is masked.
Every artifact that labels C1 "text only" (Tables 3.1, 6.4; Fig. 3.2; Finding 3) is mislabeled, and
the load-bearing-text inference is unsupported (at best it shows the tautology "zeroing the input
collapses training").

**What would resolve it.** (a) Fix `_apply_mask` so `TEXT_ONLY` keeps `input_ids`; (b) add the
*real* external text baseline the design called for (a BERT-base wrapper — the code comment at
`run_pilot.py:226–229` admits this was deferred); (c) re-run and report whether a genuine text-only
LayoutLMv3 trains and whether "text is load-bearing" survives when text is actually present.

---

#### C2 — The instrument **cannot measure two of the four advertised components** (visual, fusion). The central "per-component multimodal decomposition" claim is unmeasurable as built.

**Location.** `doccl/models/layoutlm_wrapper.py:288–328` (`param_groups`); Thesis **Ch. 1
contribution 1** (lines 130–135: "decomposes CF … into its text, visual, layout, and fusion
contributions"), **§3.1.2** (lines 57–75: defines a "Visual stream" and "Cross-modal fusion" group
and asserts "eight groups … `visual_attn`, `fusion`"), **§4.3.2** (lines 260–263: "summed within the
eight named parameter groups (… visual attention, cross-modal fusion …)"); **Table 6.2** (which
lists **seven** groups and contains **no** `fusion` and **no** `visual_attn` row); confirmed by the
author's own `PILOT_STUDY_REPORT.md §2.3` (group list omits both).

**The defect.** `param_groups` initialises `"visual_attn": []` and `"fusion": []` but **no code
path ever appends to them.** Every attention projection is routed to `text_attn`:
```python
elif "attention" in name and ("query" in name or "key" in name or "value" in name):
    # ... "mark as text_attn for now. Refine after inspecting actual module names."
    groups["text_attn"].append(p)
```
The trailing `return {k: v ... if v}` then **drops the empty `visual_attn` and `fusion` groups
entirely.** The "refine later" comment was never actioned. Consequently the Fisher instrument
physically cannot produce a visual or fusion importance value, and **the decision tree's
`fusion-dominant → Candidate A` branch (Table 3.2, Fig. 3.4) was unreachable from the first run.**

This is not merely an implementation gap; it is intrinsic to the backbone. LayoutLMv3 is a
**single-stream** encoder: text and visual tokens share the *same* Q/K/V projections, so "text
attention" vs. "visual attention" vs. "fusion" are **not separable parameter groups** at all. The
paper's framing presupposes a separability that the architecture does not provide.

**Why it matters.** The first and primary contribution — "the first protocol that decomposes CF in
a multimodal document encoder into text, **visual**, layout, and **fusion** contributions" — is not
delivered by the instrument that produced the results. The negative conclusion ("forgetting is *not*
a fusion-specific effect," §6.1.4 / §6.6) is partly an artifact of fusion never having been in the
data. The thesis never discloses to the reader that two advertised components are absent from
Table 6.2.

**What would resolve it.** Either (a) demonstrate a defensible operationalization of
visual/fusion parameters in a single-stream encoder (e.g., separate the visual patch/relative-2D
pathway, or run the decomposition on a *two-stream* backbone where the streams are physically
distinct), **or** (b) drop the "text/visual/fusion" decomposition claim and honestly reframe the
contribution as a **depth-wise + head** forgetting study (which the CKA data *do* support).

---

#### C3 — The cross-condition significance test is **mathematically incapable of rejecting H₀**, yet "fails to reject" is presented as a substantive finding that motivates abandoning the method.

**Location.** Thesis **§6.1.3** ("fails to reject H₀, p = 0.026 / 0.028 / 0.167 … Bonferroni
α = 0.0167"); **§5.5.4** and **§3.5** (decision rule); `doccl/pilot/analyze.py:200–229`
(`condition_bwt_test`, two-sided MWU); sample sizes in `PILOT_STUDY_REPORT.md §2.2` (C4: 3 seeds +
3 alt-order = 6; C1/C2/C3: 3 seeds each).

**The defect.** The C4-vs-rival comparison is a two-sided Mann–Whitney U with n₁ = 6 (C4) and
n₂ = 3 (rival). The **minimum achievable two-sided p-value**, under *perfect* separation, is
$$p_{\min} = \frac{2}{\binom{n_1+n_2}{n_2}} = \frac{2}{\binom{9}{3}} = \frac{2}{84} = 0.0238.$$
The Bonferroni threshold is 0.0167. **0.0238 > 0.0167**, so the test *cannot* clear its own bar even
if C4 and the rival are flawlessly separated — which they essentially are for C1 (|BWT| ≈ {86–88}
vs {0,0,0}), giving the reported p = 0.026 ≈ the floor. (Under the headline "3 seeds" design
without pooling alt-orders, n₁ = n₂ = 3 and the floor is 2/C(6,3) = 0.10 — worse still.) The
"failure to reject" is a **foregone conclusion of the sample size**, not evidence about multimodal
forgetting.

**Compounding problem (construct).** Even with infinite n, this test compares the *scalar magnitude*
|BWT|, which cannot answer the question it is posed against ("do multimodal encoders forget
differently **in fusion**?"). That is a question about the *distribution/location* of forgetting
across components, not about total backward-transfer magnitude.

**Why it matters.** This underpowered, mis-targeted test is the **GATE-A pivot point**: §6.1.3 reads
the non-rejection as "the original premise that multimodal encoders forget differently in fusion is
not supported," and the project consequently abandons the proposed method for the
characterization-only fallback (§3.7, report §5). A method-abandonment decision rests on a test that
was statistically impossible to pass.

**What would resolve it.** A pre-hoc power analysis (state n needed for the floor to drop below α);
substantially more seeds; and a test that targets the *component distribution* of forgetting (e.g.,
a paired test on per-group drop *profiles*), not |BWT|.

---

#### C4 — Empirical Fisher measures **importance to the current task**, but is interpreted as **"most overwritten / where forgetting lives."** The right signal is implemented and left unused.

**Location.** Thesis **§6.1.1** (lines 57–63: "The most task-critical — and therefore most
overwritten — parameters concentrate at the output side"); **Table 6.2** caption ("Empirical Fisher
… The classifier head dominates"); **§3.3.3** ("Groups whose importance drops most are losing their
task-relevant representations"); `doccl/eval/fisher.py:26–86` (`empirical_fisher_diagonal`, computed
on the **current task's** `train_loader`, `run_pilot.py:172–174`); the correct primitive
`fisher_drop` exists at `fisher.py:121–138` but is **never called**.

**The defect.** Raw Fisher $F_\phi = \mathbb{E}[(\partial \log p(y|x)/\partial\phi)^2]$ on task *t*'s
own data is the *sensitivity of the loss to each parameter for task t* — i.e. importance. A
classifier head **always** carries the largest such gradients because it is the output layer that
cross-entropy differentiates directly; this is a generic property of the loss geometry of *any*
token classifier (BERT included), not a forgetting phenomenon. "High Fisher ⇒ most overwritten" is a
non-sequitur — in EWC, high-Fisher parameters are the ones you *protect* precisely because moving
them is costly; Fisher level says nothing about whether they *were* moved. To localize *forgetting*
you need the **change** in importance (the implemented-but-unused `fisher_drop`) or the old-task
Fisher-weighted parameter displacement $\sum_\phi F^{(t-1)}_\phi (\theta^{(t)}_\phi - \theta^{(t-1)}_\phi)^2$.

**Why it matters.** §6.1 is titled "Where Forgetting Lives," and the Fisher table is one of its two
pillars — but the table does not measure forgetting. The conclusion "forgetting concentrates in the
classifier head" is, on the Fisher evidence, indistinguishable from "the output layer has the
biggest gradients," which is true by construction and untied to continual learning.

**What would resolve it.** Report `fisher_drop` (already coded) and/or old-task-weighted
displacement, and show whether the head still dominates a *forgetting-specific* measure. Note the
CKA pillar partially supports the head story; lean on that and on a corrected Fisher-drop, not on
Fisher levels.

---

### MAJOR (would each substantially lower my score even if Critical issues were fixed)

**M1 — The valid finding is a known unimodal result re-confirmed; the novel part is the unmeasured
part.** The CKA "monotonic depth gradient, peaking at late layers + head" (Table 6.5, §6.1.2) is
precisely the Ramasesh–Dyer–Raghu (ICLR 2021, arXiv:2007.07400) result — *the thesis's own cited
prerequisite reading* — reproduced on LayoutLMv3. That is a fine sanity check, but it is not novel,
and it would hold for plain BERT. The genuinely novel claim (a *multimodal* decomposition that
behaves differently from a unimodal encoder) is exactly what C1/C2 fail to deliver. As it stands the
paper has not shown that anything multimodal-specific is happening.

**M2 — CKA methodology discards the geometry that matters and is estimated at N < D.**
`doccl/eval/cka.py:114–116` mean-pools activations over the sequence (`feat.mean(dim=1)`), collapsing
up to 512 token vectors (mostly "O") into one document vector before CKA. Token classification lives
in the *per-token* representation; a document-mean CKA can read 1.000 ("no drift") while the
token-level geometry the task uses has reorganized. Worse, the actual probe size was **N = 100**
(report §7), and the first boundary (FUNSD→CORD) is scored on CORD's 100-document eval set — so CKA
is estimated from 100 mean-pooled vectors in a 768-dim space (**N < D**), where linear CKA is
high-variance and upward-biased. The thesis (§5.5.1) and code default advertise 500; the run used
100. Redo per-token, at N ≥ 500 distinct tokens, and show stability.

**M3 — The differentiating contribution (DocCL) does not exist, and the baseline suite is <50%
complete.** Every DocCL cell is em-dash (Tables 6.5, 6.7, 6.8; §3.6 "[To be completed]"; §6.2.3,
§6.3 "in progress"). Of the advertised "ten baselines + method," only Naive, Joint, EWC, LwF and a
partial ER are run; all prompt-based (L2P, DualPrompt, CODA-Prompt), LoRA (O-LoRA), and DocCL rows
are deferred (Table 6.5). The component-targeting ablation (Table 6.7) — the experiment that would
actually test the thesis's "target the dominant component" hypothesis — is empty. At a top venue
this incompleteness alone is a desk reject; for the thesis it means the second contribution is
currently a promissory note.

**M4 — Post-hoc selection over a decision tree with unreachable branches.** The three candidates
(A/B/C) are mapped to pilot outcomes *after* the pilot (§3.4–3.6, "HARKing"-adjacent). The mitigation
(pre-specifying candidates) is undercut by C2: two of the three triggers — fusion-dominant (A) and
per-modality/visual (C) — depend on signals the instrument **cannot produce**. The "pre-registered"
decision rule could therefore only ever yield *position-drift (B)* or *fallback*, regardless of the
truth. A pre-registration whose instrument can realize only half its branches is not a real
safeguard.

**M5 — Benchmark novelty and label construction.** FCS-CL (Ch. 5) is FUNSD/CORD/SROIE under three
scenario constructions; the contribution over existing document-CL / multimodal-CL benchmarks is
thin and not argued head-to-head. The DIL unified schema is author-acknowledged as "the part of the
benchmark most exposed to judgement" (§5.2.2), with **KEY present only in FUNSD** and SROIE mapped
COMPANY→KEY / everything-else→VALUE. SROIE's BIO labels are themselves heuristically run-encoded from
an `S-<FIELD>` mirror (`doccl/data/sroie.py:157–171`). Given this, ER reaching the joint oracle on
DIL (AA 88.2, BWT ≈ 0; Tables 6.5–6.6) is plausibly a **label-degeneracy artifact** (a
VALUE-dominant schema is easy to retain), not evidence of genuine cross-domain retention. Report
per-class F1 and label frequencies before interpreting the DIL result.

**M6 — Single-backbone generalization.** All conclusions rest on LayoutLMv3-base; LiLT/BROS are
stubs (Ch. 4, planned W12). "Multimodal document encoders forget …" (Ch. 1; §6.6 threats (v)) cannot
be claimed from one model — especially when the one multimodal-specific measurement (fusion) failed.

### MINOR (fix for credibility; non-blocking)

**m1 — Paper-result diagnosis run on the forbidden debug GPU.** `PILOT_STUDY_REPORT.md` header/§7:
the *entire* pilot (the primary contribution) ran on the RTX 2060 (6 GB, bs = 1, Fisher n = 50,
CKA n = 100). The project's own `CLAUDE.md` lists "Run paper-result experiments locally on RTX 2060"
under **What Claude Code Should NEVER Do**. Ch. 6's status paragraph (lines 21–24) also conflates two
different reduced configs — it says "three epochs … batch size two" for the chapter, while the pilot
used 10 epochs / bs 1.

**m2 — `attention.output.dense` is bucketed into `ffn`; "other" is an uninterpretable grab-bag.**
In `param_groups`, the `"intermediate" in name or "output.dense" in name → ffn` branch
(`layoutlm_wrapper.py:321`) captures the attention output projection (which contains "output.dense")
along with the FFN. LayerNorms, relative-position biases, visual position embeddings, CLS token, and
pooler fall into the catch-all `"other"` — which is the **second-highest** Fisher group (Table 6.2,
1.07×10⁻²). A large share of the model's "important" parameters is thus unattributed, undermining any
"per-component" reading even for the groups that *are* populated.

**m3 — Fisher is computed without the modality mask.** `fisher.py:55` calls `model(...)` with the
default `FULL` mask, so per-condition Fisher would be mask-agnostic. Only C4 Fisher is reported, so
it does not corrupt Table 6.2 — but it means the instrument cannot do per-condition Fisher even in
principle, further narrowing what "per-component, per-condition" can mean here.

**m4 — Internal inconsistency the reader is never told about.** §3.1.2/§3.3.2 promise "eight groups"
including `visual_attn` and `fusion`; Table 6.2 shows seven, silently missing exactly those two.
Table 3.1/Fig. 3.2 show C1 as "text only" while the code feeds PAD. A careful reader who cross-checks
the methodology against the results will lose trust.

---

## 5. Questions to the Authors (answers that would move my score)

1. **C1 provenance.** Did the "text-only" condition feed real `input_ids` or PAD tokens? (The code
   says PAD.) If PAD: please relabel/retract C1, implement a genuine text-only condition **and** a
   real BERT external baseline, re-run, and report whether "text is load-bearing" (§6.1.3) survives.
2. **Visual/fusion measurability.** In a single-stream encoder where text and visual tokens share
   Q/K/V, how do you measure *visual* and *fusion* forgetting when those parameter groups are empty
   (`param_groups`)? If you cannot, will you restate Contribution 1?
3. **Power.** Given Bonferroni α = 0.0167 and your n, what is the *minimum achievable* p-value for
   the cross-condition MWU? Was the GATE-A non-rejection determined by sample size rather than data?
4. **Forgetting vs. importance.** Replace Fisher *level* with Fisher *drop* (`fisher_drop`, already
   implemented) and/or old-task-Fisher-weighted displacement. Does the classifier-head dominance
   survive a forgetting-specific measure?
5. **CKA.** Recompute per-token (not mean-pooled) at N ≥ 500 distinct tokens and report stability.
   Does the depth gradient hold, and does it differ from a unimodal text encoder?
6. **DIL degeneracy.** Report per-class F1 and label frequencies for DIL. Is ER's near-oracle result
   an artifact of a VALUE-dominant schema (KEY only in FUNSD)?

---

## 6. Overall Recommendation

**Reject.** Numeric **3 / 10**. Confidence **4 / 5**.

A diagnostic paper is only as good as its measurement instrument, and here the instrument has two
independently fatal defects — the "text-only" baseline measures *nothing* (C1), and the
text/visual/fusion decomposition **cannot measure visual or fusion** (C2) — on top of a central
significance test that is mathematically unable to reject its null (C3) and a Fisher analysis that
measures importance while claiming to measure forgetting (C4). The second contribution (the method)
is not implemented, and the third (the benchmark) is thin and partly label-degenerate. The work that
*is* valid — catastrophic forgetting, a late-layer/head drift gradient, replay winning — is real but
largely known and not specific to multimodality. I am confident in the four Critical points because
they are verifiable directly in the source, not matters of taste; my one reservation (hence
confidence 4, not 5) is that a corrected re-run *might* recover a genuine, defensible characterization
result.

---

## 7. What Would Change My Mind (the single most important thing)

**Fix the instrument, then re-run, and let the depth/head finding stand on its own honest legs.**
Concretely: (a) make C1 a *real* text-only condition and add a true BERT external baseline, so
"text-only" measures text rather than nothing; (b) either operationalize visual/fusion separability
defensibly (or move the study to a two-stream backbone where the streams physically exist) **or**
honestly reframe the contribution as a **depth-wise + classifier-head** forgetting characterization —
which your CKA data already support and which is publishable in the spirit of Ramasesh 2021 / Zhai
2023 *if* you also show it differs from a unimodal encoder; (c) replace Fisher *level* with Fisher
*drop* (already coded) and per-token CKA at adequate N; and (d) power the cross-condition comparison
(more seeds and a test that targets the *distribution* of forgetting across components, not |BWT|
magnitude). If, after that, the late-layer/head concentration holds **and** is demonstrably different
from a true text-only encoder, this becomes a credible characterization-and-benchmark paper — even
with no method, the honest fallback the thesis already names. The core phenomenon is real; the
apparatus currently wrapped around it is not yet trustworthy.

---

## Appendix — Evidence Map (claim → source)

| # | Claim in review | Primary evidence |
|---|---|---|
| C1 | C1 "text-only" = PAD text + 0 image + 0 layout | `run_pilot.py:244-247` + `layoutlm_wrapper.py:204-224` + `types.py:31`; result `0.00±0.00` Table 6.4 / report §3.2; contradicted by report §2.2 "Text ✓" |
| C2 | `visual_attn` / `fusion` groups never populated; 7 groups in output | `layoutlm_wrapper.py:295-328` (no append to those keys; `return {k:v if v}`); Table 6.2 (7 rows, no fusion/visual); report §2.3 |
| C2 | Single-stream ⇒ Q/K/V shared, not separable | `layoutlm_wrapper.py:316-320` comment "text and image attend together … mark as text_attn for now" |
| C3 | Cross-condition MWU floor 0.0238 > α 0.0167 | `analyze.py:200-229` (two-sided MWU, 6 vs 3); §6.1.3 reported p; n from report §2.2 |
| C4 | Fisher = current-task importance, not forgetting; `fisher_drop` unused | `fisher.py:26-86` (on `train_loader`), `:121-138` (unused); §6.1.1 "therefore most overwritten" |
| M2 | CKA mean-pools; N=100 < D=768 | `cka.py:114-116`; report §7 "CKA n=100"; FUNSD→CORD boundary uses CORD eval (100 docs) |
| M3 | DocCL + advanced baselines unrun | Tables 6.5/6.7/6.8 em-dashes; §3.6 "[To be completed]" |
| M5 | DIL schema arbitrary; KEY only in FUNSD; SROIE heuristic BIO | §5.2.2; Table 5.4; `sroie.py:157-171` |
| m1 | Pilot on RTX 2060 (forbidden) | `PILOT_STUDY_REPORT.md` header/§7; `CLAUDE.md` NEVER list |
