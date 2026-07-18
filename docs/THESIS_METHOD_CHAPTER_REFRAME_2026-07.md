# Thesis method-chapter reframe — design + examiner annotations (2026-07-18)

**Decision (evidence-forced):** the method chapter becomes *diagnostic-driven method
design + pre-registered falsification*, with CoLaR as the constructive control. LexSlot is
demoted from headline to one falsified family (standalone = naive-level 42.2, row
[25.1, 4.8, 96.8]; the 87.3 row is the small-buffer DocCL hybrid). Produced by a
scout→design→examiner workflow; [EXAMINER: ...] annotations are the Q1-review pass and
MUST be resolved when the LaTeX is written. Scope note: the LexSlot-as-headline framing
threads through title page + ch1/ch3/ch6/ch7 — the reframe is thesis-wide, not one chapter.

---

All numbers verified. I now have full grounds to annotate the design doc. Here is the design, returned verbatim with `[EXAMINER: ...]` annotations, followed by the required-fixes summary.

---

# METHOD CHAPTER REFRAME — Diagnostic-Driven Method Design & Falsification

*A master's thesis chapter design + drafted load-bearing prose, surgical edit list, and examiner Q&A prep, for the DocCL doc-IE continual-learning thesis.*

---

## 1. CHAPTER OUTLINE

**Framing decision this outline encodes:** Chapter 3 (`ch:methodology`) keeps its diagnostic-protocol section (§3.2) untouched and **replaces §3.4 "Lexical-Slot Head-Memory Continual Learning (LexSlot)"** with a new §3.4 **"A Pre-Registered Kill-Test Program for the Readout-Marginal Snap"**. Chapter 6 correspondingly replaces §6.1.5/"§6.2.4 The Proposed Remedy: LexSlot" with the kill-test results + CoLaR-as-control. LexSlot's *design* (slot architecture) is not deleted from the thesis — it is demoted to an appendix or a one-paragraph mention in the falsification chain (§6.2, alongside DocMERGE/LCA/HGT/CUBER) as *one more falsified buffer-free family*, since standalone LexSlot ≈ naive (AA ~42) per the corrected evidence — it is not a distinguished proposal anymore.

[EXAMINER: "AA ~42" here is imprecise where the source is exact: `ROADMAP.md:94` and `RESULTS_LEDGER_DIL.md:39` both give **42.2**, and `RESULTS_LEDGER_DIL.md` gives the full row `[25.1, 4.8, 96.8]` — FUNSD survives (25.1) but SROIE/CORD are near-dead, which is a materially different story from a flat "~42 everywhere." If this row is going into a table (edit #8/#7), use the exact figure and the row, not the rounded aggregate — an examiner who cross-checks ROADMAP.md against a rounded thesis number will read the rounding as evasive given how hard this exact area (LexSlot buffer accounting) has already been flagged as a correctness liability.]

| § | Title | Content | Evidence doc / artifact | Register note |
|---|---|---|---|---|
| **3.4.0** | Chapter framing (new, ~1 para) | States explicitly: this chapter documents a *diagnostic-driven design process that terminates in a rigorous negative result*, not a proposed method. Cross-references the introduction's methodological defense (§drafted below). | `FINDINGS_ANALYSIS_PAPER_2026-07.md` "What the paper is, and is not" | Formal, sets expectations before any numbers appear |
| **3.4.1** | The root-cause hypothesis: readout-marginal snap | Restate the diagnosis chain compactly: F1 (head-localized) → F2 (locus migrates under naive freezing) → the open question *why* the head collapses. State the mechanism in one paragraph: after each task the shared head's output marginal on old-task inputs realigns to the just-trained task's label marginal. | `RCA_FORGETTING_BASELINES_2026-07.md` root-cause statement (p.164–174) | Present as a *hypothesis to be stress-tested*, not settled fact — n=1 seed caveat stated immediately |
| **3.4.2** | Pre-registered hypotheses (H1–H4) | Table of the four competing mechanisms (modality-asymmetric drift / head interference / late-layer drift / recency-logit bias), each with its falsification criterion, registered *before* Tier B data existed. This is the section that earns "diagnostic rigor" credit with an examiner — show the commitment device. | `RCA_HYPOTHESES_2026-07.md` (dated 2026-07-16, before any Tier B JSON existed) | Emphasize the **pre-registration date precedes the data** — this is the section's rhetorical anchor |
| **3.4.3** | Adjudication: which hypothesis survives | One paragraph per H1–H4 with verdict + the one decisive number. H1 REFUTED (layernorm moves most, not any modality embed — 2.4–20.9× the largest modality embed); H2 SUPPORTED (KEY/HEADER F1 = 0.0 under every valid mask); H3 REFUTED (displacement front-loaded, early ≫ late); H4 SUPPORTED-AS-AMENDED (H4′: cos-to-just-trained-marginal 0.87–1.00 vs cos-to-own-gold 0.10–0.79 for failing methods). Disclose the H4 amendment openly (as-registered falsifier had a drafting flaw — O was both the dominant label and the falsifier target). | `RCA_FORGETTING_BASELINES_2026-07.md` §Verdicts, §H1–H4 sections | This is where "hedged, n=1" register matters most — every number carries its seed count |
| **3.4.4** | Method design space derived from the RCA | The brainstorm's judged shortlist (4 lenses → novelty audit → 2 adversarial judges → synthesis), presented as *what the diagnosis licenses you to try*: (a) eval-time marginal correction (3 variants: marginal-match oracle / prior-ratio one-step / per-doc EM), (b) frozen-trunk causality probe, (c) training-time KL anchor to prior-task marginal, (d) training-time logit adjustment (balanced softmax). State explicitly why each is a *reasonable* next move given H4′ (a marginal-level pathology suggests a marginal-level fix) and why two brainstormed alternatives were killed before implementation (OT recalibration — no mixed eval stream; head-only replay — decoupled-carrier risk). | `STATE.md` "DONE 2026-07-17 pm" block; `RCA_KILLTESTS_PREREG_2026-07.md` novelty-framing section | Show the reasoning, not just the list — this is the "method design" the chapter title promises |
| **3.4.5** | Pre-registered decision rules | The kill-test contract: smoke gates, support bars (>15 AA recovery both pooled and old-task-only), non-regression guards (CORD ≤2pt drop), the saturation-check split (KEY/HEADER probability mass ≥/< 1e-3 distinguishes "mechanism ineffective" from "information destroyed pre-correction"), survival bars for the training-time methods (KEY/HEADER F1 > 20), and the mandatory acquisition guard (must stay within 5pt of naive's diagonal — screens for EWC-style plasticity damage masquerading as retention). | `RCA_KILLTESTS_PREREG_2026-07.md` full doc (rules 1–5) | This section is procedural/contractual in tone — numbered rules, not narrative |
| **6.x.1** (results) | Kill-test outcomes | Five results, each: number, verdict against its pre-registered bar, one-line mechanism reading. See drafted prose below. | `RCA_FORGETTING_BASELINES_2026-07.md` kill-test sections (readout suite, frozen-trunk, marginal_kl, logit_adjust); `STATE.md` ACTIVE block (adversarially verified) | This is the chapter's evidentiary core |
| **6.x.2** | Consolidated verdict: what an all-null result proves | The synthesis paragraph: the snap is a *symptom*, not an invertible mechanism; no eval-time reweighting recovers it in aggregate; no training-time marginal objective prevents it at registered strength; extinction reproduces with the trunk frozen (ruling out "it's really a trunk problem"); the head-refit oracle (55.3) proves the *features* survive — so the failure is squarely readout-side and squarely un-fixable by anything short of re-exercise. | `RCA_FORGETTING_BASELINES_2026-07.md` "Consolidated kill-test verdict" | The thesis's strongest rhetorical move: closes the standard reviewer objection ("did you try cheap recalibration?") with a *registered* null, not an post-hoc excuse |

[EXAMINER: This §6.x.2 cell overstates its own source. The RCA doc's own frozen-trunk section (lines 211–220) says explicitly: *"this test does not confirm pure-head causality"* — three of four conjunctive confirmation clauses failed (AA gap 9.5pt outside ±5 band; O-collapse *milder* frozen than full naive, the opposite of confirmation; acquisition-invalid at-learning). The outline cell's "ruling out 'it's really a trunk problem'" is not what the evidence supports — the correct, and source-accurate, framing (which §2(b)'s own drafted prose gets right, see below) is "old-class extinction reproduces qualitatively with the trunk frozen; quantitative head-causality is NOT confirmed." This table cell contradicts the more careful prose drafted later in the same document (§2(b) test 2) — that is an internal inconsistency an examiner will catch on a second pass through the chapter, not just a hedge omitted once.]

| § | Title | Content | Evidence doc / artifact | Register note |
|---|---|---|---|---|
| **6.x.3** | CoLaR as the constructive control | Reframe CoLaR (currently absent from the chapters entirely) as answering the necessary question the kill-tests raise: *is re-exercise achievable cheaply, or is "replay" a euphemism for "give up and store everything"?* CoLaR: per-document SVD-compressed latent replay, DIL r128 = AA 87.6 @ 60 MB, lossless 2.7× vs raw replay (16 MB raw-doc replay in Finding 3b's table is a *different*, smaller-buffer configuration — state both numbers and don't conflate). It satisfies the necessary ingredient (whole-document (feature, position, label) consistency, Finding 3b) at a fraction of raw-replay's footprint — proving the negative results across §6.x.1 were not effort-limited or compute-limited; a method that respects the one ingredient the kill-tests show is necessary *does* work. | `FINDINGS_ANALYSIS_PAPER_2026-07.md` positioning note + Finding 3b table; `ROADMAP.md` "CoLaR is the constructive control" | First place CoLaR appears in the whole thesis — must be introduced from scratch (architecture, 1 paragraph) since Ch.3/6 currently never mention it |
| **6.x.4** | Boundary of the contribution | Explicit, undefended statement: this chapter's contribution is a diagnosis + a pre-registered falsification suite + a constructive control, not a new SOTA method. Table: what *is* claimed vs what is *not*. | `FINDINGS_ANALYSIS_PAPER_2026-07.md` "What the paper is, and is not" | Directly reused near-verbatim — that section was already written in examiner-defense register |
| **7.x** | Conclusions update | Replace LexSlot buffer-free headline with: (1) root-cause diagnosis (readout-marginal snap), (2) five pre-registered kill-tests, all null, closing the "cheap fix" reviewer hole, (3) CoLaR as the constructive control at 60MB, (4) explicit non-claim of a new method. | Same as above | Mirror of the Ch.1 rewrite (see edit list) |

[EXAMINER: §6.x.3's own logic needs sharpening before it's defensible. CoLaR is a **replay** method — it stores compressed real document activations and replays them. The kill-test program's target was *cheap, buffer-free* remedies (marginal reweighting, KL anchoring, logit adjustment) — none of them store any past data. Framing CoLaR as proof "the negative results were not effort-limited" is fine, but the outline must be explicit that CoLaR does NOT contradict or soften any of the five buffer-free nulls — it is a *different class of intervention* (replay, just compressed) succeeding where a *different class* (post-hoc/training-time correction without storage) failed. As drafted the outline risks reading as "we found a fix after all," which undercuts the whole "all-null closes the reviewer hole" rhetorical move in 6.x.2. The Q3 answer (§4 below) actually gets this distinction right — the outline table cell for 6.x.3 should say so as plainly.]

**Narrative arc as one sentence per stage** (for the introduction, drafted below): *RCA → why baselines forget (the snap) → pre-registered hypotheses → judged design space → kill-tests with pre-registered bars → all-null verdict closes the cheap-fix objection → CoLaR shows the necessary ingredient (whole-doc consistency) is achievable cheaply → the contribution is the diagnosis, the falsification, and the control, stated as such.*

---

## 2. DRAFT PROSE

### (a) Chapter introduction — the examiner-facing defense

*Proposed location: opening of the reframed §3.4 (Methodology) — the "why this counts as a contribution" paragraph an examiner reads before any numbers.*

> ### 3.4 A Pre-Registered Kill-Test Program for the Readout-Marginal Snap
>
> The diagnosis in \S\ref{sec:results-diagnosis} answers *where* forgetting lives (the classifier
> head and late encoder layers) and *what happens under naive protection* (the damage migrates
> rather than disappears, \S\ref{sec:results-locus-migration}). It does not yet answer *why* the
> head collapses — what, mechanically, does a new task do to the shared readout that five
> distinct consolidation families (weight merging, subspace transfer, parametric slot memory,
> input-anchored memory, feature-Gaussian generative replay) all fail to repair? This section
> answers that question and then asks the harder one: given the mechanism, is there a cheap
> intervention that substitutes for storing data at all? We pre-register four competing
> mechanistic hypotheses, adjudicate them against instrumented baseline runs, and — where the
> adjudication licenses a candidate fix — pre-register a falsification test for it *before*
> running it. Every one of the five candidate fixes fails its own pre-registered bar. We report
> this as the chapter's central finding, not as a failed detour on the way to one.

[EXAMINER: "five distinct consolidation families...all fail to repair" conflates two different evidence bases in one sentence. The five families (weight-merge, subspace-transfer, parametric slots, input-anchored memory, feature-Gaussian replay) are the **Finding 3 falsification chain** — a *separate* result set, established earlier in the thesis (§6.2, per the edit list), at 3-seed DIL, LayoutLMv3. The kill-test program (this section) is a *different, later, n=1-seed* investigation into *why*. Opening the section by citing the five-family result as the reason "why the head collapses" needs investigating is fine as motivation, but the sentence structure ("what does a new task do to the shared readout that five distinct families...fail to repair?") makes it sound like the kill-tests explain *those* five families' failures. They don't — they test five *different*, mostly cheaper candidates (marginal reweighting variants, frozen-trunk, KL anchor, logit adjustment). An examiner tracking method count across the chapter will find **ten** distinct negative results in total (5 Finding-3 families + 5 kill-tests) and the prose needs to keep them visibly separate or it reads as double-counting the "how many things did you try" tally to inflate rigor.]

> A negative result earns its place in a methodology chapter only if it is *load-bearing* —
> if the failure changes what a reader should believe about the problem, and if the way it was
> obtained rules out the ordinary objections to negative results (the test was too weak; the
> implementation was under-tuned; the author only tried the easy things). We address each
> directly. First, **the hypotheses were registered before the data that adjudicates them
> existed** (\S\ref{sec:rca-hypotheses}, dated 2026-07-16; the instrumented baseline runs that
> decide them did not yet exist at registration time), and the kill-tests were registered with
> numeric decision rules — support bars, non-regression guards, and a saturation check that
> distinguishes "the mechanism doesn't work" from "the information was already destroyed before
> the mechanism had a chance" — before any kill-test was executed
> (\S\ref{sec:killtest-decision-rules}). Every deviation from the as-registered rule, including
> one drafting flaw in the H4 falsifier discovered only at adjudication time, is disclosed inline
> with a dated amendment rather than silently corrected. Second, the candidates are not a
> straw-man selection: they were produced by a judged brainstorm (four independent lenses, a
> novelty audit against the CIL/domain-adaptation literature, two adversarial judges) and two
> plausible alternatives were killed *before implementation* for identifiable reasons
> (\S\ref{sec:method-design-space}) — the five that were built are the shortlist's strongest
> entries, not its first ideas. Third, each result is reported with the seed count it was run
> at; where a verdict is load-bearing for the thesis's central claim it is flagged for multi-seed
> confirmation rather than asserted at $n=1$.

[EXAMINER: "the hypotheses were registered before the data that adjudicates them existed" is true of H1–H4 (registered 2026-07-16, adjudicated against Tier B data that didn't exist yet — confirmed against source, `RCA_HYPOTHESES_2026-07.md:3-4`). But "pre-registration" is being asked to do a lot of rhetorical work here, and a sharp examiner in a *master's thesis* defense (not a registered-report journal venue) may reasonably ask: pre-registered *by whom, verified by whom*? The registration and the adjudication were both performed by the same author (with LLM-agent assistance per STATE.md's "16-agent adjudication/verification pass" and "5 independent recomputation agents") in the same multi-day sprint, not submitted to an external registry or a second party before data collection. This is a meaningfully weaker form of pre-registration than the term connotes in its home discipline (clinical trials, psychology registered reports) — worth being honest that this is *self-administered* commitment-device rigor, valuable but not equivalent to third-party pre-registration. The current draft's confident use of the bare term "pre-registered" without that caveat is the paper's own worst violation of the "hedged, n=1" register it insists on for its numbers.]

[EXAMINER: The phrase "novelty audit against the CIL/domain-adaptation literature, two adversarial judges" — check this against `RCA_KILLTESTS_PREREG_2026-07.md`'s own "Novelty framing" section, which is candid that `logit_adjust` is *not novel* (cites Menon ICLR'21 / Ren NeurIPS'20 directly) and `per_doc_em` is Saerens–Latinne (2002), with only the calibration-unit granularity as the delta. That's honest and good — but this drafted paragraph's claim that "the five that were built are the shortlist's strongest entries" risks reading, out of context, as claiming novelty-strength when the actual criterion for inclusion (per the prereg doc itself) was mechanistic relevance to H4′, not novelty. Don't let "strongest entries" imply "most novel" — the prereg doc is explicit these are known mechanisms re-tested in a new regime, and the drafted intro paragraph should say that as plainly as the source doc does, not soften it.]

> What this buys the thesis is a stronger claim than any single method result would: not "we
> tried X and X didn't work" but "we identified the mechanism, enumerated the space of cheap
> interventions a reader would reasonably propose against that mechanism, and closed each one
> with a registered test." The chapter's constructive counterpart — \textbf{CoLaR}
> (\S\ref{sec:results-colar}), a compressed latent-replay method that *does* satisfy the
> one ingredient the kill-tests show is necessary — demonstrates that this is a finding about
> the mechanism, not a ceiling on what any method can achieve: the negative results are not
> effort-limited.

[EXAMINER: "the kill-tests show is necessary" — the whole-document-consistency ingredient is NOT shown by the kill-tests. It's Finding 3b, a *separate, earlier* ablation (four-way controlled comparison of real-doc replay vs synthetic Gaussians vs real-centroid coresets vs carrier-diversity variants, per `FINDINGS_ANALYSIS_PAPER_2026-07.md` lines 149–188), run at a different time, on a different question (what must replay content contain), using a completely different experimental apparatus (frozen-trunk latent-replay ablation, not the marginal-snap kill-test suite). The kill-tests (marginal reweighting, KL anchor, logit adjustment, frozen-trunk-naive) never touch replay content at all — they test corrections to a *non-replay* head. Attributing the "necessary ingredient" finding to "the kill-tests" is a factual misattribution that a source-checking examiner will flag immediately; it must read "...the one ingredient Finding 3b's consistency ablation shows is necessary."]

---

### (b) Kill-test results section — numbers, verdicts, pre-registration discipline

*Proposed location: Chapter 6, replacing §6.1.5/§6.2.4.*

> ### 6.x The Kill-Test Program: Results
> \label{sec:results-killtests}
>
> All five tests below ran on the domain-incremental scenario (FUNSD → SROIE → CORD),
> LayoutLMv3, seed 42, and are adjudicated strictly against the decision rules pre-registered in
> \S\ref{sec:killtest-decision-rules} (`docs/RCA_KILLTESTS_PREREG_2026-07.md`). Verdicts and
> numbers below were independently recomputed from the raw result JSONs by five adversarial
> verification passes; two draft verdicts were corrected during that process and the corrections
> are folded in here rather than reported separately. All results are single-seed and provisional
> in the sense pre-registered (rule 5): none is promoted to a headline claim without multi-seed
> confirmation.

[EXAMINER: "domain-incremental scenario (FUNSD → SROIE → CORD)" — confirm this matches the actual DIL task order used throughout the RCA docs. `RCA_HYPOTHESES_2026-07.md:46` states "DIL = funsd→sroie→cord" — consistent. Good, no issue here, just flagging it was checked.]

> **Smoke gate.** The retrained naive model used for the readout suite reproduced the reference
> baseline: final pooled AA $39.78$ against the pre-registered gate $37.9 \pm 2$
> (reference: head-refit oracle $39.6$). The gate passed; the readout-suite results below are not
> contaminated by retrain drift.

[EXAMINER: Numerically correct against `RCA_FORGETTING_BASELINES_2026-07.md:186-188` and `RCA_KILLTESTS_PREREG_2026-07.md:30`. Note however the smoke gate's reference point 37.9±2 for "39.78" is a loose comparison presented as clean — 39.78 is comfortably inside the band, no issue, just worth knowing the band is ±2 wide on a headline AA metric, i.e. this "gate" would also have passed at 36 or 42. Not a defect, just don't oversell "the gate passed" as more discriminating than it is if an examiner asks how tight the smoke-test tolerance actually is.]

> **(1) Eval-time readout repair — KILLED.** Three variants of marginal-level reweighting were
> tested at the trained head's final-boundary logits: a marginal-match oracle (iteratively rescale
> the softmax until the predicted marginal equals the old task's stored gold marginal — the
> diagnostic ceiling, requires task ID), a one-step prior-ratio reweighting (single rescale by
> $q_{\text{old}}/q_{\text{last}}$, requires task ID), and a task-ID-free per-document
> Saerens–Latinne EM correction (each document's own $\sim$512 tokens serve as its calibration
> batch). The pre-registered support bar required $>15$ AA recovery on both the pooled metric and
> the old-tasks-only metric. The best variant (prior-ratio) reached pooled AA $39.70$ against the
> uncorrected $39.78$ — a **recovery of $-0.1$ AA**, i.e. no variant improved on the uncorrected
> head, and CORD non-regression held ($\le 2$pt) so the negative is not an artifact of sacrificing
> the newly-trained task to protect the old ones. The pre-registered saturation check resolves
> *why*: mean uncorrected probability mass on old-task KEY/HEADER tokens sat at $\sim 4$–$7\times
> 10^{-6}$, two to three orders of magnitude below the $10^{-3}$ branch threshold that would mark
> "mechanism ineffective on live logits." The correct reading is **"information destroyed
> pre-correction"** — by the final boundary the head has not merely mis-weighted the old classes,
> it has driven their live probability mass to a level from which no *reweighting* of the existing
> distribution can recover, since reweighting can only redistribute mass that is still present.
> One partial exception is disclosed: FUNSD's KEY class recovers from F1 $0.0$ to $16.4$ under the
> prior-ratio variant, but the same forced marginal degrades SROIE to a fully collapsed
> all-O prediction (O-accuracy $100\%$, F1 $0.0$), so the pooled number is unchanged. HEADER (both
> tasks) and SROIE-KEY recover nothing under any variant tested. This is a registered null, not an
> absence of positive evidence: the pre-registered rule specifies exactly this bifurcation, and
> the data land unambiguously on the "destroyed" side.

[EXAMINER: This paragraph title says "KILLED," but the source's own adversarial-verification precision note (`RCA_FORGETTING_BASELINES_2026-07.md:199-208`) explicitly qualifies this as: *"the tested eval-time mechanisms are dead for THIS artifact, but the rule reserves judgment on eval-time correction applied to a head whose mass survives (e.g. after a training-time keep-alive)"* — i.e. the source itself refuses the word "closed"/"family verdict" and reserves it explicitly. "KILLED" as a section header is stronger than the source's own adjudicated language ("family verdict deferred, mechanism untested" was the sub-1e-3 branch's actual wording). The body text gets this mostly right ("information destroyed pre-correction," "needs a training-time intervention" implied) but the bolded verdict word "KILLED" at the top overclaims relative to the source's explicit hedge. Recommend "NULL (mechanism untested beyond this artifact)" or similar, matching source discipline.]

> **(2) Frozen-trunk naive — NOT CONFIRMED as clean head-causality, but the extinction
> reproduces.** The pre-registered confirmation required KEY/HEADER extinction, O-collapse, and
> snap-cosine to match full naive within tight tolerances, at matched final AA ($\pm 5$). Old-class
> extinction *does* reproduce with the trunk fully frozen (final FUNSD HEADER/KEY $=0.0$, SROIE KEY
> $=0.0$ — identical qualitative pattern to trainable-trunk naive), but three of the four
> conjunctive confirmation clauses fail: final AA is $28.3$ against full naive's $37.8$ (a
> $9.5$-point gap, outside the $\pm 5$ comparability band), and — the more informative failure —
> O-collapse is *milder* under the frozen trunk ($13.3\%$ residual O-accuracy vs full naive's
> exactly $0.0\%$), the escape-hatch direction the pre-registered rule was built to detect. The
> honest reading is not "head causality confirmed" but "the trunk is not required for extinction,
> and it is also not irrelevant to its completeness" — the frozen-trunk run is **acquisition-invalid**
> (at-learning F1 $41.4/47.7/68.4$ vs full naive's $87.8/82.5/96.1$): a fully frozen pretrained
> trunk simply cannot fit these tasks well enough to license a clean effect-size comparison. This
> test therefore supports the qualitative claim (old-class extinction can occur with the trunk
> fixed) without licensing the quantitative one (equivalent severity to full naive).

[EXAMINER: This is the most careful paragraph in the whole draft — it correctly matches the source's own hedge, correctly flags the "escape-hatch direction" as informative rather than burying it, and its numbers check exactly against `RCA_FORGETTING_BASELINES_2026-07.md:211-220`. No fix needed here. This is also why the outline table's §6.x.2 cell (flagged above) is inconsistent with this very paragraph — the outline oversimplifies what this drafted prose gets right. Fix the outline cell to match this paragraph, not the reverse.]

> **(3) Training-time KL anchor to the prior-task marginal — provisional NULL.** `marginal_kl`
> adds a KL term anchoring each batch's predicted output marginal to the mixture of *prior*-task
> gold marginals (a 9-float-per-task memory object — current task deliberately excluded from the
> anchor, a red-team-caught correction to the pre-registered design made before any run). The
> pre-registered adjudicable rule is a dissociation test: recovery should concentrate on
> O/VALUE confusions (the H4$'$-fixable component) while KEY/HEADER should stay below F1 $5$
> (the H2 zero-re-exercise-extinction reading); KEY/HEADER $>20$ would refute that part of H2.
> Result: KEY/HEADER final F1 $= 0.0$ — the H2 terminal-case reading **holds**, the refutation
> never fires — but the H4$'$-direction movement is negligible (FUNSD VALUE F1 $+8.5$, O-recall
> $0\%\to3.1\%$), and final pooled AA landed at $39.1$, below the $45$–$55$ sanity window the
> pre-registration flagged (not itself a verdict criterion, but a trigger for investigation before
> adjudicating). Two explanations remain open and unregistered as a follow-up: KL-vs-CE gradient
> scale at $\lambda=1$, and batch-size-2 Monte-Carlo noise in the batch-level marginal estimate.
> The anchor's implementation was independently verified correct (inactive at task 0, excludes the
> current task). Verdict: **provisional null** — the training-time marginal signal, at the
> registered strength, does not prevent the snap.

[EXAMINER: Numbers check exactly against `RCA_FORGETTING_BASELINES_2026-07.md:222-236`. One precision nit: "a red-team-caught correction to the pre-registered design made before any run" — the source (`RCA_KILLTESTS_PREREG_2026-07.md:56`) says the current-task-inclusive version was "caught as a bug before any run," which this paragraph paraphrases faithfully. No issue. This paragraph, like (2), correctly declines to over-claim — appropriately labeled "provisional null," matching source.]

> **(4) Training-time logit adjustment (balanced softmax) — no buffer-free signal at the
> registered setting.** `logit_adjust` (Menon et al., ICLR'21 / Ren et al., NeurIPS'20 mechanism,
> tested here in a fixed-head domain-incremental regime it was not designed for) trains with a
> $\tau(\log q_{\text{task}} - \log q_{\text{cum}})$ correction inside the cross-entropy at
> $\tau=1$, no tuning, and predicts with raw logits under the cumulative prior. The mandatory
> acquisition guard passed: at-learning F1 $86.0/82.9/96.0$ sat within $5$ points of naive's actual
> diagonal ($87.8/82.5/96.1$) on every task — ruling out an EWC-style plasticity regression
> masquerading as retention. The pre-registered survival bar (KEY or HEADER final F1 $>20$) failed:
> KEY/HEADER final F1 $=0.0$, final pooled AA $36.9$. We report this as **no signal at $\tau=1$**,
> not as "the family is closed" — the pre-registration deliberately scoped the verdict to the one
> registered point on the balanced-softmax temperature curve, and a $\tau$-sweep was explicitly
> not part of the registered test.

[EXAMINER: Numbers check against `RCA_FORGETTING_BASELINES_2026-07.md:238-246`, and correctly reproduces the source's own correction note about the stale reference (88.5/84.0/97.6 → actual 87.8/82.5/96.1) — though the drafted prose here silently absorbs that correction without disclosing it happened, whereas the RCA source flags it explicitly as a `[CORRECTION 2026-07-17 eve, at adjudication]`. Given this whole chapter's rhetorical claim is "every deviation from the as-registered rule...is disclosed inline with a dated amendment rather than silently corrected" (from the drafted intro, §2(a)), silently smoothing over this one correction in the results prose is a direct self-contradiction of the chapter's own stated discipline. Either disclose the stale-reference correction inline here (one clause suffices: "against a corrected reference — the pre-registration's hard-coded diagonal was stale, verdict unaffected under either") or don't claim blanket "every deviation is disclosed" in the intro.]

> **Consolidated verdict.** Across all five tests, the readout-marginal snap behaves as a
> *symptom*, not an *invertible* mechanism. The trained head's predicted probability mass on
> under-exercised old classes has already collapsed to $\sim10^{-5}$–$10^{-6}$ by the point any
> of these interventions could act on it; no eval-time reweighting recovers it in aggregate
> ($-0.1$ AA, best variant); no training-time marginal-level objective prevents it at the strength
> tested; and old-class extinction reproduces even with the trunk fully frozen, ruling out
> "it's really a trunk-drift problem in disguise." Set against this, the head-refit oracle
> (a *fresh* linear probe on the same frozen features recovers pooled AA $55.3$, per-task
> $41.9/66.0/97.2$ vs the joint reference $88.7$) shows the underlying representation is not the
> casualty — the *readout* is. The damage lives in what the existing, already-trained head does
> with features that still carry the information; only task-consistent re-exercise
> (the replay stratum — ER, DER++, and CoLaR all clean the marginal-snap and confusion-flow
> probes at every boundary) prevents it. Table~\ref{tab:killtest-summary} collects the five
> verdicts against their pre-registered bars.

[EXAMINER: Same problem as the outline's §6.x.2 cell, repeated here: "old-class extinction reproduces even with the trunk fully frozen, ruling out 'it's really a trunk-drift problem in disguise'" directly contradicts test (2)'s own paragraph three sentences above it in this SAME drafted section, which says explicitly "NOT CONFIRMED as clean head-causality" and "the trunk is not... irrelevant to [extinction's] completeness." You cannot write "ruling out X" in the consolidated verdict when the very test that was supposed to rule out X explicitly declined to, two paragraphs earlier, in the same section, by the same author. This is the single most concrete, checkable internal-consistency defect in the whole document — it is not a matter of interpretation, it is two adjacent paragraphs in the same drafted prose disagreeing about what test (2) showed. Must fix: change "ruling out" to something like "with the milder-but-nonzero O-collapse under freezing indicating the trunk plays some completing role even though it isn't required for the core extinction pattern" — or simply drop the clause and let paragraph (2)'s own hedge stand as the record.]

[EXAMINER: "41.9/66.0/97.2" for the head-refit oracle per-task figures — checked against `RCA_HYPOTHESES_2026-07.md:29-31`: "funsd 41.9, sroie 66.0, cord 97.2; joint ref 88.7" — correct.]

>
> | Test | Pre-registered bar | Result | Verdict |
> |---|---|---|---|
> | Eval-time readout repair (3 variants) | $>15$ AA pooled recovery, $>15$ AA old-task recovery | best variant $-0.1$ AA; KEY mass $\sim7{\times}10^{-6}$ (branch threshold $10^{-3}$) | **KILLED** — information destroyed pre-correction |
> | Frozen-trunk naive | extinction/O-collapse/snap-cos match full naive $\pm5$ AA, $0.02$ cos | extinction reproduces; O-collapse *milder* (13.3\% vs 0.0\%); AA gap 9.5pt; acquisition-invalid | **NOT CONFIRMED** as clean causality (qualitative support only) |
> | `marginal_kl` (training-time) | KEY/HEADER $>20$ would refute H2 | KEY/HEADER $=0.0$; VALUE $+8.5$; AA 39.1 (below sanity window) | **Provisional NULL** |
> | `logit_adjust` (training-time) | KEY/HEADER final F1 $>20$ | KEY/HEADER $=0.0$; acquisition guard passed; AA 36.9 | **No signal at $\tau=1$** |
> | Consolidated | — | replay stratum (ER/DER++/CoLaR) clean at every probe; head-refit oracle 55.3 | **Re-exercise is the only lever tested that works** |

[EXAMINER: The table row for "Eval-time readout repair" says "$-0.1$ AA; KEY mass $\sim7\times10^{-6}$" — but the prose above it says "$\sim4$–$7\times10^{-6}$ for KEY and $<7\times10^{-7}$ for HEADER" (matching source `RCA_FORGETTING_BASELINES_2026-07.md:193-194` exactly). The table compresses this to a single "KEY mass" figure and drops HEADER's separate (and even smaller) figure — acceptable compression for a summary table, not an error, but flag that the table is lossy relative to the prose it summarizes; make sure the caption or a footnote says "see prose for HEADER's separate, smaller figure" so a reader who only reads the table isn't misled into thinking KEY and HEADER behaved identically.]

---

## 3. SURGICAL EDIT LIST

All old→new pairs below preserve validated numbers where those numbers survive re-examination (the 87.3 ± 1.0 AA / −2.3 ± 1.2 BWT figure itself is real — it is the DocCL-hybrid-with-200-exemplar-buffer result, per `clue-thesis-lexslot-row-source`), but relocate the finding to where it now belongs: the falsification chain (as one more "buffer-based, not the constructive answer" method), not the headline remedy.

| # | File : line(s) | Old wording | New wording | Rationale |
|---|---|---|---|---|
| 1 | `main.tex:4–5, 48` | *"Where Does a Document Encoder Forget? A Per-Component Diagnosis of Catastrophic Forgetting in LayoutLMv3 and a Mechanism-Targeted Remedy"* | *"Where Does a Document Encoder Forget? A Per-Component Diagnosis of Catastrophic Forgetting in LayoutLMv3, and Why Cheap Remedies Fail"* (or: *"…A Diagnostic and Falsification Study"*) | Title page still promises a novel method; the evidence base forecloses that. This is the highest-leverage single edit — it sets the reader's expectation for the entire document. |
| 2 | `chapter1.tex:103` (objectives list) | *"Derive and evaluate a mechanism-targeted remedy. Propose a CL method whose mechanism is concentrated on the dominant-forgetting component identified by the diagnosis—LexSlot, a buffer-free isolated slot memory at the head/late locus—and evaluate it against ten baselines…"* | *"Derive and falsify a design space of mechanism-targeted remedies. From the diagnosis, pre-register four competing mechanistic hypotheses for the forgetting locus and a judged shortlist of cheap interventions against the supported mechanism; adjudicate each against a pre-registered numeric bar; evaluate the resulting null result against a constructive control (CoLaR) that satisfies the one ingredient the falsification identifies as necessary."* | The objective as written commits to "propose and evaluate a method"; the actual achievement is "propose, pre-register, and falsify a design space." |
| 3 | `chapter1.tex:130–133` (contributions list, item 2) | *"LexSlot, a mechanism-targeted continual-learning remedy whose target is derived from the diagnosis…concentrating its effort on the dominant-forgetting locus…with a buffer-free, isolated slot memory gated by a drift-immune lexical signature"* | *"a pre-registered kill-test program that identifies the forgetting mechanism (a readout-marginal snap onto the just-trained task's label distribution) and closes five cheap candidate remedies derived from it, each against a numeric bar fixed before the test ran; and CoLaR, a compressed latent-replay control that satisfies the identified necessary ingredient (whole-document feature/position/label consistency) at 60 MB, showing the negative results are not effort-limited"* | Rewrites the second headline contribution from "a new method" to "a falsification suite + control," matching what was actually established. |
| 4 | `chapter1.tex:172–189` ("A mechanism-targeted continual-learning method (LexSlot)" paragraph) | Full paragraph claiming LexSlot "reaches AA $87.3\pm1.0$…approaching replay buffer-free…competitive with replay without a buffer" | Replace with a paragraph on the kill-test program (mirroring the drafted §2(b) consolidated-verdict paragraph above, condensed to ~150 words) plus one sentence noting LexSlot itself was tested and falsified as a standalone buffer-free mechanism (AA ≈ 42, near naive) — the 87.3 figure belongs to a *different*, buffer-based configuration reported later in the falsification table, not to this claim. | This paragraph is the thesis's single largest correctness liability: it asserts buffer-freedom for a number that structurally required a 200-exemplar buffer (`use_replay=True`), and it asserts LexSlot is *the* proposed remedy when the evidence base has since falsified it. Two independent errors compounding in one paragraph. |

[EXAMINER: This row's own replacement text says "AA ≈ 42" — inconsistent with the exact figure this same design doc's §1 preamble already cites as "AA ~42" and with the true source value **42.2** (`ROADMAP.md:94`, `RESULTS_LEDGER_DIL.md:39`). Minor, but for a thesis edit whose entire point is closing a precision/correctness gap around this exact number, use the exact figure throughout the edit list, not a second-order rounding. Also: `RESULTS_LEDGER_DIL.md:39` gives the row `[25.1, 4.8, 96.8]` — worth citing the row, not just AA, since "≈ naive" is doing real interpretive work (naive's DIL row is roughly [17.7, 3.7, 97.3] at the final boundary per `RCA_HYPOTHESES_2026-07.md:29`, so LexSlot-standalone is genuinely close, but showing the row substantiates "≈ naive" rather than asserting it.]

| # | File : line(s) | Old wording | New wording | Rationale |
|---|---|---|---|---|
| 5 | `chapter1.tex:204` | *"This strong-baseline context is what the diagnosis-grounded LexSlot must match buffer-free."* | *"This strong-baseline context is what the five kill-tests below are measured against: only replay-family methods (ER, DER++, and the compressed-replay control CoLaR) close this gap; every buffer-free correction tested does not."* | Same paragraph's closing sentence — must be consistent with the corrected framing above it. |
| 6 | `chapter1.tex:216` (dissertation overview) | *"…the per-component forgetting diagnostic and the mechanism-targeted CL method derived from it (LexSlot)."* | *"…the per-component forgetting diagnostic and the pre-registered kill-test program derived from it, adjudicated against a compressed-replay control (CoLaR)."* | Chapter-map sentence must point to the new §3.4, not LexSlot. |
| 7 | `chapter3.tex:484–711` (§3.4 "Lexical-Slot Head-Memory Continual Learning (LexSlot)", `sec:method-lexslot`) | Full section: slot architecture, head logit-slots, late-layer representation-slots, sharing/depth ablation design | Replace section title and content with the outline's §3.4.0–3.4.5 (framing, root-cause hypothesis, pre-registered H1–H4, adjudication, design-space brainstorm, decision rules) drafted above. **Keep** the slot architecture as a compact subsection or move to Appendix A ("A falsified buffer-free candidate: LexSlot") — it is legitimate as *evidence in the chain*, illegitimate as *the chapter's spine*. | This is the largest structural edit (≈230 lines). The architecture prose (label `sec:method-lexslot-slots`, the TikZ figure) need not be deleted — thesis committees generally accept "we designed and tested X, here is why it failed" — but it cannot remain the section's raison d'être. |
| 8 | `chapter6.tex:794–870` (§6.1.5/`sec:results-lexslot`, "The Proposed Remedy: LexSlot") | Full section incl. `\emph{isolated, buffer-free slot memory}`, `\emph{without a replay buffer}`, Table `tab:lexslot-ablation` | Replace with the drafted §6.x "The Kill-Test Program: Results" (§2(b) above) + a short subsection folding LexSlot's own result back in as one row of the falsification table (`AA ≈ 42.2 standalone`, cross-referenced to `FINDINGS_ANALYSIS_PAPER_2026-07.md`'s falsification-chain table) rather than as the section's headline. If the sharing/depth ablation numbers ($87.3\pm1.0$ isolated vs $86.3\pm0.6$ soft; head/late $86.9$ vs uniform $84.8$) are kept, they must be re-labeled: this is the DocCL-hybrid-with-buffer configuration, not standalone LexSlot — state the buffer size (200 exemplars) explicitly in the table caption. | Same correction as #4, applied to the results chapter's numbers. This closes the specific gap the scout flagged: 87.3/−2.3 was hand-patched from `_off` runs and never buffer-free. |
| 9 | `chapter6.tex:555, 563, 589, 629, 638, 644, 654, 677, 709` | Various "buffer-free"/"without a replay buffer"/"without storing any past data" references threaded through the main-comparison discussion around LexSlot | Each: either (a) delete if referring to the now-removed LexSlot headline, or (b) reword to *"a 200-exemplar buffer configuration"* if the surrounding sentence is retained as a falsification-chain data point. Grep-and-fix pass; no single template covers all nine — each needs its surrounding clause read. | Listed in the scout map as threaded through every chapter; must be swept, not just the two headline sections. |

[EXAMINER: This line list is **materially incomplete**, confirmed by direct grep against the live file. `chapter6.tex` contains "buffer-free" (or "without a replay buffer" / "without storing") occurrences at lines **555, 559, 563, 589, 590, 629, 638, 644, 654, 677, 709, 798, 803, 867, 1124, 1133-1134, 1174, 1200** — at least 19 distinct hits, not the 9 line numbers listed in row #9 (which also gets 559 vs 563 slightly off — 559 has "without a replay buffer" verbatim and is missing from the list; row #10 separately claims only 1134/1174/1200, but 1123-1124, 1131-1133 also carry the framing and aren't listed either). The design doc's own row #9 undercounts by roughly half, and its companion row #10 catches only 3 of the ~6 late-chapter occurrences. Since the entire point of this edit-list is "sweep every buffer-free reference, don't leave a stray one that contradicts the reframe," an incomplete line list defeats the edit's purpose — recommend replacing the hand-enumerated line list with an instruction to run `grep -n "buffer-free\|without a replay buffer\|without storing" chapter6.tex` as the actual execution step, rather than trusting a manually transcribed line list that is already stale relative to the file on disk.]

| # | File : line(s) | Old wording | New wording | Rationale |
|---|---|---|---|---|
| 10 | `chapter6.tex:1134, 1174, 1200` | Late-chapter references to LexSlot buffer-free framing (compute/ablation/extended-scenario sections) | Same treatment as #9 — reword or excise depending on whether the surrounding paragraph survives the reframe. | — |
| 11 | `chapter7.tex:103–130` | *"Its second contribution is LexSlot, a head/late-targeted method…all without storing any past data…LexSlot approaches replay buffer-free"*; Future Work section repeats *"the head/late-targeted LexSlot approaches replay buffer-free"* | Replace with the conclusions mirror of the Ch.1 rewrite: root-cause diagnosis → five pre-registered kill-tests (all null) → CoLaR as constructive control at 60MB → explicit statement that no new buffer-free method is being claimed. Future Work: replace "cross-lingual LexSlot confirmation" with "cross-lingual/multi-backbone confirmation of the kill-test nulls and the CoLaR control (the B+ grid)." | Same correctness liability as Ch.1, restated at the point examiners read last and weight most for the thesis's final claim. |
| 12 | New content needed, not an edit: introduce CoLaR | Currently **zero mentions** of CoLaR, PLaR, LexMem, or the B+ scope decision anywhere in chapters 1–7 | Add: (a) one architecture paragraph in the new §3.4 or a new §3.5 ("CoLaR: a constructive control") — per-document SVD compression of layer-$k$ activations, replayed into the plastic head+late layers, DIL r128 = AA $87.6$ @ $60$ MB, lossless $2.7\times$ vs raw-document replay; (b) the Finding-3b consistency ablation (4 whole real documents AA $63.8$ vs 4 decoupled real-centroid carriers AA $36.7$ — same count, same boundary, same real features, +27 AA from consistency alone) as the mechanism explaining *why* CoLaR works where every marginal-summary memory failed. | Without this, the chapter has a diagnosis and a falsification chain but no demonstration that the necessary ingredient is achievable cheaply — exactly the gap an examiner will probe (Q&A item 3 below). |

[EXAMINER: Confirmed by grep — `chapter1.tex` through `chapter7.tex` contain zero occurrences of "CoLaR" or "colar." Row #12's claim is accurate. But note the 60 MB / 2.7× lossless figures need one more piece of context if this becomes a from-scratch introduction for a reader who has seen nothing about CoLaR before: the comparison baseline for "lossless 2.7×" is raw-document replay at ~163 MB for the same content (per `STATE.md:556`: "raw-d50 87.3@163MB · CoLaR-r128 87.6@60MB"), not the 16 MB figure the outline's own §6.x.3 row correctly warns against conflating with. Make sure the from-scratch paragraph states the 163 MB comparator explicitly — "2.7×" is meaningless to a first-time reader without the number it's relative to.]

| # | File : line(s) | Old wording | New wording | Rationale |
|---|---|---|---|---|
| 13 | `chapter3.tex:721` and any residual `sec:method-lexslot` internal cross-references (`\S\ref{sec:method-lexslot}`, `\S\ref{sec:method-component-hypotheses}`) | Cross-references throughout Ch.6 pointing at the old LexSlot method section | Repoint to the new §3.4 kill-test section labels (`sec:rca-hypotheses`, `sec:method-design-space`, `sec:killtest-decision-rules`), or to the appendix if the architecture is relocated there. | Mechanical but necessary — LaTeX `\S\ref` will silently point at the wrong (renamed) section otherwise; a `grep -rn "method-lexslot"` sweep across all chapter files is the concrete next action. |
| 14 | `thesis/generated/table_lexslot_ablation.tex` (referenced, does not exist) | N/A — chapter6.tex:841 references a file that was never generated; falls back to the hand-hardcoded table | If the sharing/depth ablation table is retained (edit #8), either (a) generate the real file from `analyze_results.py` with the `_off`/`_head_only`/`_uniform` variant folding fixed, or (b) keep the hardcoded fallback but add an inline LaTeX comment documenting that it is hand-patched from the buffer-based `_off` runs (per memory `clue-thesis-lexslot-row-source`) so a future editor does not assume it is parser-generated. | Prevents the exact class of silent staleness that produced the buffer-free mislabeling in the first place. |

**Suggested execution order:** #7 and #8 first (they are the structural rewrites everything else depends on for section labels), then #12 (new CoLaR content, needed before #2/#3/#5/#11 can cross-reference it), then the mechanical sweeps #1/#2/#3/#4/#5/#6/#9/#10/#11/#13, then #14 last (table regeneration, lowest risk/most optional).

---

## 4. EXAMINER Q&A PREP

**Q1. "So your method chapter has no method — isn't that a contradiction in terms for a methodology chapter?"**

> A methodology chapter documents the *process* by which the thesis reached its findings, not necessarily a proposed artifact. This chapter documents a complete method-design process: a root-cause diagnosis, four pre-registered competing hypotheses, an adjudication, a judged brainstorm over the design space the surviving hypothesis licenses, and five pre-registered falsification tests. That is methodology in the fullest sense — it is *more* rigorous than a chapter that proposes one method and reports it working, because every step commits to a falsifiable prediction before seeing the data that tests it (§3.4.2's H1–H4 were registered 2026-07-16, before Tier B existed; the kill-test bars were registered before any kill-test ran). What the chapter does not contain is a *positive* method result — and we say so explicitly (§6.x.4) rather than dressing a null result as a qualified success. The distinction the question is really asking about is "novel proposed artifact" vs "rigorous negative result with mechanism," and the thesis is honestly the second.

[EXAMINER: This answer is fine as far as it goes but dodges the harder, more likely follow-up a Q1-calibre examiner will actually ask: *"a master's thesis is a demonstration of competence to conduct independent research and typically expects at least one constructive contribution — where is yours?"* The honest answer is CoLaR, but CoLaR is explicitly scoped in the source docs as *not* a method contribution either ("Two later results slot into the argument, not as method contributions," `FINDINGS_ANALYSIS_PAPER_2026-07.md:5`). So the thesis's actual answer to "where is the constructive contribution" is: the *diagnostic instrument* (Finding 1's modality-ablation protocol) and the *benchmark* (FCS-CL) — both of which are real, positive, reusable artifacts. The Q&A prep should route the "no method?" objection to those two things explicitly, not leave the answer resting entirely on "rigor of the negative result," which is a much harder sell to a committee expecting to see something built. This is a gap in the prep, not in the design.]

**Q2. "Five failed methods is a lot to report as a contribution — how is this different from just saying 'we tried things and they didn't work'?"**

> Three things distinguish it from an unstructured list of failures. First, every candidate was chosen *because* the adjudicated mechanism (H4′, the marginal snap) predicts it should plausibly work — a marginal-level pathology licenses marginal-level fixes (eval-time reweighting, training-time marginal anchoring, training-time logit adjustment), so failure of all of them is evidence *about the mechanism's invertibility*, not just about five unrelated ideas. Second, each has a pre-registered numeric bar and a documented verdict against it, including a saturation check (§2(b), test 1) that distinguishes two qualitatively different failure modes — "the mechanism doesn't work" vs "the information needed for it to work was already gone" — and the data land decisively on the second, which is itself a specific, falsifiable claim about *when* in training the damage becomes irreversible. Third, the failures triangulate: the frozen-trunk test rules out "it's really a trunk problem," and the head-refit oracle (55.3) rules out "the features are gone too" — so what remains is a precisely bounded claim (the readout, specifically, and specifically once trained, is what's broken) rather than a diffuse "nothing worked."

[EXAMINER: Repeats the same overclaim flagged in §2(b)'s consolidated verdict: "the frozen-trunk test rules out 'it's really a trunk problem'" is not what that test showed (see the CONFIRMED annotation above — 3 of 4 confirmation clauses failed, O-collapse was *milder* frozen, which is evidence *for* some trunk contribution, not against it). This Q&A answer needs the same fix: soften to "the frozen-trunk test shows the trunk is not *required* for the core extinction pattern, though it isn't fully ruled out as a contributing factor either — see the test's own hedge." Since this error appears in three separate places in the document (outline table §6.x.2, the drafted consolidated-verdict prose, and this Q&A answer), it reads as a load-bearing rhetorical move the author reached for repeatedly rather than a one-off slip — worth double-checking there isn't a fourth instance elsewhere before submission.]

**Q3. "If nothing cheap works, doesn't that mean the whole research direction was a dead end? What did you actually contribute?"**

> No — it means the *cheap* direction was a dead end, and the chapter proves that precisely rather than leaving it ambiguous. CoLaR (§6.x.3) is the chapter's answer to "then what does work, and how expensive is it really": it satisfies the one ingredient the kill-tests and the companion consistency ablation (Finding 3b: 4 whole real documents AA 63.8 vs 4 decoupled-carrier documents AA 36.7, same count, same features, only consistency differing) show is necessary — whole-document (feature, position, label) consistency — and it does so at 60 MB with a lossless 2.7× compression ratio versus storing raw documents, reaching AA 87.6 on the domain-incremental scenario, within 1.1 points of the joint oracle (88.7). That is the constructive half of the argument: the negative results were not effort-limited or compute-limited. A method that respects the identified necessary ingredient works; every method that tried to approximate or summarize it did not. The contribution is locating that boundary precisely, which is a more useful result for the field than one more point method, because it tells the next researcher *where not to look* with evidence, not intuition.

[EXAMINER: "the kill-tests and the companion consistency ablation...show is necessary" repeats the same misattribution flagged in §2(a) — the kill-tests never touch replay content or consistency; only Finding 3b does. Fix to "Finding 3b's consistency ablation." Separately: "88.7 (joint oracle)" and "87.6 CoLaR" giving "within 1.1 points" is arithmetically correct (88.7 − 87.6 = 1.1) — checked, fine. This answer is otherwise the sharpest and most honest of the four Q&A items — it correctly keeps CoLaR's replay-based success conceptually separate from the buffer-free kill-tests' failure, which is exactly the distinction flagged as missing from the outline's §6.x.3 cell. Recommend using this Q&A answer's phrasing to fix that outline cell, once the misattribution is corrected.]

**Q4. "Every result here is n=1 seed. How much of this survives multi-seed variance — couldn't 'KEY F1 = 0.0' just be noise?"**

> Two different confidence levels apply and the chapter keeps them separate. The kill-test *numbers* (§6.x, all five tests) are n=1 and explicitly flagged provisional per the pre-registration's own rule 5 ("all n=1 seed — verdicts provisional; only multi-seed if a verdict becomes load-bearing"). But the *qualitative* pattern they test — KEY/HEADER extinction under naive/LwF — is not a single-seed claim: it is corroborated by the 3-seed Tier A per-class forgetting ledger (`RCA_HYPOTHESES_2026-07.md` "Verified anchors," naive KEY=0.0/HEADER=0.0 across the class-asymmetric-extinction finding) and by the pre-existing Finding 1/Finding 2 evidence (3-seed permutation tests on head-localization). What is genuinely single-seed is the *kill-test adjudication itself* — whether marginal_kl's AA 39.1 or logit_adjust's AA 36.9 would move materially at seeds 7/123. We do not claim they wouldn't; we flag them as the concrete next step (ROADMAP's B+ grid), and the chapter's central claim (no cheap correction substitutes for re-exercise) is deliberately built to be robust even if individual AA numbers shift a few points, because it rests on the *saturation check* (probability mass ~$10^{-6}$ vs the $10^{-3}$ branch threshold — three orders of magnitude of headroom) rather than on a numerically close call.

[EXAMINER: This answer's citation of "the 3-seed Tier A per-class forgetting ledger" needs a precision check: `RCA_HYPOTHESES_2026-07.md:17-23` ("Verified anchors") does state naive KEY=0.0, HEADER=0.0, and cites `results/rca/a1_summary.md` — but that same section's header explicitly says these anchors were "copy-checked on disk 2026-07-16," and the doc doesn't itself state Tier A's a1/a2 seed count on the line cited; the thesis's Finding 1 (which *is* independently confirmed 3-seed, per `FINDINGS_ANALYSIS_PAPER_2026-07.md:53-56`, "n = 3 seeds supports shared location") is about *head-localization*, not specifically about KEY/HEADER=0.0 class-level extinction. The Q&A answer conflates "Finding 1 is 3-seed" with "the KEY=0.0/HEADER=0.0 anchor is 3-seed" — these may both be true, but the design doc doesn't actually verify the Tier A a1/a2 artifacts' seed count from the text it cites; it asserts it. Before this answer ships, confirm `results/rca/a1_summary.md`'s actual seed count directly (the "Tier A" label alone, without checking the artifact, is not sufficient grounds for "3-seed" — RCA_HYPOTHESES calls Tier A "3 seeds" in its adjudication-protocol caveats section for the *hypotheses* overall, but that's a different claim than each specific number's seed count).]

**Q5. "Why does the thesis title/introduction still say 'a Mechanism-Targeted Remedy' if the actual finding is that mechanism-targeted remedies don't work?"**

> It shouldn't, and that is a documented correctness gap this reframe fixes, not a defensible design choice. [Reference the surgical edit list, item #1.] The honest answer to why it currently reads that way is chronological: the thesis text is frozen at an earlier stage of the research program (pre-2026-07-02), when LexSlot was the leading candidate and had not yet been re-evaluated under the corrected buffer-accounting or superseded by the RCA's falsification chain. The reframe corrects the title, the three chapters' headline claims, and the ~20 threaded "buffer-free" references to match the evidence base as it now stands: LexSlot is retained in the document as one falsified candidate among five, not as the headline contribution, and the title changes from promising "a remedy" to accurately describing what was delivered — a diagnosis of why remedies at this level of the design space fail, and a demonstration (CoLaR) of what does not fail.

[EXAMINER: Good, honest answer, and the "~20 threaded references" figure now matches the corrected count from the grep above rather than the design doc's original undercount of ~9+3. One factual nit: "frozen at an earlier stage...pre-2026-07-02" — check this date against when the 87.3-with-buffer LexSlot number was actually computed and hand-patched into the thesis. `clue-thesis-lexslot-row-source` memory (cited throughout) says the row was "hand-patched bc analyze_results parser cant fold `_off` variant" — this implies the thesis text was updated *after* the number existed, not merely frozen from an earlier unaware state. If the 87.3 number was hand-patched into the thesis text *knowing* it was the buffer-based number (rather than through an honest but since-corrected belief that it was standalone), "chronological drift" is a materially different, more favorable explanation than what actually happened, and an examiner who pulls the git history on that hand-patch commit could catch the discrepancy. Verify the commit history/STATE.md timeline before asserting "frozen at an earlier stage" as the explanation — if the correction chronology doesn't support it, use a more neutral phrasing ("a correction made after buffer-accounting was clarified, not yet propagated to the affected sections") rather than implying the error long predates knowledge of it.]

---

**Evidence docs used (verified, not re-derived):**
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/docs/RCA_FORGETTING_BASELINES_2026-07.md`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/docs/RCA_KILLTESTS_PREREG_2026-07.md`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/docs/RCA_HYPOTHESES_2026-07.md`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/STATE.md`, `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/ROADMAP.md`

**Thesis files read for exact current wording (all edits above are grounded in the actual text, not paraphrase):**
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/thesis/main.tex`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/thesis/chapters/chapter1.tex`
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/thesis/chapters/chapter3.tex` (§3.4, lines 484–711)
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/thesis/chapters/chapter6.tex` (§6.1.5/§6.2.4, lines 794–870, plus scattered lines 555–1200)
- `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/thesis/chapters/chapter7.tex` (lines 90–135)

No code was written or run for this task — it is a drafting/design deliverable only, per the request.

---

## REQUIRED FIXES SUMMARY

The design does **not** survive cleanly. Ranked by severity:

**CRITICAL (internal self-contradiction, will fail on a second read-through):**
1. §2(b)'s "Consolidated verdict" claims the frozen-trunk test "rul[es] out 'it's really a trunk-drift problem in disguise'" — this directly contradicts test (2)'s own paragraph three sentences earlier in the *same section*, which says "NOT CONFIRMED as clean head-causality" and reports O-collapse was *milder*, not equal, under freezing. Same overclaim recurs in the chapter outline (§6.x.2 cell) and in Q&A item 2. Fix all three occurrences to match test (2)'s own correctly-hedged language; do not let the summary silently upgrade a hedged qualitative result to a ruled-out quantitative one.
2. Both the chapter intro (§2(a)) and Q&A item 3 attribute the "whole-document consistency is the necessary ingredient" finding to "the kill-tests" — it is Finding 3b (a separate, earlier ablation using a different apparatus). The kill-tests never touch replay content. This is a factual misattribution, not a framing choice; fix both locations to cite Finding 3b explicitly.

**HIGH (overclaim relative to source, or self-contradicts the chapter's own stated discipline):**
3. Test (4)'s drafted prose silently absorbs the "stale reference" correction without disclosing it happened — contradicting the chapter intro's explicit promise that "every deviation...is disclosed inline with a dated amendment." Either disclose it in the results prose or drop the blanket claim in the intro.
4. Test (1)'s section header says "KILLED" where the source's own adversarially-verified precision note explicitly declines that word ("family verdict deferred, mechanism untested" for the sub-1e-3 branch). Soften the header to match the source's own hedge.
5. The "pre-registered" framing throughout (intro, Q1) never discloses that registration and adjudication were both self-administered by the same author/agent team in one sprint, not externally registered — a materially weaker form of the term than a Q1 examiner will assume it means. Add one clarifying sentence.

**MEDIUM (precision/completeness gaps):**
6. Surgical edit-list row #9/#10's line-number enumeration for "buffer-free" references in chapter6.tex is confirmed incomplete against a direct grep (lists ~9 lines; actual count is ~19+, spanning 555 through 1200). Replace the hand-transcribed line list with a grep-driven execution step.
7. LexSlot-standalone is cited inconsistently as "~42," "≈ 42," and "AA ≈ 42.2" across the document; the exact sourced figure is **42.2** with row `[25.1, 4.8, 96.8]` (`ROADMAP.md:94`, `RESULTS_LEDGER_DIL.md:39`) — use the exact number and row throughout, especially given this exact figure is the document's own subject of a precision correction.
8. Q4's claim that the KEY=0.0/HEADER=0.0 anchor is "3-seed" conflates Finding 1's independently-confirmed 3-seed head-localization with the specific Tier A a1/a2 class-extinction numbers, whose seed count is asserted, not verified from the cited text. Confirm `results/rca/a1_summary.md`'s actual seed count before shipping this claim.
9. Q5's "frozen at an earlier stage" explanation for the stale title/LexSlot claims should be checked against the actual commit timeline for when the 87.3-with-buffer number was hand-patched in — if the patch postdates awareness that it was buffer-based, "chronological drift" is a more favorable (and less accurate) explanation than what happened.
10. The outline's §6.x.3 cell (CoLaR as control) doesn't clearly separate "buffer-free kill-tests failed" from "CoLaR, a replay method, succeeded" — risks reading as "we found a fix after all," undercutting the all-null rhetorical close. Q3's Q&A answer gets this distinction right; use its phrasing to fix the outline cell.
11. Q1's answer doesn't address the more likely follow-up ("where's your constructive contribution, then") — route to the diagnostic instrument and FCS-CL benchmark explicitly rather than resting entirely on rigor-of-negative-result.

**LOW:**
12. The chapter intro's opening sentence risks double-counting "how many methods failed" by juxtaposing the 5-family Finding-3 chain and the 5-test kill-test program in one breath without a clear seam — keep the two negative-result sets visibly separate.
13. CoLaR's "60 MB, lossless 2.7×" figures need the 163 MB raw-replay comparator stated explicitly on first introduction (currently only in STATE.md, not surfaced in the drafted thesis paragraph) — "2.7×" is meaningless without it.