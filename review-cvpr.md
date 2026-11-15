# Review of "Memory Is Not Enough: Separating Drift from Background Conflict in Continual Document Understanding"

Reviewed on 2026-11-15 from `paper_cvpr/main.tex`, `supp_body.tex`, every generated table (including the ones the paper does not include yet) and the built PDF (21 pages, main text ends on page 11).

Standard applied: a demanding CVPR / NeurIPS / ICML reviewer. Scores are my estimates of how such reviewers would score the paper as it stands, not a prediction of the outcome.

Severity labels used below:
- **Critical**: by itself enough for a reject from a careful reviewer.
- **Major**: lowers the score by about a point, or will dominate the rebuttal.
- **Minor**: costs credibility or clarity; cheap to fix.

---

## 1. Verdict

| | Score |
|---|---|
| Overall (1 to 10) | **4, borderline reject** (three reviewers: 4, 3, 5) |
| Weighted score (skill formula) | 5.1 / 10 |
| Soundness / Presentation / Contribution (1 to 4) | 2 / 2 / 2 |
| Confidence | 4 / 5 |
| Decision today | Reject |

Why: the central idea is good and the oracle controls are better than what most continual-learning papers offer, but the evidence that would convince a reviewer is not in the paper yet. The comparison with baselines mixes two training protocols and never uses the headline metric for a baseline. Several baseline ports are visibly broken. The mechanism is shown on one partition of one dataset, and the paper's own second stream and second backbone partly contradict it. The paper claims five datasets and delivers method results on four, with a win on two. It is also three pages over the limit.

None of this is fatal to the idea. Section 9 lists what moves the score, in order.

---

## 2. What the paper claims

1. Replay closes the gap to joint training when only the domain changes, but not when the label set grows.
2. The reason is the background label "O": a shared softmax treats an "O" as evidence against every class, while it is evidence only against the classes annotated in that session. A replay memory repairs drift but replays these stale negatives.
3. CoLaR-SBR (a compressed latent memory of whole documents, plus one background logit and one softmax per session) removes most of the gap on CIL-CORD: 89.1 full-label F1 against 30.7 for the same memory with a shared softmax and 92.0 for joint training.
4. On one annotation schema, per-session low-rank deltas (SBR-Recall) replace the memory: 92.6 with no stored document.
5. A benchmark against eight baselines on six streams over five datasets.

---

## 3. Strengths

1. **A testable causal story.** Each hypothesis has a stated prediction and an intervention. The relabel and mask oracles (92.1 and 89.7 full-label F1 with the shared softmax unchanged) are the strongest evidence in the paper and are rarely done in this literature.
2. **A crisp, useful insight.** "An 'O' is evidence only against the classes its annotator could see" is easy to state, easy to check, and Fig. 1 conveys it in one picture.
3. **Large effect.** 30.7 to 89.1 on CIL-CORD with the memory held fixed; 52.2 to 87.2 on CIL-FUNSD; 88.8 ± 0.6 on BROS (this last result exists in `supp_bros.tex` but is not in the paper).
4. **The readout, not the memory format, carries the gain.** Raw documents with the same readout reach 88.4. This is an honest ablation that many papers would hide.
5. **Honest reporting.** Negative results are stated (DIL, DIL-XFUND, rank 32, CIL-FUNSD controls), per-seed values exist, every number is generated from result files, and tuning leakage is discussed.
6. **Session-by-session forgetting figure** and the error-destination analysis go beyond a final-accuracy table.

---

## 4. Critical issues

### C1. The comparison with baselines is not on equal terms
- Every baseline cell on CIL-CORD, Mixed and DIL in Table 3 is marked legacy: early stopping on the *test* split, epoch cap 100 or 30 against 10, partly mixed precision, mostly two seeds. Every cell of the proposed method is clean protocol, three seeds.
- The headline metric, full-label F1, is "--" for all eight baselines. The paper's main number therefore has no baseline to stand against.
- Budgets differ: ER and DER++ keep 200 documents, the proposed memory 250; ER's configuration replays 8 stored documents per step, the proposed method 4; ER trains every layer.
- What to do: rerun at least ER, DER++, LwF, CPFD, IS3 and CL-LoRA on CIL-CORD under the clean protocol with 250 documents and three seeds, so that Table 3 has full-label F1 in every row. If time allows only two, run ER and DER++.

### C2. Several baseline ports are visibly broken, and none is validated
- The paper says so itself: "All baselines are our ports, none checked against published numbers; four end below naive training on every seed".
- CPFD scores 2.7 and 0.0 AA where naive fine-tuning scores 18.3. In its own benchmarks CPFD is far above fine-tuning. A reviewer will read 0.0 as a bug, not a finding.
- ICE-O, the closest prior method, "early-stopped at 0.0 F1 on session 3 on every seed" and was "still improving at the epoch cap". The comparison with ICE-O is therefore not usable as evidence that a learned background or a trainable encoder helps.
- LwF lost a seed to a mixed-precision collapse; CL-LoRA on DIL predates a bug fix.
- What to do: validate at least two ports where published numbers exist. The cheapest route is the other direction: the readout uses nothing specific to documents, so run "raw memory + SBR" with BERT on a standard continual-NER benchmark (CoNLL-2003, OntoNotes 5, I2B2) where CPFD, CFNER and IS3 report numbers. One table there validates the ports and widens the claim at once. For ICE-O, lift the epoch cap until every session converges.

### C3. The mechanism rests on one partition of one dataset, and the paper's other evidence partly contradicts it
- Every rung of the ladder runs on one class order of CORD, with 100 test receipts and three seeds.
- CIL-FUNSD reverses two conclusions: true stored labels do *not* rescue the shared softmax there (52.2 against 52.2), and the current side *is* needed. A MiB-style loss inside a shared softmax matches CoLaR-SBR there (87.3 against 87.2). The paper states "why the streams differ is untested".
- On BROS the oracle that "supports" H1 is unstable: 89.3, 60.2, 76.7 (`supp_bros.tex`, not in the paper).
- The abstract and conclusion state the finding as general ("a memory only repairs the drift … the remaining forgetting mainly comes from the background label").
- What to do: (a) complete CIL-WildReceipt, the third class-incremental stream; (b) run two more class orders of CIL-CORD for the four key rows (CoLaR, oracle before each session, CoLaR-SBR, SBR-Recall); (c) test the obvious explanations of the CORD / FUNSD difference: CIL-CORD cut to two sessions, and CIL-FUNSD with fifty stored forms for every row; (d) scope the abstract to what holds on all streams: the per-session readout closes most of the gap on every class-incremental stream; which side of the conflict dominates depends on the stream.

### C4. Claims in the abstract and introduction exceed the tables
- "5 public document datasets … significant improvement in the class-incremental setting": CIL-WildReceipt has no method result; on DIL-XFUND both proposed methods are 3 points below ER; on DIL they are below DER++ and ER. The win exists on CORD, FUNSD and Mixed.
- "eight CL baselines on six streams": baselines exist on three streams, three of them on a fourth, none on CIL-FUNSD and CIL-WildReceipt.
- "upper-bound baseline Joint" in the abstract against "Joint training is a reference, not a bound" in the limitations. SBR-Recall and CoLaR-SBR on Mixed are above it.
- "significant" with no test.
- What to do: either land the WildReceipt and XFUND runs, or write "four datasets" and state the DIL result as "no gain where the label space is fixed, as the analysis predicts".

### C5. The stream construction re-presents the same documents in every session
- Appendix A: a session's pool keeps "every receipt that contains at least one of its entities". The pools are 800, 382, 549, 723 and 780 of CORD's 800 training receipts. So almost every receipt returns in almost every session with a different label mask.
- Consequences a reviewer will raise: (1) the motivation "old documents cannot be kept" does not match a stream in which they come back; (2) the memory of 250 documents is 31% of the training set, far above usual budgets; (3) the strength of the "current documents call old entities 'O'" effect is maximal by construction, since the current document *is* the old document.
- The main text never says this.
- What to do: state it in Section 3, and add a variant with disjoint documents per session (each receipt assigned to one session). If the effect and the gain survive, the paper is much stronger; if they shrink, the reviewer would have found it anyway.

### C6. The joint-training reference is weak
- Joint training reaches 92.0 full-label F1 on CORD. LayoutLMv3-base is reported at about 96.5 on CORD. Batch size 2, patience 2 and a 10% hold-out are the likely causes; joint runs finish in 0.26 to 0.42 hours.
- On CIL-FUNSD one joint seed stopped at its first epoch (77.8), so "CoLaR-SBR above joint" there is an artefact.
- "Within 2.1 points of joint" and "above joint" are then statements about an under-trained reference.
- What to do: train the joint reference to convergence (fixed 30 to 50 epochs or patience 10) and report the gap to it. Expect the gap of CoLaR-SBR to grow by a few points; it is still a large gain over 30.7.

### C7. Page limit
- The main text ends on page 11; the limit is 8. Related Work alone is about 1,150 words, the investigation 1,300.
- Section 9 gives a cut plan.

---

## 5. Major issues

### M1. Novelty is thin as currently framed
- SBR is ICE-O's per-session softmax with a learned background, plus replay. The paper's own ablation cannot resolve the learned background from the constant one on LayoutLMv3 (89.1 ± 2.0 against 82.4 ± 6.9; the constant one wins on seed 123).
- CoLaR is latent replay with a per-document truncated SVD. It is not compared with any other compression (REMIND's product quantisation, 8-bit pixels, fp16 raw).
- SBR-Recall is per-task low-rank adapters routed by an out-of-distribution score, close to MORE and to task-identity inference methods.
- What to do: make the diagnosis the first contribution and say plainly that the readout is an instantiation of prior ideas. Then show the delta against ICE-O properly: the BROS runs already favour the learned background (88.8 ± 0.6 against 78.9 ± 8.8), so put them in the paper, and add two more seeds on LayoutLMv3.

### M2. The alternative explanations of H3 are not ruled out
- "The conflict lives in the shared normalisation" is tested only with SBR, which changes three things at once: the normalisation, the number of background parameters, and the decision rule (a claim threshold of 0.5 instead of an arg-max).
- Missing controls, all with the same memory: per-class sigmoid outputs (binary cross-entropy); a separated softmax in the SS-IL style; ER-ACE; a shared softmax with bias correction (BiC, WA).
- The limitations admit sigmoid heads were not run. A reviewer will not accept that for the control that matters most.

### M3. The suppression finding (H0) may be an artefact of the replay ratio and the memory size
- Four replayed documents per two current ones means two thirds of every batch label the new classes "O". The paper says the ratio "was not varied".
- No buffer-size ablation exists (10, 20, 50 documents per session), although 50 per session is a third of the data.
- What to do: one ablation table on CIL-CORD, CoLaR and CoLaR-SBR at replay ratios 1:1 and 4:2 and at 10, 20, 50 documents per session.

### M4. Which method is "the method" is unclear
- The named method is CoLaR-SBR. The best row on CIL-CORD is SBR-Recall, which uses no memory, under a title that says memory is not enough.
- SBR-Recall was run only on CIL-CORD (three seeds) and Mixed (one seed, legacy). Its rank was chosen on CIL-CORD with early stopping on the test split. It costs one forward pass per session and 28 MB per session in fp32.
- What to do: present SBR-Recall as a variant with stated costs, run it on CIL-FUNSD and CIL-WildReceipt, and report inference time.

### M5. The memory-efficiency claim is not supported in its current form
- At rank 128 CoLaR takes 2.2 times *more* bytes than raw documents with 8-bit pixels; the paper says so. Rank 64 is unstable (86.6, 60.6, 91.1) and rank 32 collapses to 27.4, below plain CoLaR.
- Table 4 lists SBR-Recall at 56.6 MB, a size "no run evaluated" (fp16); as trained it is 113.2 MB, more than CoLaR-SBR.
- Table 4 compares stored *data* with a transient *teacher copy* (LwF and others, 501.5 MB). A reviewer will call this apples and oranges: LwF stores no data.
- Why rank 32 falls below a shared softmax is not analysed.
- What to do: (a) evaluate SBR-Recall with fp16 deltas from the saved checkpoints (no training needed), which turns the 56.6 MB into a measured row; (b) add equal-byte comparisons (ER with 8-bit pixels at 100 MB, about 560 documents); (c) split Table 4 into "data kept" and "parameters kept"; (d) claim what holds: no pixels or token ids, and a memory-free variant on one schema.

### M6. Statistics
- Three seeds, one test set of 100 receipts, "on every seed" sign statements, and about twenty contrasts read from the same runs. No interval, no test.
- Seeds vary training only, not the partition or the class order.
- What to do: five seeds for the five key rows, and a paired bootstrap over test documents for the contrasts under two points (learned against constant background, raw against latent memory, oracle against CoLaR-SBR).

### M7. Metrics
- AA uses "session-restricted" scoring, which maps predictions outside a session's label set to "O". It is non-standard and lenient for weak models (the appendix says so), yet it is the only metric the baselines have.
- F1 is micro-averaged while session 1 holds 57% of the test spans. Continual NER reports macro F1 as well.
- AF is defined against the F1 right after learning, not the usual maximum, and takes values such as −29.9 that the text does not explain. Bolding the lowest AF rewards methods that never learned (CPFD: AF 21.9 at AA 2.7).
- What to do: full-label micro and macro F1 for every row; keep AA and AF as secondary; do not bold AF, or bold it only among rows within ten points of the best AA.

### M8. No direct measurement of the conflict
- For shared-softmax runs the paper has "span counts but no per-token outputs", so the claim that old entities are pushed to "O" is bounded, not shown.
- What to do: from the seed-42 prediction dumps that already exist, add the confusion of old-entity tokens (to "O", to a same-session class, to another session) for CoLaR with a shared softmax, and one plot of the "O" logit margin over sessions. A short derivation of the gradient an "O" token sends to an unannotated class under a shared softmax, and its absence under Eq. 2, would add rigour to "the principle".

### M9. Cost is not reported in the main text
- From the result files: joint training 0.3 h, naive 1.3 h, CoLaR-SBR 2.1 h, raw memory + SBR 2.8 h, SBR-Recall 1.6 h per run. Joint training on everything is seven times cheaper than the proposed method on this benchmark.
- A reviewer will ask why one would not retrain. The answer has to be data retention, and then C5 bites.
- What to do: one row of training time and inference passes in Table 4, and one sentence that owns the trade-off.

### M10. Fit to the venue
- Nothing in the method uses vision or layout, the image appendix is a negative result with below-published SLCA numbers, and BERT appears only on DIL.
- What to do: either add a modality ablation on CIL-CORD (text only, text and layout, full) to show the phenomenon is modality-independent in a document model, or cut the image appendix and add one qualitative figure of a receipt with predictions from ER, CoLaR and CoLaR-SBR.

---

## 6. Section by section

### Title and abstract
- **Major.** "Document Understanding" is wider than what is studied (token classification for KIE). Generative models are excluded by the paper itself.
- **Major.** The finding is stated without scope; see C3 and C4.
- **Minor.** First sentence is hard to parse ("We study catastrophic forgetting issues in document understanding as key-information extraction (KIE) models must adapt…"). "…when the label space grows, known as class-incremental learning" has no subject for "known".
- **Minor.** The abstract does not say that SBR-Recall is the best row, nor give any cost.

### 1. Introduction
- **Major.** Paragraph 3 is a results section: five numbers and four hypothesis labels before the reader knows the setting. Keep one sentence per finding.
- **Major.** Contribution 3 is "we conduct experiments". Replace it by something a reader can use: the benchmark of six streams with code, or the measured decomposition.
- **Minor.** "we examine a ladder of hypotheses on two class-incremental streams" against Section 4, "Every rung runs on CIL-CORD".
- **Minor.** Table 1 sits on page 3, before any of its terms (CoLaR, blocked readout, AF, verdict words) is defined.
- **Minor.** No figure shows the headline result. A four-bar teaser (ER, CoLaR, CoLaR-SBR, joint) next to Fig. 1 would do more than Table 1 here.

### 2. Related Work
- **Major (length).** About 1,150 words in seven paragraphs. Target 450.
- **Major (missing).** Multi-dataset training with different label spaces in vision is the same problem and is not cited: Zhao et al., "Object Detection with a Unified Label Space from Multiple Datasets" (ECCV 2020), which names the ambiguous background explicitly, and Zhou et al., "Simple Multi-dataset Detection" (CVPR 2022). `beryozkin2019tagset` is in `refs.bib` and never cited.
- **Major (missing).** Bias correction in class-incremental learning (BiC, WA), sigmoid classifiers with replay, task-identity inference with adapters or prompts (HiDe-Prompt, CLOM), and continual-NER methods with replay or pseudo-labels (RDP, Learn-and-Review).
- **Minor.** A result (13.0 against 92.0) sits inside the first paragraph of Related Work. Some reviewers dislike results there; one clause is enough.
- **Minor.** The difference from ICE-O and MORE is one sentence. A five-row table (per-session background, learned background, memory, trainable encoder, needs task identity) would make the position clear in the space of three lines of text.
- **Minor.** Six bibliography entries have author lists typed from memory (LayoutLM, LayoutLMv2, GeoLayoutLM, Donut, UDOP, PNN). Check them.

### 3. Problem Formulation
- **Critical.** The overlap of documents across sessions is not stated (C5).
- **Major.** No formal definition. Write the stream as pairs (x, y^t) with y^t_i = y*_i if y*_i is in the session's tag set and "O" otherwise; the conflict is then one line.
- **Minor.** "the complete CORD test set with all 30 classes" defines the metric for one dataset only.
- **Minor.** The caption of Table 2 lists document counts for three datasets; the table now has a row for all.
- **Minor.** "Appendices A to F are in the supplementary material." interrupts the paragraph.
- **Minor.** The symbol for a session's tag set and the one for a block's tag set are the same letter with different indices; say once that a block is a session that brought new tags.

### 4. Why Replay Fails under Label Growth
- **Critical.** Scope of evidence (C3).
- **Major.** H0 is partly a straw man and is confounded: the ER row differs in three ways and is legacy; "raw beats compressed" is not what the section then discusses.
- **Major.** Forward references: the section uses CoLaR, the blocked readout, claims and owning blocks before Section 5 defines them, and uses the proposed method as a rung.
- **Major.** No statement of which predictions were written before the runs. The repository has pre-registration notes; one sentence saying so would answer the "hypotheses after results" objection.
- **Major.** Why the softmax slice fails (45.9) while masking works (89.7) is asserted ("stale tokens still train the shared 'O' logit"), not tested.
- **Minor.** Six verdict words (supports, refutes, partly, does not support, not needed, open). Use three.
- **Minor.** In Table 1 the row "raw-document memory, blocked readout: does not support" under H0 and "blocked readout, no memory: supports" at 11.4 F1 are hard to read.
- **Minor.** Fig. 2 has three panels of small type; panel (c) is unreadable at print size.

### 5. Method
- **Major.** No algorithm box, no complexity statement, and no word on how the 50 documents per session are chosen.
- **Major.** Freezing the embeddings and layers 1 to 4 after session 1 limits what later sessions can learn. DIL-XFUND (new languages, both proposed methods 3 points below ER) is the place where this shows; the paper does not connect the two. No ablation of the split layer.
- **Major.** The claim threshold of 0.5 is "not tuned", and claims of blocks trained at different times "need not be on one scale". No calibration analysis, and no test with more than five blocks. A long stream (ten or more sessions) is missing.
- **Major.** The all-background rule for other schemas needs the schema identity of each training document and "can call a real entity background". There is no ablation without it.
- **Minor.** Fig. 4 is a small single-column sketch. A full-width overview (encoder, frozen split, memory record, blocks, routing) is what CVPR readers expect.
- **Minor.** "The principle" is stated in words only (see M8).

### 6. Experiments
- **Critical.** C1, C2, C6.
- **Major.** Missing baselines that address the same problem: ER-ACE, SS-IL, sigmoid outputs with replay, MiB with its distillation plus replay, PLOP-style pseudo-labels plus replay, SpanKL (cited as avoiding the shared "O"), CFNER. L2P, DualPrompt and EWC were dropped for failing; a reviewer may ask for them in the appendix table with their numbers.
- **Major.** Table 3 has two streams that are almost entirely "--" and one column pair (CIL-FUNSD) with only our rows. Either fill them or move them out.
- **Major.** "Results on CIL-CORD" is four sentences; the comparison with baselines is not analysed (why is ER at 24.3? which sessions do baselines lose?).
- **Major.** On Mixed, CoLaR-SBR (89.1) is above joint training (87.6). The likely reason is that joint training with one softmax has the same conflict across schemas. This is an interesting point for the paper and is not discussed; "joint training with SBR" is the right reference there.
- **Major.** On DIL, CoLaR-SBR (86.2) is below CoLaR (87.8) although with one block the two readouts are almost the same model. The two cells also come from different legacy run families, so this is either a protocol difference or run-to-run noise of about 1.5 points, which is larger than several differences the paper interprets.
- **Minor.** The component ablation (the 2 × 2 of memory and readout) is in the supplement. It is the cleanest table the paper has and belongs in the main text.
- **Minor.** No qualitative example.

### 6.4 Memory efficiency
- See M5 and M9.
- **Minor.** Two paragraphs after Table 4 repeat sizes already in the table.

### 7. Limitations and conclusion
- **Strength.** The limitations are thorough and honest.
- **Minor.** One paragraph of 270 words; reviewers skim it. Four short items read better.
- **Minor.** The conclusion lists controls and no future work. End with the takeaway and one sentence on what comes next (your earlier papers do this).

### Supplement
- **Minor (error).** Appendix B says "rank 64 is not evaluated" (`supp_body.tex:49`); the main text and Table 4 report it.
- **Major (omission).** Generated tables that are not included anywhere: `supp_bros.tex` (second backbone on CIL-CORD), `supp_newstreams.tex`, `supp_perseed.tex` (per-seed values and hours per run), `supp_agree.tex`, `supp_dropped.tex`, `supp_delta.tex`. The BROS table should be in the main paper.
- **Minor.** Captions end with the word "Generated."
- **Minor.** Appendix F (images) reports SLCA and joint numbers below the published ones; it invites criticism and supports no claim of the paper.

### Compliance and reproducibility
- **Major.** No statement on releasing code, streams or run files.
- **Major.** The LLM usage statement mentions experiments and literature search only. The assistant also drafted and edited text in the abstract, introduction, related work, experiments and captions. The statement should say so; check the CVPR 2027 policy for the required wording.
- **Minor.** Hardware (one 6 GB GPU) and batch size 2 are disclosed; say that batch size is a hardware limit.

---

## 7. Inconsistencies and errors to fix (no new runs)

| # | Where | Problem |
|---|---|---|
| 1 | `supp_body.tex:49` | "rank 64 is not evaluated" contradicts Section 6.4 and Table 4 |
| 2 | Abstract, `main.tex:59` | "5 public document datasets" and "upper-bound baseline Joint" (C4) |
| 3 | `main.tex:75` | "eight CL baselines on six streams" |
| 4 | `main.tex:71` against `:121` | "two class-incremental streams" against "Every rung runs on CIL-CORD" |
| 5 | `main.tex:112` | Table 2 caption lists three datasets' document counts |
| 6 | `main.tex:117` | full-label F1 defined for CORD only |
| 7 | `main.tex:121` | "under the clean protocol" while two rows of Table 1 are legacy |
| 8 | Table 1 | negative AF values not explained; verdict wording |
| 9 | Table 3 | AF bold on methods that learn nothing; CIL-FUNSD joint is under-trained |
| 10 | Table 4 | SBR-Recall size is a computed fp16 size shown next to measured accuracy |
| 11 | Limitations | "four encoder-only backbones": three of them appear only on DIL, legacy |
| 12 | Appendix A, sessions table | "sub.nm, sub.cnt" appears in sessions 2 and 5 with different meanings |
| 13 | LLM usage statement | incomplete |
| 14 | Section 6.2 | "The clean CIL-WildReceipt runs are not complete" inside a results paragraph |

---

## 8. Questions reviewers will ask

1. Do the conclusions hold when each document appears in one session only?
2. What do ER and DER++ score in full-label F1 under your clean protocol with 250 documents?
3. Why is CPFD at zero? Can you reproduce its published numbers with your port?
4. With the same memory, how do sigmoid outputs, ER-ACE and a separated softmax compare with SBR?
5. Does the suppression of the newest session remain at a replay ratio of 1:1 and with 10 documents per session?
6. Why does a MiB-style loss match SBR on FUNSD and reach only 60 on CORD?
7. Is the learned background better than ICE-O's constant one beyond seed noise?
8. How does accuracy change with ten or twenty sessions, and how often is a token routed to the wrong block?
9. Joint training takes 0.3 hours. When is your method preferable?
10. What does CoLaR add over 8-bit raw documents if it is larger and no privacy is claimed?
11. Why is the joint reference 4.5 points below the published LayoutLMv3 number on CORD?
12. Three seeds and 100 test receipts: which differences are significant?

---

## 9. Road map to 7 or more

Estimated effect on the overall score is my judgement.

### Step 1. No new runs (about one day). Brings the paper to about 4.5 to 5
1. Cut to 8 pages: Related Work to 450 words; the investigation to 800 words by moving per-seed control details to the appendix "Rung details"; Table 1 and Fig. 2 merged or one of them moved to the appendix; the two prose paragraphs after Table 4 deleted; limitations as four short items.
2. Align every claim with the tables (Section 7, items 2 to 4, 11, 14) and scope the finding (C3d).
3. Put the BROS table and the component 2 × 2 in the main text.
4. State the document overlap and the memory fraction in Section 3.
5. Reframe the contributions: (1) the diagnosis with oracle controls; (2) a readout that implements it, as an instantiation of per-session softmaxes; (3) a memory-free variant with its conditions and costs.
6. Fix Table 4 (measured and computed sizes apart; data and parameters apart), stop bolding AF, add training time.
7. Add the missing citations and the comparison table against ICE-O and MORE.
8. Fix the LLM statement, add a code-release sentence.

### Step 2. Runs that fit on the current hardware (each CIL-CORD run is 1.5 to 3 hours). Brings it to about 6
In order of value per GPU hour:
1. ER and DER++ on CIL-CORD, clean protocol, 250 documents, three seeds (6 runs). Removes most of C1.
2. Joint training to convergence on CIL-CORD and CIL-FUNSD (6 short runs). Removes C6.
3. CIL-WildReceipt: naive, ER, CoLaR, CoLaR-SBR, SBR-Recall, three seeds. Makes "five datasets" true.
4. SBR-Recall with fp16 deltas, evaluation only. Makes Table 4 honest.
5. Sigmoid outputs with the CoLaR memory, and ER-ACE (6 runs). Answers M2.
6. Replay ratio and buffer size on CIL-CORD for CoLaR and CoLaR-SBR (about 8 runs). Answers M3.
7. Two more class orders for CoLaR, CoLaR-SBR and SBR-Recall (one seed each is acceptable if stated).
8. ICE-O without the epoch cap.

### Step 3. What makes it a clear accept (7 or more)
1. A disjoint-document variant of CIL-CORD (C5).
2. Port validation, or the text-only continual-NER experiment (C2). This is the single most persuasive addition, since it also answers "is this specific to documents?".
3. The CORD against FUNSD explanation (C3c).
4. A long stream of ten or more sessions with routing accuracy per block.
5. Five seeds and bootstrap intervals for the key contrasts.

If only three things can be done before submission: Step 1 in full, Step 2 item 1, and Step 2 item 3.

---

## 10. Reviewer scores

| Dimension | R1: experiments | R2: impact | R3: novelty | Mean |
|---|---|---|---|---|
| Overall (1 to 10) | 4 | 3 | 5 | 4.0 |
| Soundness (1 to 4) | 2 | 2 | 3 | 2.3 |
| Presentation | 2 | 2 | 3 | 2.3 |
| Contribution | 2 | 2 | 3 | 2.3 |
| Originality | 2 | 2 | 3 | 2.3 |
| Quality | 2 | 2 | 3 | 2.3 |
| Clarity | 2 | 2 | 3 | 2.3 |
| Significance | 2 | 2 | 3 | 2.3 |
| Confidence (1 to 5) | 4 | 4 | 4 | 4.0 |
| Decision | Reject | Reject | Borderline | Reject |

Weighted score with the skill's weights (Overall 1.0, Contribution 0.4, Presentation 0.2, the rest 0.1 each): 5.1 / 10.

- **R1 (harsh but fair, wants experiments that give insight).** "The oracle controls are the best part and I believe the effect on CORD. I cannot accept a benchmark table in which the baselines ran under another protocol, never report the headline metric, and include a method at zero. One partition of one dataset is not enough for a causal claim."
- **R2 (harsh, looks for impact).** "Background shift is known in segmentation, detection and NER; the readout is ICE-O with replay. The streams are tiny, joint retraining takes twenty minutes, and the documents recur in every session. I do not see the deployment this solves, nor what is specific to vision."
- **R3 (open-minded, looks for new ideas).** "Separating what a memory repairs from what the labels break, with oracles, is a clean idea, and the memory-free variant is interesting. The paper reads as unfinished: empty columns, a contradicting second stream that is not explained, and three pages too long."

All required sections are present (abstract, introduction, method, experiments, limitations and conclusion).

---

## Sources checked for this review
- LayoutLMv3-base on CORD (about 96.5 F1) and FUNSD (90.29): [GraphLayoutLM, Table 1](https://arxiv.org/pdf/2308.07777), [survey table](https://arxiv.org/pdf/2403.04080)
- Background ambiguity across detection datasets: [Zhao et al., ECCV 2020](https://arxiv.org/abs/2008.06614)
- Continual-NER benchmarks and reported gains of CPFD and RDP over fine-tuning: [CPFD](https://arxiv.org/pdf/2310.14541), [SpanKL](https://arxiv.org/pdf/2302.12200), [Learning "O" Helps](https://arxiv.org/pdf/2210.04676)
