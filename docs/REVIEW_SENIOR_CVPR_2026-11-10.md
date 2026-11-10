# Senior-reviewer read of the CVPR draft (2026-11-10, 8-page body, commit after the page cut)

Scores on the CVPR scale. Recommendation: **Borderline / Weak Reject as framed; Accept-track if reframed** (see "What would change my score").

## Summary
Diagnosis-first study of forgetting in continual key-information extraction on LayoutLMv3/LiLT/BROS/BERT. Claims: (1) forgetting is head/late-layer localised and architecture-general, and protecting the locus relocates rather than removes it; (2) six classical remedies order as the diagnosis predicts, with an EWC sign flip across backbones; (3) five buffer-free families fail for one cause (readout-marginal snap); (4) CoLaR, per-document truncated-SVD latent replay, matches replay on DIL at 60 MB with no raw bytes; (5) class-incremental "floor" is mostly a scoring artefact plus a shared-background-logit contradiction, fixed by a session-blocked readout (SBR, 88.0) and per-session LoRA recall (SBR-Recall, 92.8 > oracle).

## Strengths
- Unusually honest reporting: seed counts per cell, disclosed nulls, pre-registered kill-tests, negative results kept.
- The scoring-artefact finding (oracle 33 → 92 under session-restricted scoring) is a genuinely useful correction for the document-CIL sub-field.
- SBR-Recall is a clean, strong result: flat retention columns, > joint on three seeds, no bank needed on a single schema. The failure analysis on Mixed (no-bank 44.7 → bank 85.9) is the kind of mechanism-level reporting reviewers rarely get.
- Multi-backbone protocol for a claim class (buffer-free remedies) where single-backbone papers routinely disagree.

## Major concerns
1. **Three papers in one; the headline is misaligned with the title and the teaser.** Title and Fig. 1 sell CoLaR and even depict a component the caption calls a null (the OCR router). The strongest, most novel result is SBR/SBR-Recall (§8, half a page, last). A CVPR reader decides the contribution from title + Fig. 1 + contributions list; today those point at a latent-replay variant that ties ER on DIL. Either retitle/reteaser around "label growth is a readout problem" or cut SBR to a short section and own CoLaR as the contribution. Trying to carry both at 8 pages leaves each under-evidenced.
2. **Statistical claims exceed what 3 seeds can support.** "A per-component test rejects the null ... (p < 0.01)" via Mann–Whitney with n = 3 per group is not possible unless groups are pooled across backbones/conditions; state exactly what was pooled. "Above the joint oracle on every seed" (92.8 ± 0.5 vs 91.9 ± 1.3) is within overlapping spreads; say "at or above", not "above". The "conservation law", the +27 single-variable control, and Finding 2's freeze arms are seed-42 single runs presented as laws.
3. **Comparator fairness in Table 3 (the table that carries the method claim).** (a) Baselines bf16, SBR rows fp32. (b) CODA-Prompt and CL-LoRA run without their official old-class logit masking and are then reported as collapsing. (c) CPFD 1.4 and IS3 6.6 on CIL-CORD, far below naive, read as misconfiguration, not as a property of the methods; "did not converge" needs a diagnosis or the rows should be dropped. (d) Most baseline rows are two-seed while the method rows are three. A reviewer will ask whether any shared-softmax baseline was given the same restricted-scoring-aware care SBR received.
4. **Test-set early stopping on FUNSD/SROIE.** Disclosed, but it touches DIL and Mixed headline cells on both sides. The held-out re-run promised "in the supplement" must be in the submission, not promised.
5. **Privacy claim is asserted, not tested.** "No raw bytes" is not privacy: layer-4 latents of a 12-layer encoder are invertible to a useful degree, and boxes + BIO labels + token count already leak document structure. Either run a feature-inversion / membership probe or soften to "stores no raw pixels or token ids".
6. **The storage comparison is ambiguous.** "Matching raw replay at 2.7× less storage" compares with raw *latent* replay (163 MB), not with ER's 200 raw documents, whose byte cost is never given. The abstract reads as the latter.
7. **SBR-Recall lacks its natural baselines.** It is parameter isolation with task-agnostic routing; compare against per-task LoRA + entropy/energy routing, HAT/Expert-Gate style, or at least CL-LoRA with the same O_b routing. CL-LoRA at 18.6 "despite a diagonal of 93" is suggestive but is not the controlled comparison.
8. **Generality claims outrun the evidence.** Three English datasets, 149–800 documents each; "architecture-general" from four BERT-like encoders; DIL on a 4-class unified schema. One XFUND-style multilingual or a larger receipt corpus would help more than another ablation.

## Minor
- Legacy-scored CIL-CORD columns in Tab. 1/2 are, by the paper's own argument, measuring an artefact; consider moving them to the supplement and keeping one scoring scheme in the body.
- In-house method names (LexSlot, LexMem v3b/v5, CoLaR-R, HGT, CUBER, Ledger) appear without definition in the body now that the ledger table is in the supplement.
- "Registered"/"pre-registered" kill-tests: say where (internal docs are not a registry); otherwise write "specified in advance".
- Prose carries too many numbers; several paragraphs are number lists. Move final-row vectors to tables.
- LwF rows with one collapsed seed (12.3 ± 10.7) should be re-run before submission, not footnoted.
- Code "on acceptance": CVPR reviewers increasingly expect an anonymised repo at submission.

## Questions for the authors
1. What exactly is pooled in the Mann–Whitney/Bonferroni test, and what is the actual p-value?
2. Does SBR's supervision rule need the dataset-schema oracle (which schema a session belongs to) at training time? If so, state it as an assumption.
3. Inference cost of SBR-Recall: B upper-trunk forwards per document. Latency numbers?
4. Why is SBR-Recall's Mixed row single-seed when the method rows elsewhere are three?
5. Would SBR alone (no replay) with a frozen trunk be the right "readout-only" control? You report 14.8; is that at matched epochs?

## What would change my score
Reframe around the readout finding (title, teaser, contributions 1–2 compressed to motivation), equalise the comparator protocol (precision, logit masking, three seeds, held-out early stopping), add one isolation baseline for Recall, and either test or soften the privacy claim. With those, this is a clear accept at CVPR; the diagnosis material then reads as motivation rather than as a second paper.
