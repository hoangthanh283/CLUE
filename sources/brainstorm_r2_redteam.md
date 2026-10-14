# Brainstorm round 2 — RED TEAM + RECOMBINE (2026-10-14/15)

Inputs: `brainstorm_r1_{memory_labels,optimisation,vision_layout,protocol_benchmark,readout_head}.md`.
Finalists = all candidates ≥10/14 + readout top-2 (Snap index, Keep-correctable).

## New measured facts (this round, CPU only)

**Document re-show across sessions** (`ex["row"]` identity on the training splits):

| Stream | Session sizes | Docs already seen in an earlier session |
|---|---|---|
| CIL-CORD (5×6) | 800 / 382 / 549 / 723 / 780 (800 distinct) | s1 **382/382**, s2 **549/549**, s3 **723/723**, s4 **780/780** — every later session is a subset of session 0 |
| Mixed | FUNSD 149 → FUNSD 147 → SROIE 626 → CORD 800 → CORD 798 → FUNSD 149 | s1 147/147, s4 798/800, s5 (revisit) 149/149; SROIE once |

Consequence: on CIL-CORD the per-document union of labels observed up to session t equals the
"oracle" refresh labels (all classes ≤ t present in that receipt). **CoLaR-R-oracle is not an
oracle on these streams — it is implementable from the stream with a ≤0.5 KB/doc label ledger.**
The same ledger also fixes E2O on *live* data exactly (the live session-t receipt was seen with
its old-class labels earlier), which CPFD/IS3 only approximate.

Caveat that must be stated: standard CNER protocols (CFNER / ExtendNER lineage) partition the
corpus into *disjoint* slices by greedy sampling ("Split-All", "Filter-Filter"); re-show is a
property of dense multi-class documents filtered by "≥1 in-session entity", i.e. of document
streams, not of CNER in general. The ledger therefore (i) is a legitimate CL memory (labels, no
inputs, <1 MB), (ii) generalises to any stream where documents recur (annotation extension —
the realistic deployment case), (iii) gives nothing under disjoint sampling — there only the
label-free fixes (sliced softmax, pseudo refresh) apply. Both regimes must appear in the paper.

**Novelty spot-checks (WebSearch extended, 2026-10-15):**
- Sliced / masked softmax on replay: SS-IL (ICCV'21), ER-ACE (ICLR'22), "Revisiting Softmax
  Masking" (arXiv 2309.14808) — all image CIL; IS3 (ACL'24 Findings) scales old-row gradients
  on live data. Application to a replay buffer on a growing token head is unreported → **N=1**
  (confirmed as a control, not a contribution).
- Label union via document identity in CNER: no hit; closest is RCLPOD (predicted labels on
  replay, object detection). **N=2** holds, but reviewers will frame it as "exploiting the
  benchmark" unless the re-show table and the disjoint-regime caveat are in the paper.

## Part A — finalists under attack

Scores: round-1 → revised. Verdicts: KEEP / DEMOTE / KILL.

| Finalist (lens) | R1 | Attack | Rev | Verdict |
|---|---|---|---|---|
| Method-rank inversion (protocol) | 13 | Needs a real SLCA cell on documents, else it is "B2 re-read as SLCA" and a reviewer will say the port is missing; SLCA-token ≈ 1 d + 3 DIL runs (~8 GPU-h) — honest. Significance: genuine (explains why PTM-CL favourites fail on structured outputs). | 12 | KEEP (do the port) |
| Headroom certification (protocol) | 12 | Write-only; survives everything; risk = "nice analysis" unless tied to a ranking failure — it is (replay cells tie within noise). | 12 | KEEP |
| DocCL benchmark release (protocol) | 12 | Novelty real (no LayoutLM-family CL suite). A release is not a result; CVPR scores it under reproducibility, not significance. | 11 | KEEP (supp + README) |
| Position-axis anatomy (vision) | 12 | Needs checkpoint pairs at boundaries; pilot saved CKA/Fisher JSON, not weights → 1 naive DIL run per seed (~1 h each) + position-masked `collect_activations`. Novelty 2 (encoder-only per-position locus unreported). A finding, not a gain. | 11 | KEEP |
| Silent-modality probe (vision) | 12 | 2 seed-42 cells (~5 h). Fixes D2's confound; either outcome is a sentence, "structure" outcome is a short one. | 11 | KEEP |
| Document Ledger (memory) | 11 | Re-show measured at 100 % (CIL-CORD) / 98–100 % (Mixed CIL legs) → effect bounded by the oracle, expected large. Biggest risk is framing (label leakage); mitigated by the disjoint-regime caveat and by shipping the label-free variants alongside. Novelty 2. Feasibility: side store keyed by row, ~60 LOC; reuses `labels_full` plumbing (ledger = running union of observed labels instead of privileged full labels). | 12 | KEEP (headline) |
| Capture-space-masked replay CE (optim.) = Sliced softmax (memory) | 11 / 10 | Same idea proposed twice (convergence). Novelty 1 after lookup (SS-IL/ER-ACE/IS3 adjacent). Value = the mechanism-isolating control: removes the stale negative gradient without adding positives. One line in `_replay_forward_inner`. | 10 | KEEP as control (not a contribution) |
| Conservation on ViT (vision) | 11 | ~6 GPU-h; `freeze_groups` exists; reuses cil_cifar100. Makes conservation modality-general — the strongest vision hook if positive. | 10 | KEEP |
| Conservation as a metric (protocol) | 11 | Write + script over existing displacement records; survives. | 10 | KEEP (write) |
| Method×backbone interaction test (protocol) | 11 | Write + stats on 68 cells; cite arXiv 2604.21927 (depth flips rankings). | 10 | KEEP (write) |
| Modality-stream attribution as step 0 (protocol) | 11 | Packaging of existing C1–C4 data; overlaps vision lens. | 9 | DEMOTE (fold into protocol figure) |
| Modality-asymmetric rank (vision) | 11 | DIL-only bytes lever; conservation law of memory-side levers predicts ≈0 AA; 2 cells. | 8 | KILL this cycle |
| Position-ablation ladder (vision) | 11 | Two rungs already exist (D1/D2); the missing rung is the silent-modality cell. | 9 | DEMOTE (merge into silent probe) |
| Abstention refresh (memory) | 10 | Dominated: ledger (exact) above, pseudo below; only matters if pseudo-labels are noisy. | 9 | DEMOTE (run only if oracle>pseudo gap is large) |
| Privacy inversion audit (memory/vision) | 10 / 9 | Needs a decoder trained on public pages (~0.5 d GPU); can *weaken* the "no raw bytes" claim — honesty gain, significance loss. Activation inversion exists (DRAG ICML'25). | 9 | KEEP optional (supp, after Gate 1) |
| Depth-targeted replay gradients (optim.) | 10 | Masking replay gradients away from early/mid is a protection-like constraint; Finding 2 says constraints relocate drift; DIL replay is already at oracle. Predicted ≤ +0.5. | 8 | DEMOTE (ablation only if GPU idle) |
| Gradient-conflict diagnostic on head rows (optim.) | 10 | One hook, one JSON, zero extra runs if attached to the queued cells. Gives P1 its mechanism figure (cos<0 on new rows under stale labels; ≥0 after masking). | 10 | KEEP (attach to P1 runs) |
| Stale-label audit (protocol) | 10 | With 100 % re-show the audit is exact and GPU-free: fraction of buffered/live tokens whose label is stale per session. Motivation figure for P1. | 11 | KEEP (write/script) |
| Snap index (readout) | 9 | Needs per-boundary logits → only for new runs; cheap hook; label-free monitor is a supp instrument, not a CVPR headline. | 9 | KEEP (supp, hook into P1 runs) |
| Keep-correctable cap (readout) | 9 | Conservation predicts the information dies in features → likely inert; 9 GPU-h; informative null but third-order. | 8 | DEMOTE (only if GPU idle after Oct 26) |

Budget reality check: one 6 GB GPU, R2 queue holds 30 cells at ~3–5 h (CoLaR family) / ~1.5–3 h
(CPFD/IS3) ≈ 100–130 GPU-h → the queue alone runs to ~Oct 19–20, not Oct 18. Window to the Oct 30
mechanism freeze ≈ 250 realistic GPU-h. P1 additions (ledger 6 cells, sliced 6 cells, ER+ledger 3
cells ≈ 50 h) + P2 (8 h) + P3 (15 h) fit only if seeds 7/123 of the lowest-priority queue items
(IS3, CPFD on Mixed) are capped at `method.epochs=5` or dropped to seed 42 + 7.

## Part B — packages (ranked)

### P1 — "Live labels for a drift-free memory" (headline mechanism)
**Claim:** *On class-incremental document streams, replay falls below naive fine-tuning because the
buffered labels are stale; keeping the labels live — exactly via a ≤1 MB per-document label
ledger, or approximately via label-free pseudo-refresh / sliced-softmax replay — recovers ≥4 AA
of the 14-point CIL-CORD gap and ≥1 AA on Mixed, on three seeds.*
Components: CoLaR-plain, CoLaR-R-ledger (replaces "oracle"; same code path, labels sourced from the
stream), CoLaR-R-pseudo, CoLaR-sliced (control: negatives removed, no positives), ER+ledger
(buffer-general?), CPFD, IS3 (comparators); stale-label audit figure (GPU-free); head-row
gradient-conflict hook on the same runs; re-show table + disjoint-regime caveat paragraph.
GPU: queued 30 cells (~110 h) + 15 new cells (~50 h). Writing: 3 days (method ¶, Table 1 rows,
mechanism figure, caveat). Rubric: novelty 2, **significance 3 (the only AA mover)**, rigor 3
(controls isolate labels-vs-gradient, exact-vs-pseudo, latent-vs-raw), vision 0, reproducibility
yes, honesty: requires the re-show caveat.
Biggest risk: reviewers read the ledger as benchmark exploitation. Mitigation is structural, not
rhetorical: report the re-show rates, show the label-free variants, and state that under disjoint
CNER sampling only those apply.
Kill: ledger ≤ plain+1 AA on CIL-CORD (then stale labels are not the cause and P1 becomes a
ledger row — the stale-label audit still explains why). Secondary kill for the deployable claim:
pseudo and sliced both ≤ plain+1 while ledger passes → "exact labels required" (still a result).

### P2 — "Evaluation methodology for document CL" (guaranteed section)
**Claim:** *Two measurements must precede any method ranking on document CL: the Joint−Independent
headroom (≤0.5 and negative on LiLT/BROS for DIL, so DIL cannot rank methods) and the
method-rank inversion across domains (head-statistics methods beat latent replay on image CIL and
are null on token-level documents, because whole-document binding, absent in image
classification, is what the readout needs).*
Components: SLCA-token port + 3 DIL runs (the document cell; B2 as corroboration), headroom-
normalised recovery table, method×backbone interaction test on the 68-cell grid, conservation-
as-metric paragraph, DocCL benchmark + protocol release (supp/README).
GPU: ~8 h. Writing: 3 days. Rubric: novelty 2–3, significance 2 (practice-changing for evaluators),
rigor 3, vision 1 (image-CIL side of the inversion), reproducibility 3, honesty 3.
Biggest risk: scored as "analysis" — mitigated by the SLCA-token falsifier making the inversion a
tested claim.
Kill: SLCA-token ≥ CoLaR−1 AA on DIL ⇒ inversion false; the headroom step stands regardless.

### P3 — "Where forgetting lives inside the sequence" (vision hook)
**Claim:** *Within a shared document encoder, drift is text-position-localised: visual-patch
positions stay representationally fixed (CKA ≈ 1.0) while text positions drift, yet replay needs
the visual positions as structural carriers (silent-modality probe); and the conservation law
reproduces on a pure-vision ViT stream.*
Components: position-axis anatomy via `doccl.diag` with a position mask (1 naive DIL run/seed if
no checkpoints), silent-modality cell ×2 (seed 42), conservation-on-ViT freeze arm ×3 seeds.
GPU: ~15 h. Writing: 2 days. Rubric: novelty 2, significance 1–2, rigor 2, **vision 2**,
reproducibility 3, honesty 3.
Biggest risk: all three are findings; if visual tokens are "structure only" and ViT migration is
weak, P3 shrinks to one paragraph.
Kill: visual-position CKA at layer 11 within 0.05 of text-position CKA (no within-sequence locus);
ViT early/mid CKA > 0.9 under head+late freeze (no migration).

Ranking: **P1 > P2 > P3.** P2 is executed in parallel regardless (write-only except 8 GPU-h); P3
runs in the GPU gaps after the P1 queue.

## Part C — convergence

Clear #1: **P1**, because it is the only package that moves accuracy on a scenario with headroom,
and the round-2 measurement (100 % document re-show) upgraded its "oracle" arm from a ceiling to an
implementable method. Two independent lenses converged on the same control (sliced softmax), and
a third (protocol) produced its motivation figure (stale-label audit) — convergence signal.

If the running CoLaR-R cells come back null (ledger/oracle ≤ plain+1): the #1 package **changes to
P2**. P1 collapses to a falsification-ledger row plus the stale-label audit (which would then say
stale labels are *not* the cause of replay < naive — itself worth a sentence), and the paper ships
as the diagnostic + evaluation-methodology paper. P3 is unaffected by that outcome. No further
brainstorm rounds are warranted: the #1 is stable across rounds conditional on one running
experiment, and the alternative is already fully specified.
