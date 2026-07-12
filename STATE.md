# STATE

## ACTIVE (2026-07-10 late): SLR falsified → successor idea "Consistency Law → PLaR/CoLaR"

SLR/AGLR/coreset all FALSIFIED (see GATE-0 FINAL below) but proved the **consistency law**
(whole-doc (feature,position,label) binding is the necessary replay ingredient; +27 AA single-var
control). Successor idea (wiki `ideas/2026-07-10-consistency-law-replay.md`): keep consistency,
remove/shrink private storage —
- **PLaR (proxy latent replay):** bank pseudo-labeled PUBLIC docs (WildReceipt) as whole-doc
  frozen-trunk latents; ZERO private bytes. Implemented (`doccl/methods/proxy_latent_replay.py`,
  3 tests green, registered).
  **First run (dil 5ep, 5 docs, τ=0): AA 45.8 [FUNSD 40.2, SROIE 4.1, CORD 93.1]** — beats every
  falsified marginal (36.7–41.9; FUNSD retention 40.2 vs their 14–29 = the consistency effect is
  real on public docs) but below real-private-doc replay at same count (d4=63.8) and below the
  ≥70 gate. Mid-task SROIE still collapses.
  **d50 RESULT: AA 57.2 [FUNSD 73.3, SROIE 5.1, CORD 93.3]** — the count lever WORKS where
  pseudo-labels are informative: FUNSD 40.2→**73.3** (ABOVE private-doc d4's 60.8!). But SROIE
  stays dead (5.1). **Diagnosis: class density.** FUNSD labels are dense (most form tokens are
  question/answer → proxies carry rich signal); SROIE has 4 sparse entity types → argmax
  pseudo-labels on receipts ≈ 99% "O" → replay carries no SROIE-class gradient.
  **SOFT RESULT (00:52): AA 58.9 [FUNSD 76.1, SROIE 7.0, CORD 93.5]** — incremental (+1.7 AA);
  FUNSD keeps climbing (76.1 = +15 over matched private d4!) but SROIE barely moves. So sparse-class
  starvation is only part of it: either the head puts ~zero SROIE-class mass on WildReceipt tokens
  (now SELF-DIAGNOSED: banking logs a pseudo-label class histogram per boundary), or the mid-task
  needs the CONVERGENCE budget (every PLaR run so far = 5ep diagnostic; the 87.3 reference needed
  10ep early-stop).
  **CONVERGENCE RUN IN FLIGHT overnight** (`_run_plar_d50_soft_conv`, task `by803p7bw`, ~2.5-3h):
  d50+soft at the default 10ep/val-F1-early-stop. ⚠️ writes to the SAME run-name dir
  `dil_proxy_latent_replay_seed42_d50_soft` (epochs not in name) — the 5ep result is archived at
  `..._d50_soft_5ep_archive/`. Morning read: (1) the class histograms at each boundary (does the
  post-SROIE head label ANY proxy tokens with SROIE classes?), (2) the converged matrix. Decision:
  SROIE ≥30 → PLaR fully alive → Vast.ai sweep + CoLaR build. SROIE still ~7 + histogram shows no
  SROIE mass → the proxy pool carries no SROIE signal → mixed/matched pool or class-balanced
  weighting; if those fail, PLaR is a *partial* method (grounds dense-label tasks only) and folds
  into the law paper as the bounded-transfer finding.

  **PLaR ladder so far (dil, zero private bytes):** d5-hard 45.8 → d50-hard 57.2 → d50-soft 58.9
  → d50-soft-CONV pending. References: private d4-5ep 63.8, private d5-conv 87.3, best falsified
  marginal 41.9, naive ~41.

  **DIAGNOSIS v1 (01:30) — RETRACTED at 02:15.** First reading of the histograms ("head assigns
  zero SROIE-class labels; task 0 captures the pool's LABELS") was label-centric and WRONG for
  dil: `build_dil` gives every task the SAME `DIL_UNIFIED_LABELS` (9 shared BIO tags — there are
  no task-specific classes to capture; histogram ids 0-6 are 7/9 unified tags, ids 7-8 = the
  OTHER tags simply never predicted on proxies). Consequently the built `task_masked_labels` fix
  is a **NO-OP on dil** (kept in the code — tested, and genuinely relevant for CIL later); the
  planned tmask dil run was NOT launched (would prove nothing).

  **DIAGNOSIS v2 (02:15, fits ALL of tonight's data): FEATURE-REGION COVERAGE.** Proxy replay
  anchors the head's conditional only on the PROXY POOL'S OWN feature region (as encoded by the
  trunk that was FUNSD-tuned before freezing). FUNSD's region is covered → held (76.1 at 5ep);
  SROIE's region (different OCR/layout stats, encoded through a FUNSD-shaped frozen trunk) is NOT
  covered by WildReceipt latents → drifts freely (5.5) no matter the label format (hard/soft
  identical) or count (d5→d50 no help on SROIE). Also explains why CONVERGENCE HURT proxy replay
  (58.9@5ep → 50.3@conv) while it helps real replay (63.8→87.3): longer training = more drift in
  the uncovered region while the anchor holds only elsewhere — a noisy/offset anchor amplifies.

  **MORNING PLAN (in order, cheap→decisive):**
  1. **Coverage probe (no training, ~1 min):** mean/quantile L2 (or CKA) distances at layer 8
     between each task's latents and (a) the WildReceipt pool, (b) the other tasks. Prediction:
     d(SROIE, pool) ≫ d(FUNSD, pool). Confirms/kills v2 before any run.
  2. **Coverage-targeted proxy selection (~20 lines):** at boundary t, RETRIEVE from the pool the
     `docs_per_task` docs nearest to task t's latent centroid (CIL-QUD-style) instead of random.
     Run dil 5ep d50+soft. If SROIE recovers → PLaR alive + the coverage law is the paper's second
     mechanism. If the pool simply contains NO docs near SROIE's region (probe shows a floor) →
     enlarge/mix pools (XFUND + more corpora) or accept bounded-transfer.
  3. Optional: pool-mix ablation; CoLaR build (per-doc SVD, independent of the proxy question).
- **CoLaR (compressed latent replay):** per-DOC SVD is low-rank (r64=87.7%, r128=95.2% measured —
  unlike the full-rank pooled space) → 6–12× private-latent compression preserving whole-doc
  binding. NOT built yet; next after PLaR verdict.

## SUPERSEDED same-day (kept for the record): DIRECTION PIVOT → method-led ICML paper "SLR"

**This supersedes the 2026-07-04 "diagnostic + falsification, NOT a method paper" lock recorded
lower in this file.** Trigger: Exp #5 latent replay (dil AA 87.3 / BWT −2.3, ≈ ER-200, −1.4 from
joint oracle, **zero raw documents stored**) is a *method*, not a tombstone — and a new NTK theorem
("Catastrophic Forgetting is Low-Rank", arXiv 2606.18024, ICML'26 wksp) formalizes our migration law.

**SLR — Spectral Latent Replay (one line):** generative latent replay, rank-budgeted to the
forgetting eigenmodes, injected as gradients confined to the low-rank old-task subspace, on
structured head-growing document IE (BIO CIL → Re-DocRED relations).

**User decisions locked (2026-07-10):** (1) method-led spine; (2) include the Re-DocRED relational
extension as a headline; (3) feasibility on local RTX 2060 first, headline sweep on Vast.ai.

**Design + pre-registered gates** live in the wiki:
`CL4IE/wiki/ideas/2026-07-10-spectral-latent-replay-icml.md` (+ 4 new `CL4IE/wiki/sources/` pages:
forgetting-is-low-rank, aglr-cl-generative-latent-replay, replay-can-increase-forgetting, gpm; and
provable-effects-data-replay). Gate order:

- **Gate 0 (existential, LOCAL, do first):** beat a faithful AGLR-CL port (GMM whole-latent replay,
  the closest prior art). No delta → no paper.
- **Gate 1:** rank-`r` spectral memory vs raw-latent buffer at equal bytes (target AA ≥ 84 @ ≤25% bytes).
- **Gate 2:** subspace-projected replay vs full-gradient replay.
- **Gate 3 (Vast.ai):** AA-vs-memory-bytes Pareto sweep over `r` = the headline figure.
- **Gate 4:** cross-backbone (LiLT/BROS/BERT).
- **Gate 5:** Re-DocRED continual-relation stream (relational headline extension).

**Code base:** existing `latent_replay` method (commit 93aa144, 13 tests green) — do NOT rebuild.
**bd is write-blocked** (schema-migration fork, see memory [[clue-bd-dolt-lock]]) → SLR tasks tracked
here + in ROADMAP, not in beads, until the user reconciles bd.

**Gate-0 implementation progress (2026-07-10 session):**
- Built `SpectralMemory` (`doccl/methods/spectral_memory.py`, SLR) + `AGLRReplay`
  (`doccl/methods/aglr_replay.py`, the AGLR-CL comparator), both subclassing `LatentReplay`
  and sharing its layer-`k` replay hook. Wired into `train.py` (registry + `_STD_FORWARD` +
  run-name suffixes), configs added, 12 unit tests green (incl. 2 GPU-free regressions for the
  LayoutLMv3 image-patch width bug: layer-`k` hidden is text+patches = 709, wider than the
  512 attention_mask; capture slices to text, replay reconstructs at full width).
- **Bug found + fixed in first smoke:** initial SpectralMemory synthesised replay features
  from the *global* task distribution, decoupled from each token's label → replayed
  head-gradient taught noise → AA 39.3 / BWT −56 (worse than naive). Rewrote as
  **class-conditional inside the shared low-rank basis**: per-class Gaussians in the top-`r`
  forgetting subspace, so feature↔label stay coupled AND the "index by forgetting subspace"
  identity holds. Both methods vectorised (per-token loop → gather+matmul, 57→25 ms/call).
- **Operating-point finding (important):** 1-epoch is NOT a valid judging budget. Reference
  `latent_replay` at 1 epoch = **AA 64.9** (not the converged 87.3) — under-converged by ~22
  pts. So Gate 0 MUST run at convergence (grid default: ~10ep, val-F1 early stop), not on
  1-epoch smokes. Corrected class-conditional `spectral_memory` convergence run is IN FLIGHT.
- **Memory razor confirmed in bytes:** latent_replay banks 5.4–16 MB (raw docs);
  spectral_memory banks ~0.15 MB (≈36–70× smaller). Whether it holds AA is the pending answer.
- **⚠️ Clobber to flag:** my 1-epoch reference run wrote metrics to the run-name dir
  `results/dil_latent_replay_seed42/` (hydra.run.dir only redirects logs, not metrics),
  overwriting a prior result there (now AA 64.9, 1-epoch). `results/` is not git-tracked so the
  prior value is unrecoverable, but it is regenerable by re-running at convergence and STATE
  records the canonical 87.3. Future smokes must use a method/seed that doesn't collide.

**Compute reality (2026-07-10):** a converged dil run WITH active replay is ~90–120 min on the
local RTX 2060 (CORD alone ~5 min/epoch × up to 10ep; a 50-min timeout died mid-CORD at ep4/10,
no matrix). The local box has now served its purpose — it proved the *mechanism* (methods run
e2e on real LayoutLMv3, replay fires, class-Gaussians bank, 36× memory compression, the
class-conditional fix is correct). **Recommendation: move the actual Gate-0 comparison + the
whole gate sequence to Vast.ai** (user pre-approved), where a dil run is minutes not hours. A
capped `method.epochs=5` local run is in flight ONLY to get a first read on the retention matrix
(row 3 = does task-0/task-1 survive); it is a feasibility confirmation, not the headline number.

**Observability gap (follow-up):** replay loss is only visible via tqdm postfix (`\r`, not
flushed to piped logs) and `train/loss` is logged once per task-end, so replay *magnitude*
can't be watched live. Mechanism is verified 3 ways (unit test injection→plastic-grad, fit log
shows banked class-Gaussians, `_sample_replay` returns non-None when populated), but before the
Gate-1 rank sweep, add a per-step `train/replay_loss` TB scalar to `LatentReplay.train_task`
(or override in SpectralMemory) so replay activity is directly auditable.

**GATE-0 RESULT (2026-07-10, spectral_memory dil 5ep, class-conditional, rank-16): NEGATIVE.**
AA 41.9 / BWT −66.3, matrix row3 [FUNSD 29.3, SROIE **3.2**, CORD 93.1] — the LexMem-v5 mid-task
collapse signature. Only the current task survives; all prior tasks destroyed. Memory 0.454 MB.

**Diagnosis (the failure is isolated and mechanistic):** the difference vs latent_replay (which
grounds the head to AA 87.3 at the SAME frozen `k`=8, replaying REAL activations) is *real vs
synthesized* features. A rank-16 per-class Gaussian is too crude an approximation of the layer-`k`
feature distribution — the head trained on that thin surrogate can't hold its boundary against
the trunk's rich real features for the new task. NOT the label-coupling (fixed, still collapses),
NOT the frozen boundary (real replay at same k works). **The compression itself is the failure:
forgetting may be low-rank in OUTPUT space (the theorem), but reconstructing head-INPUT features
from a low-rank model loses what the head needs.** The razor's premise ("compress to the
forgetting rank and it still works") is FALSE at rank 16 on doc-IE.

**RANK-FIDELITY PROBE (2026-07-10, cheap GPU probe, `scratchpad/rank_probe.py`): the razor's
premise is broken.** Layer-8 LayoutLMv3 features are NOT low-rank — cumulative variance explained:
r16=42.9%, r32=55.2%, r64=68.3%, r128=81.0%, r256=91.6%, r512=98.3%. So rank-16 discarded 57% of
the feature signal (why the head collapsed), and you need r≈256 (1/3 of d=768) for 90% — which is
NOT compression. **The theorem's low-rank is the OUTPUT/NTK space; the head-INPUT features are
near-full-rank. SLR conflated the two.** A rank that works ≈ a rank that doesn't compress → the
accuracy-vs-memory Pareto curve the paper rests on almost certainly does NOT dominate raw replay.

**FORK RESOLVED (2026-07-10): AGLR-CL full-`d` ALSO COLLAPSES.** dil 5ep: AA 39.4 / BWT −69.8,
row3 [FUNSD 22.4, SROIE **3.0**, CORD 92.9], mem 0.387 MB. Full-`d` per-class Gaussians (NO rank
truncation) are *worse* than SLR rank-16 (41.9). So rank was never the issue.

**DECISIVE VERDICT: generative *feature-space* latent replay is dead for doc-IE.** Both variants
(rank-`r` subspace AND full-`d` class Gaussians) reproduce the LexMem-v5 mid-task collapse; only
REAL banked activations ground the head (latent_replay 87.3 @ same frozen k=8). The three-way at
matched budget/frozen-boundary isolates it to the *synthesis*:
| method (dil, k=8) | replay features | AA | SROIE final F1 |
| latent_replay | REAL activations | 87.3 (conv) / 64.9 (1ep) | survives |
| SpectralMemory (SLR) | synth, rank-16 subspace | 41.9 | 3.2 |
| AGLR-CL | synth, full-`d` per-class | 39.4 | 3.0 |
**Mechanism:** a per-class *marginal* Gaussian (mean+var, even full-`d`) discards the joint
feature structure real activations carry; the head trained on that marginal can't hold its
boundary against the trunk's real new-task features. This is a NOVEL, sharp addition to the
falsification chain — not just "buffer-free memory fails" but "the fabricated features must carry
joint structure a per-class Gaussian can't, and doc-IE features are too high-rank to summarise."

**RECOMMENDATION → revert to the diagnostic+falsification framing (the pre-2026-07-04-pivot
plan).** The method-led pivot was worth testing (Exp #5's 87.3 real-latent-replay looked like a
method) but the buffer-free version hits the wall the program already mapped. SLR/AGLR become the
strongest buffer-free tombstone yet, with a crisp mechanism. This is an ICML-viable *diagnostic*
paper, not a method paper — matches STATE history's original venue read (CoLLAs/TMLR/ACL-Findings
for pure-diagnostic; the migration law + this convergent negative could carry an ICML main-track
diagnostic).

**DECISION TAKEN (user, 2026-07-10): option (b) — test the coreset variant.**
Built `CoresetMemory` (`doccl/methods/coreset_memory.py`, subclasses AGLRReplay to reuse its
capture): stores **k-means centroids of REAL layer-k activations** per (task,class) — real feature
points (joint structure preserved), not a fitted marginal. 5 unit tests green incl. a check that
every replayed vector EQUALS a stored real centroid. Wired (registry/`_STD_FORWARD`/run-name/
config). **dil epochs=5 run IN FLIGHT** (`_run_coreset_c5`). This is the decisive test:
- Coreset survives (task-0/1 F1 hold, AA ≫ 42) → real feature modes DO ground the head → a genuine
  buffer-free-ish method; the failure was *synthesis*, not feature-replay per se. Pivot back toward
  a method paper (coreset = the method), run rank/centroid sweep + AGLR/latent baselines on Vast.ai.
- Coreset ALSO collapses → feature-space replay (synthetic AND real-coreset) is dead for doc-IE;
  falsification chain complete and airtight → diagnostic paper. Either way, publishable.

**Bench (dil, k=8, 5ep unless noted), AA / SROIE-final-F1 / mem:**
latent_replay(real, full) 87.3(conv)/survive/16MB · SpectralMemory 41.9/3.2/0.45MB ·
AGLR-CL 39.4/3.0/0.39MB · **CoresetMemory (real centroids) 36.7/2.9/0.48MB — ALSO COLLAPSED.**

**CORESET RESULT (2026-07-10): real centroids collapse TOO (AA 36.7, worst of the three).** So the
failure is NOT synthesis (real feature points fail identically) — my marginal-vs-joint hypothesis
was WRONG. New root-cause hypothesis, sharper: **the CARRIER design is the bottleneck.** All three
failing methods drape features onto a few (4) fixed carrier skeletons, whereas latent_replay
replays 5 WHOLE REAL documents/task. Two possible culprits, being tested:
1. **Carrier diversity**: 4 layouts/task can't represent the task (latent_replay has 5+ varied).
2. **Feature↔position decoupling** (deeper): a carrier pastes a class-`c` feature onto position `t`
   whose bbox/mask came from a DIFFERENT original token — so feature[t], bbox[t], label[t] are no
   longer mutually consistent, unlike a real replayed doc where they co-occur. This may be fatal
   regardless of carrier count.

**CARRIER-DIVERSITY TEST RESULT: still collapses.** 50 carriers + 16 centroids/class → AA 39.7,
SROIE 3.1, at 4.05 MB (8× the memory). Diversity is NOT the fix. So the carrier-scaffold approach
is dead: pasting features (real or synthetic) onto layout skeletons cannot ground the head at ANY
carrier count.

**DATA-HYGIENE (fix before write-up):** (1) the ladder table must have ALL methods at epochs=5 for
a fair compare — the `dil_latent_replay_seed42` on disk is the stale 1-epoch REFERENCE (AA 64.9),
NOT the converged 87.3 (that was a prior-session early-stop run) and NOT 5ep. Need latent_replay
@ 5ep for the honest headline; the docs=4 control gives one 5ep latent point, may also want docs=5
@ 5ep. (2) `dil_coreset_memory_seed42` (4-carrier, AA 36.7) artifact was rm'd launching carrier-50;
numbers recorded here but re-run to regenerate the artifact if needed for the paper table.
(3) `replay_memory_bytes` only persists for runs AFTER the instrumentation (spectral/aglr/coreset
have it; the old latent ref doesn't) — the 5ep latent re-run will capture it.

## ==== THREAD CLOSED (2026-07-12 pm): CoLaR r128 = LOSSLESS 2.7× — d50 accuracy at 60 MB ====

**CoLaR k4/d50/r128 (grid): AA 87.6 / BWT −1.7, row3 [89.2, 76.3, 97.2], 60.4 MB** — MATCHES raw
d50 (87.3 / −2.2 / 163 MB) on every cell; BWT slightly better. The per-doc-variance prediction
held exactly (r128 = 95.2% var → full AA; r64 = 87.7% → −6.7 AA). **Final Pareto (dil, k4):**
raw-d50 87.3@163MB · **CoLaR-r128 87.6@60MB** · CoLaR-r64 80.6@32MB · d5 76-78@16MB · naive 41@0.
The d50→d5/d0 answer: docs can't be cut (E1: selection ≈ random; E3: public-proxy d0 caps ~57);
**bytes CAN — losslessly to 2.7×, lossy dial to 5×.** Untested cheap multiplier: int8 on the
factors (~2× more → r128 @ ~30 MB). For the paper: single-seed/single-scenario — needs seeds 7/123
+ cil_cord on Vast.ai. CoLaR = the constructive leg of the consistency-law finding (whole-doc unit
compression preserves the binding; the falsified marginals are the control group).

**NOW RUNNING: lexslot-off VERIFICATION** (post-gate-fix code, grid, task `bzxun4oo3`) — decides
whether the thesis table_main 87.3 LexSlot row is real or a pre-fix task-oracle artifact.

## SUPERSEDED — E2 RESULT (2026-07-12): CoLaR r64 = AA 80.6 @ 32 MB — compression is a real but lossy dial

CoLaR k4/d50/r64 (grid): **AA 80.6 / BWT −12.3, row3 [82.3, SROIE 62.4, 97.1], 32 MB** — sits
between raw-d50 (87.3 @ 163 MB, −6.7 AA) and random-d5 (76-78 @ 16 MB, +3-5 AA at 2× bytes).
Compression loss concentrates on the mid-task (SROIE 76.7→62.4), consistent with everything else
in the program. Exactly on the per-doc-variance prediction (r64 = 87.7% var). **r128 point IN
FLIGHT** (`_run_e2_r128`, 95.2% var, ~64 MB): ≥85 → Pareto reads "d50 coverage at 2.5× less
memory, tunable to 5×"; else per-doc SVD is a shallow dial and int8-on-raw is the better lever.
The d50→d5/d0 thread's final ledger so far: selection NO (E1 77.6), d0-public NO (E3 57.3),
compression PARTIAL (E2 80.6@r64, r128 pending).

## SUPERSEDED VERDICTS (2026-07-12 am): E3 and E1 falsified their hypotheses; E2 re-running

- **E3 (PLaR@k4, d0-private): FALSIFIED depth hypothesis.** AA 57.3 [68.7, SROIE 5.9, 97.4] ≈ k8's
  58.9. k4 improves ACQUISITION (SROIE at-learning 82.0, CORD 97.4) but retention structure is
  identical → PLaR's wall is the POOL CONTENT, not interface depth. d0-at-parity via WildReceipt
  proxies is bounded ~57-59 across every knob tried (k, count, hard/soft).
- **E1 (kcenter d5, coverage-not-count): NOT SUPPORTED.** kcenter 77.6 [71.2, 64.5, 97.2] vs
  random 76.0 [72.6, 57.6, 97.9] (s42) — within random-d5's 3-seed noise (78.2±2.8). Nuance:
  SROIE retention +6.9 (64.5 vs 57.6) but FUNSD −1.4; selection redistributes, doesn't add.
  **Count (coverage MASS), not curation, drives the d5→d50 gap.** → The d50→d5 reduction lives
  or dies on E2 (compression).
- **E2 (CoLaR) crash ROOT-CAUSED (not OOM):** parent banking-log summed `d["hidden"]` over the
  whole store, but CoLaR pops "hidden"→us/v for earlier tasks → KeyError at task-1 after_task.
  Both crashes, same bug; first was misdiagnosed because chain stderr was piped through tail -2.
  Fixed (format-agnostic count) + 2-task regression test; pushed. **E2 re-running**
  (`_run_e2_colar_fix`, task bcpeq2ajd, ~3.5h): colar k4/d50/r64, grid → target ≥ ~85 @ ~32 MB.
  Note from the crashed run's log: replay loss during T1 was ~0.000-0.007 — plausibly normal
  (task-0 docs still well-classified early), but if E2's AA lands LOW with near-zero replay loss
  throughout, suspect reconstruction-too-easy/fp16-fidelity and check r128.

## SUPERSEDED (2026-07-11 pm): "reduce d50 → d5/d0" for latent replay k4 — 3-run chain

User directive: shrink k4/d50 (87.3, ~163 MB) toward d5 (78.2, ~16 MB) or d0, possibly via
lexical/latent memory. Critical constraint from our own chain: marginal memories are falsified —
only selection / compression / augmentation / public-substitution / isolation+replay doors remain.
Three hypotheses → three runs chained (task `bpmugeger`, ~8.5h, grid budget unless noted):
1. **E1 — H1 coverage-not-count:** `latent_replay` + NEW `doc_selection: kcenter` knob (greedy
   farthest-point over per-doc mean latents, pool=100; default `random` byte-identical). k4/d5
   → `dil_latent_replay_seed42_k4_kc` vs random-d5 76.0 (s42). If ≥ ~83, selection closes half
   the gap for free.
2. **E2 — H2 bytes-not-docs:** NEW method `colar` (CoLaR): per-DOC rank-r SVD of banked latents
   (whole-doc binding preserved; per-doc r64 = 87.7% var measured). k4/d50/r64 →
   `dil_colar_seed42_k4_d50` vs raw d50 87.3 at ~28 MB (below raw-d5 bytes ~16MB? factors ≈
   0.19 MB/doc × 150 = 28.4 MB — 5.7×; +int8 later → ~14 MB). Target: ≥ ~85.
3. **E3 — H3 d0-private:** PLaR@k4 relaunch (killed at final eval last night; banking had
   completed, histograms still 7/9 tags). 5ep → vs k8's 58.9. Tests the generic-interface
   coverage fix.
Both prior threads' pending runs (lexslot-off verification) queue AFTER this chain — the killed
`bldaw8zmb` chain never reached it. `docs/RESULTS_LEDGER_DIL.md` is the reference table.

**Chain incidents (2026-07-11 pm), both recovered:**
- E1 died at launch: `method.doc_selection=` isn't in the immutable latent_replay.yaml → Hydra
  struct error → fixed with NEW option file `configs/method/latent_replay_kc.yaml` (committed
  144a806); E1 re-queued behind E3 (waiter task `bp56bzp6b`).
- **E2 (colar) died ABRUPTLY** 3s after SROIE's early-stop (ep7 STOP 14:11:03; wandb EOF
  14:11:06): no Python traceback in the job log → hard SIGKILL signature (likely host OOM-killer;
  journal not readable). Task-0 leg had worked (banked 50 + **compressed to 10.7 MB vs 54.5 raw,
  5.1×** — the CoLaR mechanic is proven live). Rerun queued THIRD with full stderr capture
  (waiter `b7ddipxom` → `scratchpad/e2_full.log`, session 82a55d2e) — my chain's `tail -2` had
  swallowed the crash output (lesson: never pipe run stderr through tail in chains).
- Execution order now: **E3 (running, PLaR@k4 5ep) → E1 (kcenter d5, grid) → E2-retry (colar
  k4/d50, grid, full logs)**. Host RAM at E3-task-2: 6G used / 9G avail.

## SUPERSEDED (2026-07-11 am): deep dive on the two 87.3 rows — chain was KILLED (task bldaw8zmb)

User picked latent-replay(k4/d50) + LexSlot-hybrid as directions. Deep dive (full write-up:
`CL4IE/wiki/analyses/2026-07-11-two-87point3-rows-deep-dive.md`) found:
1. **latent k4/d50 is genuine**: final FUNSD 88.8 > at-learning 86.3 (positive backward transfer
   — the head re-carves the joint optimum); residual gap = SROIE −7 only. Dose-response d1/d5/d50
   = 61/78/87; depth k2≈k4≫k8 → generic-interface + plasticity effects confounded (k8_d50 queued).
2. **LexSlot `_off` 87.3 is UNVERIFIED**: `_off` = slot_sharing=off (task-PRIVATE slots at
   head+late, lexical inference gate, no buffer/teacher/Fisher; shared-slots variant = 42.2). BUT
   the 3 `_off` runs (Jun 28) PREDATE the gate-normalization fix 90f5a15 (Jun 30) — the
   unnormalized gate can act as a covert task oracle (multi-head DIL, inflated). The thesis
   table_main row rests on these pre-fix runs. Pre-fix dir archived (`..._off_prefix_archive`).
**Overnight chain:** (a) PLaR k4/d50/soft 5ep — does the generic k4 interface fix SROIE coverage
(vs k8's 58.9/SROIE 7)? → `dil_proxy_latent_replay_seed42_k4_d50_soft`; then (b) **lexslot-off
VERIFICATION on post-fix code**, grid budget → `dil_lexslot_seed42_off`. Morning: read both; if
lexslot-off collapses → correct the thesis row + strongest tombstone; if it reproduces → real
buffer-free positive (parameter-isolation face of the law) → multi-seed immediately.
Queued next: k8_d50, k4_d200, k4_d50 seeds 7/123, CoLaR r64 on k4/d50, slots+replay combo cell.

**Full results ledger (all dil approaches/baselines, disk-verified): `docs/RESULTS_LEDGER_DIL.md`.**
Note a labeling correction recorded there: the latent-replay 87.3 headline is the **k=4/50-doc**
run; the matched-count 5ep anchor used for this week's comparisons is k8/d4 = 63.8 (unchanged).

## ==== GATE-0 FINAL: buffer-free feature replay FALSIFIED for doc-IE (2026-07-10) ====

Complete ladder (dil, k=8, 5ep), AA / SROIE-final-F1 / mem:
| latent_replay (WHOLE real docs)      | 87.3 (conv) | survive | 16 MB |
| SpectralMemory (synth, rank-16)      | 41.9 | 3.2 | 0.45 MB |
| AGLR-CL (synth, full-`d` Gaussian)   | 39.4 | 3.0 | 0.39 MB |
| CoresetMemory (REAL cents, 4 car)    | 36.7 | 2.9 | 0.48 MB |
| CoresetMemory (REAL, 50 car/16 cent) | 39.7 | 3.1 | 4.05 MB |

**Proven (4 independent falsifications):** buffer-free feature replay fails for doc-IE, and it is
NOT (a) synthesis quality — real centroids fail like synthetic; NOT (b) rank — full-`d` fails like
rank-16; NOT (c) carrier diversity — 50 carriers fail like 4. **Only replaying WHOLE REAL
documents grounds the head.** Every collapse is the LexMem-v5 signature (current task ~93,
all prior ~3).

**Mechanism (why whole-doc replay is special) — NOW PROVEN by controlled ablation:** latent_replay
stores each doc as an intact {hidden[709], bbox, mask, labels} bundle and replays it unchanged —
feature[t], bbox[t], label[t] all from the SAME real token (consistent co-occurrence). The carrier
scaffold breaks this: it pools features across the task, then pastes a class-`c` feature onto
position `t` whose bbox/label came from a DIFFERENT token. The head trained on those inconsistent
triples learns nothing transferable.

**CONTROL RESULT (matched count = 4, the clean isolation):**
| 4 WHOLE real docs (consistent)          | AA 63.8 | row3 [60.8, 37.1, 93.6] | survives |
| 4 carriers + real centroids (decoupled) | AA 36.7 | row3 [14.5,  2.9, 92.7] | collapses |
**+27 AA / +34 SROIE-F1 from consistency ALONE** (same count, same boundary, same real features).
=> **feature↔position↔label CO-OCCURRENCE, not count/diversity/synthesis, is the necessary
ingredient.** This is the key figure for the paper — a single-variable ablation isolating the
mechanism. (Note docs=4 gives AA 63.8, below the docs=5 converged 87.3 — count/epochs still help,
but consistency is the *binary* enabler: without it the head collapses regardless.)

**This closes the method-led pivot. Verdict: diagnostic + falsification paper** (reverts to the
2026-07-04 framing, now MUCH sharper). SLR/AGLR/Coreset extend the existing chain
(`docs/FINDINGS_ANALYSIS_PAPER_2026-07.md`) with a new terminal level: "even a real-activation
coreset — the strongest possible buffer-free feature memory — fails; only whole-document replay
works, because the head needs consistent (feature, position, label) co-occurrence that no
per-class/per-token summary preserves." Novel, mechanistic, airtight.

**Next action:** DECIDE (user) — run the rank sweep on Vast.ai (the local box is too slow, ~90
min/run), OR accept the negative result and fold SLR into the falsification chain. If sweeping,
also run AGLR-CL (does class-first full-d Gaussian beat basis-first rank-r? if AGLR works and SLR
doesn't, that's informative about what the head needs). See wiki idea page Gate 0/1.

---

## Experiment #6 RESULT — the law BENDS at parameter granularity, doesn't break (2026-07-07)

Seed-42 dil, masks verified at exact target fractions (50.0/80.0/95.0% of 125.3M):

| arm | AA | BWT | final row [t0/t1/t2] |
|-----|----|-----|----------------------|
| p=0.50 | 45.0 | −66.0 | 30.6 / 6.9 / 97.5 |
| p=0.80 | 49.6 | −59.0 | 44.5 / 6.2 / 97.9 |
| p=0.95 | 54.8 | −46.3 | 62.4 / 5.8 / 96.0 |

Reading vs pre-registered gates: neither strict conservation (54.8 > naive+3) nor a
real opening (< v3b 66.1) — the "interesting" band. Refinements:
1. GRANULARITY MODULATES THE LAW: layer-level protection conserves (freeze≈free,
   exp #4); per-tensor interleaved protection buys up to +13.5 AA over naive with
   plasticity fully intact (new-task 96–98 even at p=95). Hard masking also beats
   soft penalty at the same importance signal (fisher_mask 54.8 vs EWC 41.2).
2. PROTECTION IS TASK-SELECTIVE, NOT GENERAL: task-0 (forms, lexically distinct)
   retention rises monotonically with p (31→45→62), but the mid task (receipts,
   competing with task 2 for the same parameters) collapses to ~6 at EVERY p — the
   drift signature survives protection wherever tasks compete for capacity.
   Same scenario-boundedness pattern as lexical routing (works across domain gaps,
   fails within them).
3. CURVE COMPLETE (2026-07-07 15:55, both exit 0) — 7 points, seed 42:

   | p | 0 (naive) | 0.50 | 0.80 | 0.90 | 0.95 | 0.99 | 1.0 (freeze-all) |
   |---|-----------|------|------|------|------|------|------------------|
   | AA | 41.3 | 45.0 | 49.6 | 51.1 | **54.8** | 52.5 | 41.1 |

   Peak at p=0.95. At p=0.99 plasticity finally degrades (new-task 89.5 vs 96–98
   elsewhere) — the turn toward the freeze-all collapse. Mid-task stays 5–9 at
   EVERY p: within-domain capacity competition is never rescued by protection.

Verdict for the paper: the lottery-ticket family gets a diagnosed ceiling — masking
protects only lexically-separable tasks and never recovers within-domain competition;
still 27 pts below ER-5. Consistent with Finding 3: no protection scheme substitutes
for real past-task gradients.

## Experiment #6 launch record (2026-07-07)

The migration law was established at LAYER granularity (exp #4); this closes the
granularity attack surface ("maybe fine-grained interleaved masks share capacity where
whole-layer freezing cannot" — the PackNet/HAT/WSN lottery-ticket argument). Method
`fisher_mask` (commit 37ff257): per-tensor top-p Fisher-important entries hard-frozen
(exact freeze: grad mask + snapshot/restore across AdamW step), rest plastic, mask
rebuilt after every task from the running Fisher sum (same estimator as EWC).

**PRE-REGISTERED READINGS (set before any result):**
- Conservation holds at every p (AA ≈ naive 41.3 ± ~3, or pure-stability collapse at
  high p) → law is GRANULARITY-INDEPENDENT — stronger lead result, and explains away
  the PackNet/WSN family for task-ID-free doc-IE.
- Some p gives AA well above floor with real retention (≥ ~50 interesting, ≥ 66 = v3b
  → real opening) → law breaks at fine grain → mask-based method justified by mechanism.

3 arms queued behind exp #5 chain (bg task bi1cmr7r8, polls for free GPU):
`dil_fisher_mask_seed42` (p=0.8 canonical) / `_p50` / `_p95`.
Decided vs alternatives (2026-07-07 session): directional optimization (GPM-family)
already falsified in-house (HGT/CUBER); RanPAC premise contradicted by frozen-encoder
45.3 arm but worth one baseline run later; activation-based masking rejected in favor
of Fisher to keep one importance language.

## Experiment #5 RESULT — P1: THE MIGRATION LAW IS BOUNDED (read 2026-07-07)

Seed-42 dil results against the pre-registered gates (set before results existed):

| arm | AA | BWT | final row [funsd/receipts-1/receipts-2] | reading |
|-----|----|-----|------------------------------------------|---------|
| k=4 (8 plastic) | **76.0** | −19.3 | 72.6 / **57.6** / 97.9 | **P1** (gate ≥75) |
| k=8 (4 plastic) | 65.8 | −30.8 | 67.9 / 34.5 / 94.9 | ≈ v3b (66.1) |
| ctrl (freeze-only k=8) | 37.3 | −74.1 | 15.2 / 2.4 / 94.4 | conservation confirmed |

Reading: real past-task gradients into the plastic remainder BREAK conservation, and
monotonically in plastic depth (76.0 @ 8 plastic > 65.8 @ 4 plastic; control = law).
The mid-task drift signature (~13–17 for every buffer-free method) is broken: 57.6.
Law statement upgrades to: "forgetting is conserved under architectural constraint
*unless* the plastic remainder receives real past-task gradients — and raw inputs are
not the only sufficient carrier: frozen-space activations (no raw documents stored)
recover most of the replay effect (76.0 vs ER-5 82.1 vs best buffer-free 66.1)."
The 2×2 (real gradients × feature validity) is complete: it is the GRADIENTS, not the
storage format, that matter — but the features must not be stale.

Extension COMPLETE (2026-07-07 13:27, all exit 0):

| run | AA | BWT | final [t0/t1/t2] | note |
|-----|----|-----|------------------|------|
| k=4, 50 docs/task | **87.3** | **−2.3** | 88.8 / 76.7 / 96.6 | ≈ ER-200 (87.9), 1.4 below joint (88.7); BEATS raw ER-50 (85.6) at equal doc count |
| k=2, 5 docs | 79.7 | −13.9 | 73.1 / 68.5 / 97.4 | depth dose–response still monotone: 65.8 (k8) → 76.0 (k4) → 79.7 (k2) |
| k=4, 5 docs, seed 7 | 77.3 | −17.4 | 81.0 / 53.3 / 97.4 | |
| k=4, 5 docs, seed 123 | 81.4 | −11.7 | 79.4 / 67.9 / 97.0 | |

Headline k=4/5-docs across seeds: **78.2 ± 2.8** (n=3) — all seeds clear the P1 gate (75).
Doc-count crossover vs raw ER at k=4: 1 doc 61.3 vs 57.6 (latent wins), 5 docs 76.0–81.4
vs 82.1 (ER slightly ahead), 50 docs 87.3 vs 85.6 (latent wins). At scale, activation
replay on a drift-immunized trunk ≈ joint oracle WITHOUT raw documents. Honest caveats
for the paper: activation storage ≈1.1 MB/doc (bytes larger than raw docs — claim is
"no raw documents stored", not compression) and activations are partially invertible.

## Experiment #5 launch record (2026-07-07)

Chain level 6: frozen-trunk activation replay (Pellegrini-style), the untested cell of
the 2x2 (real gradients x feature validity). Freeze layers <k after task 0; bank
layer-k hidden (fp16) for 5 docs/task; replay via pre-hook injection → real past-task
gradients into exactly the plastic layers the migration law says absorb the drift.
Method `latent_replay` (commit 93aa144), 13 tests green, smoke OK (single_funsd 70.5@1ep).

**PRE-REGISTERED GATES (set before any result existed):**
- P1 success: dil AA ≥ 75 → law bounded ("conserved *unless* plastic part gets real
  past-task gradients"); raw-data-free method on par with ER exists.
- P2 falsified: AA ≤ 68 (v3b 66.1+2σ) → chain closes at level 6 with predicted mechanism.
- P3 68–75: boundary result; dose-response on k tells where it breaks.
- Signature to watch: mid-task row (buffer-free collapses ~67→~17; ER holds ~67+).

3 sequential runs on local box (bg task b8rnl3zyp, logs in scratchpad/latent_runs/):
1. `dil_latent_replay_seed42` (k=8, 4 plastic — matches map8) — primary
2. `dil_latent_replay_seed42_k4` (8 plastic) — dose-response
3. `dil_latent_replay_seed42_ctrl` (freeze-only, no replay) — control
References (no new runs): naive 41.3 / v3b 66.1±3.0 / ER-5 82.1 / joint 88.7.
If P1/P3: extend best k × docs {1,50} + seeds 7,123. Spec + rationale:
`~/.claude/plans/1-yes-my-motivation-foamy-boot.md`. bd still degraded — issue NOT
filed in bd; this section is the tracking record.

## Experiment #4 COMPLETE — migration robustness (2026-07-05). PROGRAM DONE -> WRITING

8 runs (lexmem_ctrl freeze vs free, seed42, drift-probed):
| arm | AA | BWT | t0-final | t0-CKA trace |
| o2_freeze / o2_free | 46.6 / 47.6 | -63 / -62 | 4.3 / 4.7 | .08,.22 / .14,.25 |
| o3_freeze / o3_free | 30.6 / 32.0 | -86 / -85 | 5.8 / 7.9 | .40,.19 / .48,.20 |
| map8 (4 plastic) / map1 (11 plastic) | 40.8 / 39.7 | -73 / -75 | 20.7 / 17.7 | ~.24 endpoint both |
| long_freeze / long_free (10-session CIL) | 9.3 / 9.5 | -83 / -80 | 0 / 0 | ~.1-.2 across 9 boundaries |

**LAW (strongest form): total forgetting is CONSERVED under architectural
constraint — freeze-vs-free AA within 1-2 pts in every order, map, and horizon;
only the locus moves.** Freeze arms prove localization structurally (head+late
frozen => endpoint CKA collapse can only originate early/mid). Dose-response:
even 4 plastic layers absorb the full drift (map8 ~= map1). Long horizon: the
effect compounds to t0-F1=0 over 10 boundaries regardless of map.

All 4 program experiments done: oracle (55.3 low branch) / gauss-replay
falsified (63.6) / retention curve (5 raw docs 82 > all buffer-free) /
migration law. results/migration/*. NEXT: thesis/paper writing.
## SESSION HANDOFF (2026-07-04 close)

- **Done this session:** v5 graph-as-memory gate run to completion → FALSIFIED
  (both arms AA 63/SROIE 13, byte-identical). Found+fixed the val-restore no-op bug.
  Then the KEY pivot (user): v5's mechanism = solved literature (FeCAM/FeTrIL/PASS) →
  **reframed the whole program as a diagnostic+falsification paper, v5 = tombstone.**
  Docs `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` + STATE updated & pushed (7514cf3).
- **Quality gates:** 287 fast tests green, ruff clean on v5+train.py.
- **NEXT SESSION = WRITE THE PAPER, not more methods.** Lead = migration finding +
  convergent negative result. Concrete next steps (cheapest first, per program review):
  (1) head-refit probe oracle already done (AA 55.3 — representation forgetting is REAL);
  (2) exp #4 migration robustness (2 freeze maps × 2 task orders + cil_cord_long) — the
  one remaining experiment; (3) relabel thesis "LexSlot 87.3" row (it's the DocCL hybrid).
- **KNOWN BLOCKER:** `bd` is in a degraded re-clone-warning state (stray dolt server
  killed but local DB still complaining). Paper-reframe follow-up issue NOT filed in bd —
  captured here + in FINDINGS doc instead. Fix bd (or `bd export`/re-clone carefully)
  before relying on it next session. Don't block on it.

## FRAMING LOCKED: diagnostic+falsification paper, NOT a method paper (2026-07-04)

- **Decision (user):** v5 (feature-Gaussian graph replay) is NOT a contribution.
  Its core mechanism = PASS/FeTrIL/FeCAM (solved 2021-2023) → un-submittable as novelty.
  v5 enters the paper ONLY as the **terminal tombstone** of the falsification chain:
  even the field's strongest buffer-free tool (+ a novel relational variant) can't beat
  representation drift (edges-on AA 63.2, SROIE→13.0).
- **The paper's actual novelty:** (1) the diagnosis — forgetting is head-localized,
  architecture-general, and the locus MIGRATES under naive protection; (2) a rigorous
  convergent negative result: FIVE buffer-free families fail for one identified reason
  (representation drift), so "only replay grounds the head." NOT any method.
- **Venue:** CoLLAs/TMLR/ACL-Findings for the negative result; AAAI/ICML main-track only
  if the LEAD is the migration finding, never a method.
- **Docs updated:** `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` (v5 = falsification level 4;
  "is NOT a method paper" section). v5 code stays (evidence), no further method work on it.
- **v5 gate COMPLETE (both arms):** edges-on = edges-off = AA 63.17, SROIE→13.0,
  BYTE-IDENTICAL (ΔAA exactly 0.00) despite differing recon-loss streams (1.42 vs 0.013).
  → do NOT claim an edges ablation; the head-side signal is swamped by trunk drift (recon
  grad reaches head but doesn't move the evaluated model). Uninformative ablation is itself
  consistent with Finding 3 (bottleneck = trunk representation drift, not head). v5 = tombstone.
  Runs: results/dil_lexmem_v5_seed42{,_bank}. Earlier byte-identical pair (pre-fix, from
  the val-restore bug) archived at results/_invalid_v5_restore_bug/.

## Analysis program: experiments 1-3 COMPLETE (2026-07-03)

**#1 Head-refit oracle (LOW branch):** pooled probe on naive trunk AA 55.3
(FUNSD per-task probe caps at 41.9). Representation forgetting is REAL in
doc-IE — Davari's "representations survive" does NOT transfer. All head-only
methods on a naive trunk ceiling-bounded at ~55. results/head_refit_oracle.json

**#2 Gaussian head-replay (FALSIFIED, 4th chain level):** AA 63.6 ≈ v3b 66.1±3.0.
Failure signature IDENTICAL across head mechanisms (slots vs replay: SROIE
~67→~17 despite balanced head gradients) => mid-task bottleneck is per-task
REPRESENTATION drift, not head misalignment. results/dil_gauss_replay_seed42

**#3 Retention-information curve (dil seed42):**
| stored | AA |
| naive (nothing) | 41.3 |
| input counts (ledger) | 33.8 |
| naive-trunk probe ceiling | 55.3 |
| ER 1 doc/task | 57.6 |
| feature Gaussians | 63.6 |
| slot memory (v3b, 3 seeds) | 66.1 ± 3.0 |
| **ER 5 docs/task** | **82.1** |
| ER 10 docs/task | 84.8 |
| ER 50/task | 85.6 | ER-200 87.9 | joint 88.7 |
=> FIVE raw documents per task beat every buffer-free memory we built by 16 AA.
Sharpened claim: raw inputs are the only storage that survives representation
drift (encoder re-renders current features); all derived quantities go stale.
Runs: results/retention_curve/er_buf{3,15,30,150}.

**Remaining: #4 migration robustness** (2 freeze maps x 2 task orders +
cil_cord_long) — needs dil task-order scenario variants. Then thesis writing.
v3b 3-seed: AA 66.1±3.0. gauss_replay code committed (bec95b4).

## LexSlot-FM: FM refit bug FIXED — AA +5.25/+5.68 on both gate modes (2026-07-03)

- **Bug found + fixed:** `_refit_old_fm_slots` ran `self.model.eval()` → forward hook
  used `_replacement_blend` (gate-routed, uniform blend) instead of `_additive_blend`
  (single-slot, `_cur_fm_idx`). Gradients dispersed to all FM slots + base head, never
  reaching only the target slot. Fixed by switching to `self.model.train()` so
  `_additive_blend` fires with `_cur_fm_idx` set to the old slot.
- **Results comparison (seed=7, dil FUNSD→SROIE→CORD):**

  | Variant | AA | T0 | T1 | T2 |
  |---------|-------|-------|-------|-------|
  | Frozen encoder (s=42, no refit) | 45.31 | 20.50 | 46.79 | 68.64 |
  | Layout+EWC (no refit) | 44.09 | 31.12 | 27.59 | 73.56 |
  | Layout+EWC+refit (**FIXED**) | **48.67** | **40.78** | 20.35 | 84.87 |
  | TF-IDF+EWC (no refit) | 39.05 | 19.53 | 8.80 | 88.81 |
  | TF-IDF+EWC+refit (**FIXED**) | **44.73** | **26.10** | **19.45** | 88.64 |

  **Key findings:**
  1. FM refit (train-mode fix) improves both gates — Layout AA +5.25, TF-IDF AA +5.68
  2. T0 jumps +11.09 (Layout) and +6.57 (TF-IDF) — FM_0 refit now actually works
  3. T2 stays high on both gates: 84.87 (Layout) / 88.64 (TF-IDF)
  4. T1 (SROIE) still degraded by encoder drift — 50-sample FM_1 replay buffer insufficient
  5. **Best AA so far: 48.67** (Layout gate + EWC λ=1000 + FM refit 3ep/50samples)
- **Remaining issue:** T1 collapses from 70.98 (post-training) to 20.35 (after CORD training).
  FM_1 refit helps (+4.82 vs buggy before fix) but the 50-sample buffer can't fully recover
  SROIE features after the encoder shifts during CORD training. Possible fixes: larger replay
  buffer (200+ samples), more refit epochs, or adaptive refit learning rate.
- **Files changed:** `scripts/train.py` (added `gate_mode` suffix to lexslot_fm run names);
  `doccl/methods/lexslot_fm.py` (train mode in refit). bd: **CLUE-85g**.
- **Blocked on:** user decision for next direction (increase refit budget, conditional gate,
  more seeds, or pivot).

## Ledger gate falsified standalone; strategy fork OPEN (2026-07-02 23:30)

- Ledger gate v2 (relational context keys): CORD OWN 76.1 (!) but cumulative
  FUNSD 21 / SROIE 23 << 60 bar; key-collision interference -18 on CORD.
  Memory-as-retention falsified at all 3 levels (parametric / +drift-ctrl /
  input-keyed). Script: scripts/lexical_ledger_gate.py (commit 12dfba2);
  results/ledger_gate{_v1_unigram,}.json.
- **USER DECISION PENDING** (asked, away): (a) analysis-paper + v3b spine
  [recommended], (b) push v3b vs replay, (c) xlingual ledger gate.
- Meanwhile: v3b seeds 7+123 running overnight (serves all forks).


## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Older history
Superseded/completed sections (June + early-July: LexMem pilot, baseline porting,
backbone-agnostic audit, HGT/DocMERGE/LCA/HRP branches, bug sagas) moved to
`STATE_ARCHIVE.md` (2026-07-04) to keep this ledger lean. All DONE/FALSIFIED — kept for provenance.
