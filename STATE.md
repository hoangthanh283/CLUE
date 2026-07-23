# STATE

## ACTIVE (2026-07-18): Thesis reframe underway + B+ grid audited & ready (user gate: rent box)

**(b) Method chapter:** decision executed — chapter becomes *diagnostic-driven design +
pre-registered falsification, CoLaR as constructive control*. Design + drafted prose +
examiner annotations: `docs/THESIS_METHOD_CHAPTER_REFRAME_2026-07.md` (scout found the
LexSlot-as-headline framing threads through title page + ch1/ch3/ch6/ch7 — reframe is
thesis-wide). **DONE now: the ~25-site "buffer-free"→"small-buffer (200 exemplars)"
correctness pass across ch1/3/6/7** (the 87.3 row is the DocCL-hybrid w/ 200 exemplars;
per-seed 88.4/87.2/86.3 now cited; grep-clean; thesis builds 125 pp, no new errors).
REMAINING (session task #5): the structural rewrite per the design doc — resolve every
[EXAMINER] annotation first (esp. the frozen-trunk-causality overclaim in one outline
cell). Note: the "stale early-stopping prose" memory was itself stale — ch4/5/6 already
describe the convergence protocol correctly.
**(c) B+ grid:** audited plan `docs/BPLUS_GRID_PLAN_2026-07.md` — 355 missing cells
(LiLT 103, BROS 126 = zero coverage, BERT 117, LM3-lexslot 9), ≈293 GPU-h ≈ $88–147.
Empty-override bug (`${VAR:-}` on CORE/PROMPT/CURRENCY_METHODS) FIXED in
run_grid_multigpu.sh + setup_remote.sh, DRY_RUN-verified. **Blocked on user renting the
Vast.ai box (task #6); commands are paste-ready.**

**CoLaR-WSVD candidate (2026-07-18):** implemented `method=colar_wsvd` as a strict CoLaR
ablation: same per-document factors and bytes, entity-token rows weighted in the SVD
objective. DIL seed42 k4/d50/r128 result = **87.08 AA / 60.4 MB**, final row
85.90/78.13/97.21. This is below CoLaR 87.58 (89.19/76.30/97.25) and near CoLaR-Bal
87.11. Read: weighting buys SROIE (+1.84 micro; KEY +1.50, VALUE +1.80) but steals FUNSD
retention (−3.30), so it is a negative ablation, not a new headline. Do not spend more on
row reweighting unless paired with a constraint that prevents task-0 capacity theft.

## DONE (2026-07-17 eve): KILL-TESTS COMPLETE + ADJUDICATED — all corrections are nulls; re-exercise is the only lever

**All 5 kill-tests ran and were adjudicated against the amended prereg, every verdict
adversarially verified (5 independent recomputation agents; 2 of my draft verdicts were
corrected by them — full detail in `docs/RCA_FORGETTING_BASELINES_2026-07.md` kill-test
sections):**
- **Readout suite: KILLED as tested, family verdict "deferred — information destroyed
  pre-correction"** (best pooled recovery −0.1 AA; saturation: KEY mass ~7e-6, 2–3 orders
  below the 1e-3 branch threshold). Per-doc EM headline is dead; pre-written fallback
  framing applies. Scope note: funsd-KEY partially recovers (0→16.4) under marginal_match
  but pooled worsens (sroie degenerates to all-O).
- **Frozen-trunk: NOT CONFIRMED as clean causality** (3 of 4 conjunctive clauses fail;
  O-collapse is LESS under frozen trunk (13.3% vs 0.0%) = the escape-hatch direction;
  acquisition-invalid: cord at-learning 68.4). Old-class extinction DOES reproduce with
  trunk fixed; effect-size match to naive is confounded.
- **marginal_kl: provisional NULL** (KEY/HEADER 0.0 → H2 terminal case holds; H4′-direction
  movement negligible: VALUE +8.5, O-recall 0→3.1%; AA 39.1 below sanity window — open:
  KL/CE scale at λ=1, batch-2 MC noise; anchor code verified correct).
- **logit_adjust: no buffer-free signal at τ=1** (survival bar 0.0 vs >20; acquisition
  guard PASSES — at-learning within 5 of naive's actual 87.8/82.5/96.1; prereg's stale
  hard-coded reference corrected in-place with a dated note).
- **Consolidated (n=1, provisional): the snap is a SYMPTOM, not an invertible mechanism.**
  The head collapses predicted mass on under-exercised classes to ~1e-5 (prediction-mass
  statistic; weight geometry unmeasured); no eval-time reweighting recovers it in
  aggregate, marginal-level training objectives don't prevent it, extinction reproduces
  with frozen trunk, features survive (head-refit 55.3). **Only task-consistent
  re-exercise (replay) prevents it.** Paper gain: the "did you try cheap recalibration"
  hole is closed with a REGISTERED NULL.
**Next: fold into thesis/paper (falsification chapter + method-chapter reframe per the
pre-written fallback), then the B+ grid critical path. The method-chapter bet: per-doc EM
is dead as headline; honest options = negative-result chapter around the kill-test suite
+ CoLaR as constructive control, or a replay-adjacent evidence-store method (the only
open readout lever).**

## DONE (2026-07-17 pm): Kill-tests queued — brainstorm done, top-5 implemented, DOUBLY REVIEWED

**Method brainstorm (judged workflow: 4 lenses → novelty audit → 2 adversarial judges →
synthesis) produced a top-5 shortlist; all implemented + PRE-REGISTERED
(`docs/RCA_KILLTESTS_PREREG_2026-07.md`, commit 6d3f019) and queued as
`scripts/run_rca_killtests.sh` behind the read-side chain (~3.5 GPU-h):**
1. **Readout suite** (`rca_readout_fixes.py`): naive retrain + logit dump, then
   marginal-match oracle / prior-ratio one-step / per-doc Saerens–Latinne EM (task-ID-FREE
   — a doc's ~512 tokens are its own calibration batch). SUPPORT bar: pooled AA AND
   AA_old(funsd,sroie) both >15 pts recovery; cord non-regression ≤2; KILL family if <5
   pooled — split by the saturation check (KEY/HEADER prob mass ≥/< 1e-3 distinguishes
   "mechanism dead" from "information destroyed pre-correction").
2. **Frozen-trunk naive** (`--freeze-trunk`): pure-head causality; CONFIRM if extinction/
   O-collapse/snap-cos match full naive ±5 F1 / cos 0.02.
3. **marginal_kl** (KL to PRIOR-task mixture, 9 floats/task): H4′-vs-H2 disentangler;
   adjudicable rule = KEY/HEADER >20 refutes H2 terminal case; AA 45–55 is sanity only.
4. **logit_adjust** (balanced softmax under cumulative prior, buffer-free): survival bar
   KEY/HEADER >20; MUST check at-learning F1 (EWC-style acquisition regression voids it).
Killed by judges: OT recalibration (no mixed eval stream), DoLa layer-contrast (head is the
damaged locus), head-only replay (decoupled-carrier risk — kill reasoning flagged as weak,
revisit next brainstorm). Gated: Bayesian last-layer behind a 2-line head-only-EWC
ablation; NCM head waits for colar_knn results.
Outputs land in `results/rca/killtests/` + `results/rca/dil_{naive_frozen,marginal_kl,logit_adjust}_seed42_rca.json`
(rca_synthesize.py picks them up). Adjudicate strictly by the prereg doc.

**Adversarial reviews BOTH DONE before any kill-test executed (commits f79ac0b + afd26e9):**
- **Plan red-team** (4 attack angles → adjudicator): survives-with-amendments. Caught:
  marginal_kl mixture included the CURRENT task (self-anchor no-op at task 0, 1/(t+1)
  dilution — CRITICAL, silently changed what rule 3 tests; fixed to prior-only); KILL-rule
  blind spot (added saturation split); rule-1 CORD dilution (added AA_old + guard); AA
  45–55 demoted to sanity check; novelty framing corrected (logit_adjust = Menon/Ren
  mechanism in a new fixed-head-DIL regime; per_doc_em = Saerens–Latinne 2002 at document
  granularity — delta is the calibration unit, closest prior TTLSA arXiv:2211.15646);
  per-doc-EM terminal-case fallback framing PRE-WRITTEN in the prereg. Stale premise
  retired: **colar_knn λ=1.0 landed at AA 46.24 / BWT −6.51 — far below CoLaR 87.6; a pure
  kNN head is NOT a hedge** (λ=0.3 + colar_meta still pending).
- **Code review** (4 dimensions, findings adversarially verified): 2 real bugs beyond the
  red-team overlap — **per_doc_em zero-prior fixed point** (π init from CORD marginal with
  exact-zero KEY/HEADER locked those classes at 0 forever regardless of document evidence;
  reproduced in fp64; Laplace-smoothed + regression test) and **no self-duplication guard
  on chain scripts** (flock added to run_rca_killtests.sh + run_readside_gate01.sh via
  mv-replace so running instances keep their old inode). Earlier fp16 hazards (logit dump
  flooring extinct probs; KL underflow-NaN under Turing fp16 autocast) fixed in f79ac0b.
⚠ A duplicate parked `run_readside_gate01.sh` instance exists (kill was permission-blocked);
flock now prevents future duplicates, but the live pair predates it — if two train.py
appear simultaneously, kill the newer.

**KILL-TESTS EXECUTION STATUS (2026-07-17 15:20): chain RUNNING** (readout-suite naive
retrain live on GPU) after a ~5 h deadlock, root-caused and fixed:
- **Postmortem:** a leftover launcher shell's cmdline contained the strings both chains'
  `pgrep -f` guards watch (`scripts/train.py`, `run_readside_gate01`) → the duplicate
  read-side instance never exited, the kill-tests chain waited on it. Killed the zombie;
  chains flowed immediately. **Rule going forward: never echo watched script names into
  long-lived shells; prefer flock guards (already added to both chain scripts) over
  pgrep-cmdline matching for NEW scripts.**
- **m3 output-dir mismatch:** train.py saved colar_meta m3 to the run-name dir (no `_m3`
  suffix — m3 is the config default) while the chain's resume-check watched the hydra dir.
  Verified artifacts copied to `results/dil_colar_meta_seed42_d50_r128_m3/` (train.log in
  that dir proves provenance). Follow-up filed (session task #4; move to bd when unblocked).
- **Monitoring now stall-proof:** completion watcher (fires on COMPLETE marker or chain
  death) + stall detector (fires if chain alive but chain.log frozen > 60 min). Session
  task list tracks: monitor → adjudicate (strictly per amended prereg) → write results +
  push.

**Frozen-trunk kill-test IN (2026-07-17 pm):** old-class extinction repeats (FUNSD
HEADER/KEY = 0.0, SROIE KEY = 0.0), but the preregistered match fails: final AA 28.3 vs
full naive 37.8, driven by poor acquisition under a frozen pretrained trunk (diag
41.4/47.7/68.4). Verdict: acquisition-invalid for pure-head causality; useful only as
"extinction can happen with fixed trunk," not a clean effect-size confirmation.

**Readout kill-test result IN (2026-07-17 pm):** retrained naive smoke-passed (AA
39.78). Eval-time marginal repair is a clean negative: marginal-match/prior-ratio/per-doc
EM give AA 38.58/39.70/39.60 vs uncorrected 39.78; AA_old stays ~11. Saturation split: old
KEY/HEADER prob mass is <1e-3 (~4e-6--7e-6 KEY, <7e-7 HEADER), so the head has already
annihilated recoverable mass. Test-time readout correction is closed; next viable lever
must prevent extinction during training or re-exercise old classes.

**Deep RCA extension implemented (2026-07-17 pm):** `scripts/rca_deep_dive.py` adds the
paper-facing CPU consolidation pass over Tier-B JSONs and writes `results/rca/d_*.csv` +
`d_summary.md`. Focused test added in `tests/test_rca_aggregation.py`. Current summary
keeps the RCA verdict intact: failing methods concentrate final old-task extinctions and
the strongest cells snap to the just-trained marginal; modality-mask table remains a
quantification layer, not a new mechanism.

**READ-SIDE GATE 1+2 RESULTS (2026-07-17, adjudicated against the pre-written bar):**
- **colar_meta m3 = AA 79.48, BWT −5.62, row [84.7, 65.2, 88.6]** vs m1 control 85.85 and
  CoLaR 87.6: metaplastic consolidation is dose-dependently harmful (m1 −1.75 → m3 −8.1).
  **RCA prediction (colar_meta fails) CONFIRMED.** Read-side direction fully closed:
  both tracks (kNN blend, metaplasticity) are clean negatives.
- **colar_knn λ=0.3: AA 87.64, row [89.2, 76.5, 97.2]** vs λ=0 CoLaR 87.6 [89.2, 76.3, 97.2].
  Formal bar (AA ≥ 87.6, FUNSD not below 89.2, SROIE > 76.3) is met — at the boundary:
  +0.04 AA, +0.2 SROIE, n=1 seed. The ROADMAP's "beats λ=0 by REAL margin" clause FAILS.
  **Verdict: NO SIGNAL** — the kNN blend is inert at λ=0.3 and destructive at λ=1.0
  (AA 46.24; pure kNN head ≈ gate0 probe quality 44–68 F1, far below the trained head).
  **Do NOT build R3 (colar_mbpa).** Mechanistic read: blending in a weaker reader cannot
  fix the snap — consistent with the RCA (the fix must repair the readout, not average it
  with a noisier one).
- **colar_meta m1 control: AA 85.85 [87.7, 75.4, 94.4]** — fails to reproduce CoLaR
  (−1.75 AA): the metaplastic machinery costs performance even at m=1. m3 (training now)
  is therefore ambiguous-by-construction; per the pre-registered control rule, read m3
  only as consistent/inconsistent with the RCA's colar_meta-will-fail prediction, not as
  a clean effect size.
- gate0 probe (pretrained-trunk lower bound): best per-task kNN F1 funsd 55.8 / sroie 67.8
  (k8_top20/top5 softmax) — passed the ≥20 go bar, but the ceiling it implies explains the
  λ=1.0 collapse.

## DONE (2026-07-17 am): RCA COMPLETE — readout-marginal snap is the root cause; read-side chain running

**Tier B landed + Tier C adjudicated, then AMENDED after a 16-agent adversarial verification
pass** (`docs/RCA_FORGETTING_BASELINES_2026-07.md`; every load-bearing number independently
recomputed): **H2 SUPPORTED** (KEY extinction mask-uniform — naive/lwf KEY F1 = 0.0 under EVERY
valid mask; replay stratum clean, er KEY 85.8/91.3), **H4 SUPPORTED AS AMENDED (H4′ marginal
snap)** — the as-registered falsifier had a drafting flaw (O is both SROIE's dominant label,
84.8% not 99%, and the falsifier target); the decisive amended test: the head's output marginal
on old tasks snaps to the just-trained task's gold marginal (cos 0.997/0.998 naive/lwf at b1 vs
0.60 to own gold; replay stays 0.99–1.00 to own gold; `results/rca/c_marginal_snap.csv`).
**H3 REFUTED** (trunk displacement FRONT-loaded: naive early 0.70 vs late 0.03), **H1 REFUTED
as mechanism** (raw spread is a baseline-floor artifact — normalized retention spread 4.7pp for
naive = lockstep annihilation; text embed moves LEAST 5/6; layernorm moves most, 262–10,055×
text embed), **QA NOT SUPPORTED** (no receipts-overlap signature).
New verified findings the snap explains: **O-collapse** (naive/lwf O-row acc exactly 0.00% at
final — invisible to seqeval; O dies at b2 not b1 because the b1 snap direction WAS O);
+7.9 "recovery" = snap rotation, not healing; **EWC protection is task-0-biased** (funsd resists,
sroie snaps 0.986 — and ewc's sroie at-learning was crippled 41.9 vs 82.5: acquisition failure);
head/trunk displacement 8–500× (ewc ~1× = Finding-2 migration, still forgets 43 pts).
**Root cause: readout-marginal snap on a class-asymmetric substrate.** Coheres with F1,
head-refit oracle, F3. Implications: attack the readout (read-side premise re-derived
independently); marginal/logit-prior recalibration = mandatory cheap baseline; targeted
minority-class replay (but SROIE's 3.7% KEY share was NOT enough — re-exercise must hit the old
distribution); colar_meta predicted to FAIL (its queued run = the falsification test).
Caveats: Tier B n=1 seed; colar row = canonical k8/d5/r64 not headline.
**3-seed conservation verdict IN:** base 87.8 = colar_bal 87.8 > kcenter 87.0; no lever beats base
at any seed; redistribution softer than seed-42 suggested (bal: SROIE +1.8 for FUNSD −0.5).
**Read-side chain relaunched** (Gate 0 probe → colar_knn λ runs → colar_meta m1/m3),
log `results/readside_gate01.log`.

## DONE (2026-07-16 eve): Deep forgetting RCA of baselines (Tier A DONE — two findings; Tier B queued)

**User pivot:** before more method design, RCA the baselines' forgetting patterns + the
multimodal correlation. Plan: `/home/thanh/.claude/plans/let-s-find-a-way-humming-flurry.md`.

**Tier A (artifact mining, commits b688682) — findings from 223 per-class runs + 308 matrices:**
1. **M5/Q6 ANSWERED — ER is NOT label-degenerate** (reviewer's suspicion refuted): er KEY-F1
   87.2 ≈ AA 86.8; er_cflat KEY 89.3. The degeneracy signature (KEY dead, VALUE carries AA)
   instead marks the FAILING methods: naive & lwf & the whole falsified buffer-free chain show
   **TOTAL KEY+HEADER extinction (F1=0.0)** while VALUE survives ~47. Forgetting is
   class-asymmetric; aggregate AA hides it. LwF protects nothing (KEY 0.0 despite distillation).
   → `results/rca/a1_*.csv`, `a1_summary.md`.
2. **Forgetting is one-boundary collapse, not decay:** naive −75.6 F1 at the first boundary
   then +7.9 RECOVERY; family signatures differ (penalty dampens ×4 to −19.5; replay flattens
   to −1..−4; prompts: small immediate but ongoing leak — immediate_share <0.6).
   → `results/rca/a2_*.csv`, `a2_signature.png`.
3. **A3 null confirmed mechanically:** 0/196 tb dirs have displacement scalars
   (tensorboard.diagnostics never enabled) → per-baseline locus requires Tier B. → `a3_verdict.md`.

**Tier B (commit 2b5645f, PROMOTED 2026-07-16 eve** — read-side gate chain KILLED before it
produced anything; queue is now seed sweep → RCA chain, log `results/rca/chain.log`): core-6
{naive, ewc, lwf, er, der_pp, colar} dil seed42, per-boundary probes: modality-ablated eval
(train FULL, eval {FULL, TEXT_ONLY, TEXT_LAYOUT, IMAGE_LAYOUT}) + per-class F1 + token confusion
(new `doccl/eval/confusion.py`) + Fisher displacement per component/depth. Output
`results/rca/dil_<m>_seed42_rca.json` (n=1 seed, provisional).

**Pre-registration + synthesis shipped (2026-07-16 eve, commits cab6c3a + 5ae2131):**
- `docs/RCA_HYPOTHESES_2026-07.md` — H1 modality-asymmetric drift (a: text / b: layout-vision),
  H2 head/label interference (modality-independent), H3 late-layer integration drift, H4
  recency/logit bias; decision rules against the Tier B JSON schema written BEFORE data exists
  (mask-validity guard ≥20 F1, method strata, H2/H4 disambiguation, QA sroie-vs-cord overlap).
- `scripts/rca_synthesize.py` (+ fixture test) — reads the 6 JSONs, emits
  `results/rca/c_modality_deltas.csv`, `c_confusion_flow.csv`, `c_displacement_vs_signature.csv`
  + 2 figures. Ready to run the moment Tier B lands.
**Tier C next session:** smoke-check naive (FULL AA ≈ 40 ± 3 vs head_refit_oracle 39.56), run
`rca_synthesize.py`, adjudicate H1–H4 per the pre-registered rules in
`docs/RCA_FORGETTING_BASELINES_2026-07.md`, then re-brainstorm methods against the RCA.

## QUEUED (2026-07-16 pm): READ-SIDE memory for CoLaR — kNN readout + metaplastic weights (built, runs queued)

**New direction (user-chosen, brainstormed):** every dead memory attempt (LARM, LexSlot-on-CoLaR,
nullspace_analytic) was a training-time WRITE — the redistribution law governs exactly that lever
class. Untested territory: READ the store at inference. Two assets nobody used: (1) layers <k are
frozen after task 0, so layer-k features are drift-free forever ⇒ CoLaR's SVD store is a stable
token-level labeled DATASTORE (LARM's stale-correction failure cannot occur below k); (2) forgetting
is head-localized (Finding 1) ⇒ replace the forgetting locus with a memory read.

**Shipped (commits dcedb94, 8ee7abd, tests 348 green):**
- `colar_knn` (R1): eval-time blend p=(1−λ)·softmax(head)+λ·kNN-vote over banked layer-k tokens
  (`doccl/methods/latent_datastore.py` shared view, zero extra bytes; λ=0 ≡ CoLaR, λ=1 = pure
  memory head). Training untouched ⇒ any gain is non-redistributive by construction.
- `colar_meta` (M1): Benna–Fusi m-level metaplastic consolidation on the plastic bucket, composed
  with replay (new no-op `_post_optimizer_step` hook in LatentReplay). FALSIFICATION-RISK track
  (EWC precedent); m=1 control must reproduce CoLaR before m=3 is read.
- `scripts/gate0_knn_probe.py`: NO-training layer-k kNN separability probe (pretrained trunk =
  conservative lower bound). Go/no-go for R1.
- R3 (`colar_mbpa`, MbPA-style episodic head adaptation) is designed in the plan but NOT built —
  conditional on R1 Gate 1 signal.

**Queued detached** (`scripts/run_readside_gate01.sh`, PID survives session, log
`results/readside_gate01.log`): waits for the seed sweep → Gate 0 probe → if ≥2/3 tasks best-F1≥20:
colar_knn λ=0.3 + λ=1.0 at k4/d50/r128 seed42 → colar_meta m1/m3 pair at k8/d50/r128.
**Gate 1 bar:** AA ≥ 87.6 with FUNSD not below 89.2 AND SROIE above 76.3 (non-redistributive win);
SROIE↑/FUNSD↓ 1:1 = just another frontier point, log plainly. λ=0 baseline = existing colar r128 run.
Plan file: `/home/thanh/.claude/plans/let-s-find-a-way-humming-flurry.md`.

## PRIOR ACTIVE (2026-07-16): Retention is REDISTRIBUTIVE under compressed replay — CoLaR is on the frontier

**Last decision:** do NOT build CoLaR+LexSlot (the KT doc's plan A). Killed on evidence, not opinion.

Chased CoLaR r128's one hole (SROIE 76.3 vs joint 83.4 = the whole AA gap) with two independent,
already-in-code levers. Both hit the target; neither improves AA; they don't stack. Full table +
reasoning in `EXPLORE.md` §7.

| dil k4/d50/r128 s42 | AA | BWT | [FUNSD, SROIE, CORD] |
|---|---|---|---|
| CoLaR r128 (base) | **87.6** | −1.7 | [89.2, **76.3**, 97.2] |
| + kcenter | 86.8 | −2.6 | [84.5, 78.9, 96.9] |
| CoLaR-Bal (soft) | 87.1 | −2.7 | [84.0, **79.9**, 97.4] |
| CoLaR-Bal + kcenter | 87.3 | −2.5 | [86.0, 78.6, 97.3] |

**The law:** SROIE rises only by spending over-held FUNSD (~1:1). FUNSD's 89.2 is *above* the joint
oracle's 87.9 — surplus that is load-bearing, not headroom. Generalizes E1's selection-redistributes
result to a mechanistically unrelated lever (loss shape) and to their combination.

**Why LexSlot helped DocCL but won't help CoLaR:** DocCL (84.7) sat *below* the frontier — slack on
every task, so LexSlot's buffer-backed capacity lifted all three at once (FUNSD/SROIE +4.7, CORD +2.0).
That row is also buffer-based (200 exemplars + KD + Fisher; standalone LexSlot = 42.2). CoLaR is on
the frontier and already replays. Same tool, different regime, opposite payoff.

**Shipped:** `doccl/methods/colar_bal.py` + config + 4 tests (commit e44e42b) — soft-target replay
for compressed latent replay. Literature gap (nobody reweights the reconstructed gradient of
compressed replay for minority classes); mechanism works (SROIE +3.6), conservation caps it.

**Blocked on / next:** the law is **PROVISIONAL — seed 42 only, dil only, LayoutLMv3 only**. The
86.8–87.6 band sits inside the ±1.1 joint seed sd. Seeds 7/123 running detached
(`scripts/run_seeds_conservation.sh` → `results/run_seeds.log`). Read those 6 runs before writing
the law into the paper. If the band survives 3 seeds → Finding 3c in the diagnostic chain.

## SUPERSEDED (2026-07-10 late): SLR falsified → successor idea "Consistency Law → PLaR/CoLaR"

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

## LARM (fuse CoLaR replay + lexical routing) — GATE 0 PARTIAL FAIL (2026-07-13)

Built `LARM(CoLaR)` (`doccl/methods/larm.py`): a lexically-keyed low-rank feature-correction memory
M, READ additively at layer k (routed by OCR sig), WRITTEN by replay CE. 6 unit tests green,
vectorized, committed `e3d7efa`/`f21456e`.

**Gate 0 ADD result (dil k4/d5 grid): AA 66.3 / BWT −34.4, row3 [60.5, 40.7, 97.7] — WORSE than its
own CoLaR/latent_replay baseline (76.0 / −19.3 / [72.6, 57.6, 97.9]).** So the "add, not replace"
safety claim is FALSE: the memory is not a helpful-or-no-op; it actively HURTS retention.

**Diagnosis (localized, mechanism clear):** LARM LEARNS each task fine (diag [87.3,82.8,97.7] ≈
CoLaR) — the damage is to RETENTION (row3). Root cause: **the additive correction is trained against
the plastic layers' state AT the time each task was learned, but applied at eval through plastic
layers that have since drifted.** Replay refreshes the correction relative to the FROZEN trunk
(below k); the drift is in the PLASTIC layers (above k) that consume the corrected features — the
correction and its consumer sit on opposite sides of the plasticity boundary. **LARM inherited
exactly LexSlot's staleness weakness, one layer up.** (Ruled out: cross-task routing leak — on dil's
disjoint vocab the softmax router correctly puts ~99.9% mass on same-task cells, verified.)

**GATE 0 COMPLETE: LARM REPLACE = AA 32.6 (collapsed below naive), vs ADD 66.3 vs CoLaR 76.0.**
Proven: (1) add > replace (consistency law holds directionally — replacing whole-doc features is
catastrophic); (2) but add < CoLaR (the layer-k fusion is a net negative). Mechanism NOT yet fully
pinned — see below (two wrong confident calls in a row; STOP asserting, TEST).

**BUG-FIX ATTEMPT (2026-07-14): the softmax→clamped-cosine gate fix did NOT recover LARM.**
Fixed run (dil k4/d5 grid): **AA 62.7 / BWT −39.4, row3 [63.5, SROIE 27.1, 97.6]** — still far
below CoLaR 76.0, ~same as buggy 66.3. So the softmax-null-option bug was NOT the whole cause
(my "fix confirmed" prediction was wrong — 2nd confident-wrong call). What the fix DID do (verified
on realistic dense keys): cross-task foreign leakage dropped to ~0.03 mass. But total gate mass
stays ~1.0 because a self-matching doc has cosine ~0.96 to its OWN cell → `clamp(min=1)` never
triggers → the in-domain correction fires at ~full strength on every eval doc. So the damage is
NOT cross-task routing; it's that the **in-domain learned correction itself is wrong at final-eval
time** (candidate: stale vs the drifted head — the drift hypothesis, now with SOME evidence but
STILL UNPROVEN). diag [87.3,82.0,97.6] = learns each task fine; row3 destroyed = retention only.

**DECISIVE DISCRIMINATOR NEEDED (do not guess again):** run LARM with memory ON at train, OFF at
eval (`mem_eval_off` flag). If eval-off → ~76: the learned correction is wrong at eval (drift) →
fixable. If eval-off still ~63: damage is during TRAINING (memory corrupted plastic weights/replay)
→ different fix. This one run splits the remaining hypotheses. GPU is free; fix is committed
(ca4e19e); do NOT conclude LARM's fate until this run decides.

**Fix hypotheses (if LARM is worth continuing):** (a) gate the memory OFF at eval / detach its
contribution to old-task eval once the head has moved; (b) put the rewrite where the head reads it
consistently — i.e. correct at the classifier-INPUT (post-plastic) not layer-k (pre-plastic), so no
plastic drift sits between correction and head; (c) make the correction itself replay-refit on the
CURRENT plastic state (re-derive up-factors each task from replayed activations run through the
current head). Option (b) is the cleanest — it moves the rewrite above the drift. Decide with user;
this may reduce LARM to "CoLaR + a head-input lexical adapter," a different (simpler) method.

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

**LEXSLOT VERIFICATION — MY 42.4 RESULT WAS THE WRONG CONFIG (corrected 2026-07-13, user caught it).**
The archived 87.3/88.4 row is **LexSlot slots layered on the FULL DocCL machinery**: git shows at
the run date (Jun 27, commit c1c6f19) `class LexSlot(DocCL)` inheriting CE + KD + reservoir-replay
+ depth-Fisher; the archived hparams confirm **use_replay=True, buffer_size=200, kd_alpha=1.0,
lambda_=2000**. After the 2026-07-02 RCA the class was refactored to `LexSlot(NaiveFineTune)` —
stripped to STANDALONE (no buffer/KD/Fisher) to isolate the slots' own effect. My "verification"
ran that standalone class (`method=lexslot`, current config use_replay-absent) → 42.4, which only
re-confirms the already-known "slots ALONE ≈ naive" RCA. It says NOTHING about the real hybrid row.
**Corrections to my prior claims:** (a) 87.3 is NOT standalone LexSlot; (b) 87.3 is NOT buffer-free
— it uses a 200-exemplar replay buffer (so it's DocCL-family, buffer-based); (c) whether it's a
pre-fix unnormalized-gate artifact is STILL OPEN — untested, not falsified.

**RESOLVED — the 87.3 row is REAL (buffer-based hybrid); my "artifact" claim was wrong.** The three
`_off` seed runs on disk are the pre-fix DocCL-HYBRID: seed42=88.4 [87.2,80.7,97.3], seed123=86.3
[85.6,76.7,96.8], seed7=87.2 [85.3,81.7,94.8] — all use_replay=True, buffer=200. Mean =
**87.3 ± 1.1**, exactly reproducing table_main. I had OVERWRITTEN seed42 with a wrong-config
standalone re-run (42.4); **RESTORED from archive** (parked the bad run at
`dil_lexslot_seed42_off_STANDALONE_wrongconfig`). The 3-seed evidence is intact again.

**Corrected status of LexSlot:** (a) 87.3 is a genuine near-joint result of DocCL+slots; (b) it is
**buffer-based** (200 exemplars), NOT buffer-free — so the chapter-7 "without storing any past data"
PROSE is factually wrong and must be corrected to "small-buffer (200 exemplars)"; the NUMBER stays;
(c) still-open (low priority): re-run the hybrid on POST-gate-fix code to confirm the normalized gate
doesn't move ~87 — plausibly it doesn't (buffer carries retention), but untested. (d) standalone
LexSlot ≈ naive was already the RCA finding; that is separate from this row and unchanged.
See [[clue-thesis-lexslot-row-source]].

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
