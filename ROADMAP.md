# ROADMAP

## DIRECTION (2026-07-10): method-led "SLR" pivot CLOSED → back to diagnostic + falsification paper

The 2026-07-10 method-led pivot (SLR = subspace-targeted latent replay) was tested to Gate 0 and
**FALSIFIED** — buffer-free feature replay does not work for doc-IE. See `STATE.md` "GATE-0 FINAL"
block and `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` Finding 3b for the full 5-run ablation. The
paper reverts to the pre-pivot **diagnostic + falsification** framing, now sharper: SLR/AGLR/Coreset
are the new *terminal level* of the falsification chain, with a clean single-variable mechanism
figure (consistency ablation: 4 whole docs AA 63.8 vs 4 decoupled carriers AA 36.7).

### Scope decision (2026-07-13): diagnostic-led, FULLY SCOPED ("B+")
A game-theoretic acceptance analysis (real 2025 venue data — `EXPLORE.md` §6) settled the venue
strategy: **diagnostic + falsification, multi-seed × multi-scenario × multi-backbone.** P(accept)
ordering B+ 22–30% > diagnostic-as-is 12–18% > method-paper 8–12% at ICML/CVPR (AAAI +~5–8pp). The
+12pp B→B+ jump is the critical path below and is **compute, not new research** — re-runs of
existing code on Vast.ai. CoLaR is the *constructive control*, PLaR the *bounded-negative*, not
method contributions.

### RESUME HERE (2026-07-18): thesis structural reframe + grid launch
(1) Execute `docs/THESIS_METHOD_CHAPTER_REFRAME_2026-07.md` (session task #5) — resolve
[EXAMINER] annotations, rewrite ch3 §3.4 / ch6 §6.1.5 / title / ch1 / ch7. Buffer-free
correctness pass is DONE (25+ sites, builds 125 pp). (2) USER: rent Vast.ai box →
`docs/BPLUS_GRID_PLAN_2026-07.md` paste-ready (355 cells, ≈293 GPU-h, ~$88–147).
(3) After grid: analyze_results.py → ingest_to_thesis.py → backbone tables.

### EXPLORATORY SIDE GATE (2026-07-23): continual DocRE salient-entity graph replay
CPU-only memory builder + synthetic test are implemented. After the active grid drains:
obtain Re-DocRED task JSON; run equal-byte random vs graph-salient construction; then train
the same DREEAM replay baseline on both outputs. Continue only if graph selection improves
relation/motif coverage and macro-F1/BWT at equal bytes. Next test intact-document graph
selection against isolated evidence/triples; adaptive realignment comes only after that
passes. Keep this outside the current paper's B+ critical path.


### DONE (2026-07-17 eve): KILL-TESTS ADJUDICATED — write-up next
All 5 kill-tests complete + verified (STATE.md ACTIVE block; RCA doc kill-test sections).
Headline: every cheap correction is a registered null; re-exercise is the only lever;
per-doc EM dead as method-chapter headline (fallback framing pre-written in the prereg).
Next: (1) fold kill-test suite + read-side closure into thesis falsification chapter;
(2) decide method-chapter shape (negative-result suite + CoLaR control vs replay-adjacent
evidence-store method); (3) resume the B+ grid critical path below. Follow-ups: λ-sweep
for marginal_kl (unregistered), weight-geometry probe if "directions" claim is wanted,
train.py run-name/hydra-dir mismatch (session task #4).

### DONE (2026-07-17): RCA COMPLETE — verdicts in; read-side chain closed
**RCA Tier C is DONE + verification-amended:** `docs/RCA_FORGETTING_BASELINES_2026-07.md` —
**H2 SUPPORTED, H4 SUPPORTED-AS-AMENDED (H4′ marginal snap), H3 REFUTED, H1 REFUTED as
mechanism, QA not supported**. Root cause: readout-marginal snap on a class-asymmetric
substrate (the head's output marginal on old tasks snaps to the just-trained task's gold
marginal, cos ≈ 1.0 for failing methods — `results/rca/c_marginal_snap.csv`;
never-re-exercised classes KEY/HEADER extinguish mask-uniformly; O collapses too (0.00%,
invisible to seqeval); trunk drift front-loaded but functionally minor; no modality pathway
is the culprit). Method implications §
"Method-design implications" — notably: cheap logit-prior recalibration baseline (test
before any new method), targeted minority-class replay, read-side memory premise
independently re-derived, colar_meta predicted to FAIL (treat its queued run as the
falsification test).
Next session: (0) rerun `uv run python scripts/rca_deep_dive.py` after any new RCA JSON lands; (1) read read-side results (`results/gate0_knn_probe.json`, colar_knn λ runs
vs CoLaR 87.6/[89.2,76.3,97.2], colar_meta m1 control BEFORE m3 — colar_meta m1 was
training as of 07-17 ~08:00); (2) read kill-test results (`results/rca/killtests/readout_fixes.md`
+ the three new `dil_*_seed42_rca.json`) and adjudicate STRICTLY by
`docs/RCA_KILLTESTS_PREREG_2026-07.md` — extend the RCA doc with a Kill-tests section;
(3) 3-seed conservation verdict is IN (base 87.8 = bal 87.8 > kc 87.0; no lever beats base
at any seed) — fold into STATE/paper as Finding 3c; (4) then decide the method-chapter bet
(per-doc EM is the headline candidate if its kill-test lands near the task-ID oracle).

### READ-SIDE memory chain — GATE 1 ADJUDICATED (2026-07-17 pm): NO SIGNAL, R3 not built
Results (vs CoLaR λ=0: 87.6 [89.2, 76.3, 97.2]): colar_knn λ=0.3 = 87.64 [89.2, 76.5,
97.2] — formal bar edge-met but "real margin" clause fails (+0.04 AA, +0.2 SROIE, n=1);
λ=1.0 = 46.24 (destructive; pure kNN ≈ gate0 probe ceiling). colar_meta m1 control =
85.85 — fails to reproduce CoLaR (−1.75), so m3 (still training) reads only as
consistent/inconsistent with the RCA fail-prediction. **R3 (colar_mbpa) is NOT built —
gate closed.** Read-side-as-blending is dead; the RCA kill-tests (marginal corrections)
attack the same readout locus directly and supersede this direction. Full verdict in
STATE.md ACTIVE block.

### CLOSED (2026-08-15): CoLaSlot-FDP token-pathway gate; FD closed
The fully matched FD run reproduces CoLaR exactly but changes AA by only +0.00285 and old-domain
mean by +0.00427 while adding 55.6% runtime. Routing is already zero-error on accepted FUNSD/SROIE
documents; measured drift is real, but a fixture shows the linear head residual cannot cancel it
while remaining null on unsupported tokens. LR increases plateau and removing null becomes
unstable, so FD and its optimizer/null sweeps are closed.

FDP d5/r64 seed-42 matrix is exactly equal to FD: AA 60.792444 and final row
[44.321885, 41.347943, 96.707504]. Versus pure CoLaR this is only +0.002849 AA and +0.004273
old-domain mean; precision is non-regressive but efficacy fails by two orders of magnitude while
runtime rises 55.9%. Eight pathways are active per owner, but random-zero bilinear initialization
makes their energy class-agnostic at the first update and scarcely target-aware thereafter. Close
FDP, temperature/LR sweeps, d50/r128, and extra seeds.

### CLOSED (2026-08-15): CoLaSlot-FDA support-gated analytic compensation
FDA passed its held-out support prerequisite but failed the matched d5/r64 seed-42 gate:
58.8440 AA versus CoLaR 60.7896, with final deltas [-11.5315, +5.9272, -0.2324]. The
analytic solve canceled 61--76% of measured drift but the additive residual harmed FUNSD
retention through repeated online residual accumulation and readout overgeneralization. Close FDA and all ridge/LR sweeps.
The literature-driven successor is a post-task route-conditioned prototype/readout alignment
with an explicit O/NA null and held-out precision gate; d50 and extra seeds stay locked.

### CLOSED (2026-08-15): CoLaSlot-Proto route-conditioned readout
The RNG-corrected matched gate reproduced every CoLaR training checkpoint but finished at
60.7526 AA versus 60.7896 control. Final deltas are [+0.4229, -0.5339, 0.0000], so efficacy fails
and SROIE crosses the -0.5 safety floor. Support precision alone cannot detect owner-specific
readout harm. Close scale/rank sweeps and promotion; admit one bounded successor only: LODO
owner-utility gating that disables prototype owners unless held-out replay improves.

### CLOSED (2026-08-15): CoLaSlot-Proto-U owner-utility gate

LODO utility rejected both old owners because base and prototype replay F1 tied at 100/100. The
result is exact CoLaR: 60.7896 AA and [44.3219, 41.3394, 96.7075]. The proxy is saturated on
memorized replay documents and cannot predict held-out utility. Close prototype gates and sweeps.
One bounded successor is admitted: age-aware CoLaSlot-R reads that suppress the newest owner at
stage evaluation, preserving exact current-task acquisition while retaining older slot reads.

### CLOSED (2026-08-15): CoLaSlot-Age read lifecycle

Single-FUNSD slots-on and newest-slots-off both equal 86.1225, below CoLaR 87.2931. The current
slot is inference-inert and the regression is training co-adaptation, so read gating cannot fix it.
Close before DIL. A successor must separate base and sidecar gradients.

### CLOSED (2026-08-15): CoLaSlot-Sidecar separated head slots

The exact matched gate gives 60.7588 AA and [44.3219, 41.2469, 96.7075], or -0.0308 AA versus
CoLaR. Base checkpoints are exact, proving separation works; head-only sidecars are redundant and
slightly harm SROIE. Close d50/seeds. One capacity discriminator remains: reuse the same method
with existing `head_late` representation slots to model the missing co-adapted upper-layer branch.

### CLOSED (2026-08-16): CoLaSlot-Sidecar late capacity

The head+late discriminator gives 59.0502 AA and [43.4563, 36.8707, 96.8236], or -1.7394 AA and
-2.6672 old-domain mean versus exact CoLaR. Detached head capacity is redundant; detached
representation capacity overgeneralizes. Close depth/rank/LR sweeps and the detached-sidecar family.

### CLOSED (2026-08-16): CoLaSlot-Shadow replay-trained branch

A 196,608-parameter shared rank-16 residual now co-adapts on the same current+replay sample while
the exact CoLaR base remains the newest/abstained fallback. Single-FUNSD matches all control
checkpoints and final 87.293087 exactly. The preregistered k4/d5/r64 seed-42 DIL gate is running;
no d50/r128 or extra seeds unless it passes every fixed efficacy, domain, and precision guard.

The task-1 boundary is already adverse: FUNSD 50.13 versus exact CoLaR 54.09. Code tracing shows
current-task gradients dominate a shared branch that the age policy suppresses on current data and
reads only for old domains. One preregistered correction is ready: `colaslot_shadow_replay` trains
that branch only on replay. It must match single FUNSD exactly, then reach FUNSD 54.594293 with
exact SROIE 80.833333 after task 1 to justify running CORD. No hyperparameter sweep.

Final result: 57.5963 AA and [41.1619, 34.9195, 96.7075], or -3.1933 AA and -4.7899
old-domain mean versus CoLaR. Old-domain precision falls 3.60/6.12 points; SROIE recall is nearly
unchanged, so the shadow produces unsupported false positives. Runtime is +77.95%. Close d50,
seeds, rank/depth/LR/router sweeps.

### GO (2026-08-16): CoLaSlot-Shadow-Replay bounded correction

The one admitted correction removes newest-task representation gradients while retaining the exact
base, owner heads, replay sample, rank, depth, LR, and router. Preflight matched single-FUNSD
exactly; task-1 gate passed (FUNSD 56.1296 >= 54.594293, exact SROIE 80.833333). Final
**62.7027 AA / [46.4097, 44.8750, 96.8236]** — deltas vs exact CoLaR **+1.9131 AA**,
**+2.8117 old-domain mean**, all domains >= -0.5, old micro-precision **+2.35/+3.80**. All four
final guards pass. Evidence: `results/dil_colaslot_shadow_replay_seed42_k4/`.

The d50/r128 scale-up is a **NO-GO**: 86.5697 AA vs a matched CoLaR e5 control at 86.3829 —
+0.1868 AA and +0.2802 old-domain mean, below the +0.5/+1.0 guards, at +76.78% runtime.
The gain is a low-coverage phenomenon: at d50 CoLaR's replay saturates retention and the shadow
is inert-positive. d5 GO stands as a mechanism finding. Remaining admissible: extra seeds at
d5/r64 only. Evidence: `results/dil_colaslot_shadow_replay_seed42_k4_d50_r128/`,
`results/dil_colar_seed42_k4_d50_r128/`.

### CLOSED (2026-08-15): CoLaSlot-RO online hard-label gate
RO preserves the exact same-state base trajectory but changes final domains by only
[-0.054, +0.026, 0.000], giving -0.009 AA and -0.014 old-domain mean. Owner CE is
4.47e-5--2.19e-3, recall is unchanged, memory is unchanged, and runtime rises by 26.9 minutes.
It fails both improvement clauses. Hard-label residual fitting is closed at post-task and
online timings; FD changes the target to measured one-step function drift.

### CLOSED (2026-08-14): CoLaSlot-RF/RA post-task residual refits
Matched d5/r64 seed-42 runs preserve an identical slot-free CoLaR trajectory. RF produces
near-zero hard-label gradients and is exactly inert at final evaluation. RA's acquisition-logit
anchor restores strong gradients but changes the final row by [-0.29, -0.99, 0.00], or
-0.43 AA; SROIE loses precision with unchanged recall. Both fail the preregistered gate, so
no d50/r128 or multi-seed expansion. Next permitted gate is support-bounded: an entity/class
residual may activate only after held-out replay evidence shows positive entity F1 with
non-negative precision; all unsupported corrections stay exactly zero. Do not add another
router, residual rank, or dense-anchor sweep before that falsification test passes.

### CLOSED (2026-08-08): CoLaSlot-R scalar-abstention follow-up
CoLaSlot-R d5/r64 improves CoLaR by +14.02 AA and -20.45 AF, but CORD -0.84 fails the
domain guardrail. Training-only 5-fold calibration found no usable global margin: 0.05 has
92.47% coverage but only 97.71% accepted accuracy on CORD; the zero-error 0.326 threshold
has 0% FUNSD/SROIE coverage. No 0.001-grid threshold meets >=50% coverage and >=99%
accepted accuracy for every seen stage/domain. **No cheap rerun, d50/r128, or extra seeds for CoLaSlot-R.**
Reopening requires a materially different router or a decoupled slot lifecycle that makes wrong
routes non-destructive, with a new preregistration; do not continue scalar threshold sweeps or
tune against final test F1. Evidence:
`results/gates/colaslot_r_margin_feasibility_seed42.json`.

### Next Up (the B+ critical path, priority order)
1. **B+ generalization grid — THE +12pp MOVE (critical path).** Every headline row at **3 seeds
   (42/7/123) × {CIL-CORD, DIL, mixed} × {LayoutLMv3, LiLT, BROS, BERT}**. This closes the
   correctness/generality gate that bounds a diagnostic paper (a reviewer's "a negative result on
   ONE setup is not a finding"). The ICML-2024 architectural-perspective diagnostic that DID land
   was architecture-general; ours must match. `run_grid_multigpu.sh` on Vast.ai (local box ~90
   min/dil-run — too slow).
2. **Evidence hygiene (blocks the paper table).** Clean grid re-run of the stale 1-epoch
   `dil_latent_replay_seed42` (64.9 → ~87 converged); regenerate the deleted 4-carrier
   `dil_coreset_memory_seed42`; CoLaR seeds 7/123 + cil_cord; int8-on-factors probe (~30 MB @ r128).
3. **Write-up.** DONE for the thesis (2026-08-15, ch6 §6.6 + backbone/ch7 refresh — see STATE);
   the PAPER-side fold of Finding 3b + consistency-law figure remains. CoLaR positioned as the
   constructive control, PLaR as the bounded-negative.
   CoLaSlot-RF/RA/RO/FD/FDP are closed same-state NO-GOs; no additional seeds or full-budget
   expansion is justified.
4. **LexSlot prose correction (correctness hygiene — a reviewer WILL catch it).** The 87.3 row is
   REAL but **buffer-based (200 exemplars, the DocCL-hybrid)** — verified this session (seeds 42/7/123
   = 88.4/87.2/86.3, all use_replay=True). Chapter-7's "buffer-free / without storing any past data"
   is factually wrong → change to "small-buffer (200 exemplars)". The number stands; only the framing
   is corrected. (Standalone LexSlot ≈ naive 42.2 is a separate config, unchanged.)

### Notes
- **bd task-tracking:** still blocked (schema-migration fork) — track here, not in beads, until
  the designated-migrator decision is made.

---

## Direction change (2026-07-02): LexSlot DROPPED → LexMem (beads CLUE-b0b)
RCA verdict: standalone LexSlot is naive-level (AA 42.2 vs naive 40–43; the pre-fix 86.9
was an unnormalized-gate artifact; the thesis 87.3 row was the DocCL hybrid). LexSlot-FM
fails on capacity (50.4 F1 single-FUNSD). Successor **LexMem** = Sparse Memory Finetuning
(arXiv 2510.15103) adapted to doc IE: frozen base after task 0 + key-value logit-memory
head, TF-IDF slot-access selection, top-t sparse SGD. Implemented + committed (all e2e
green). **Pilot `dil_lexmem_seed42` queued** behind the leftover lexslot_fm run; go/no-go:
new-task diag F1 ≥ 75, task-0 drop ≤ 5, low slot Jaccard. Go → cil_cord + dil_xlingual
×3 seeds; no-go (after `unfreeze_late_n=4` retry) → pivot to head-bank/merge (DocMERGE/LCA).
The LexSlot items below are superseded except as honest negative-result material.

## Thesis-review follow-ups (2026-06-29 deep review — bd was dolt-locked, logged here)
- **Parser: fold LexSlot variant suffixes into a clean `lexslot` cell.** `analyze_results.py`
  doesn't recognise `_off/_head_only/_uniform`, so `--proposed lexslot` empties the
  main/ablation tables and the LexSlot row is hand-maintained in
  `thesis/generated/table_main.tex`. Map `_off` → canonical cell, others → ablation rows,
  then flip `--proposed` default back to `lexslot`. Ground truth: `dil_lexslot_*_off` =
  AA 87.33±1.03, BWT −2.27±1.23. See [[clue-thesis-lexslot-row-source]].
- **Freeze-the-bucket causal control.** The depth probe is a budget-placement result
  (uniform under-fits to AA 42.6 but forgets *less*, BWT −1.8), not direct proof the head
  is the forgetting locus. Freeze the high-displacement head/late bucket vs a low-displacement
  bucket and show retained F1 recovers. Thesis now flags this as future work (ch6/ch7).
- **Pending LexSlot grid.** Validated only on DIL/LayoutLMv3 (3 seeds, `_off`). Run on
  CIL-CORD/WildReceipt/Mixed/DIL-XLing and LiLT/BROS/BERT (gate expected ineffective on CIL).
- **Lower-priority:** regenerate Fisher/CKA/condition tables from pilot artifacts (currently
  hand-kept inline); add a component-separability defense paragraph up front in ch2/ch3.

## Next Up
1. **LexSlot-FM FM refit fix landed (AA 48.67).** Next directions (decide):
   - **(a) Increase refit budget** — FM_1 (SROIE) needs >50 samples to recover from
     encoder drift. Try `fm_refit_samples=200` / `fm_refit_epochs=10` → if T1 recovers to
     ~40+, AA targets 55+.
   - **(b) Conditional gate** — TF-IDF for Korean/English decision (T2=88.64 best), then
     layout within English (T0=40.78, T1=20.35). Would combine best of both.
   - **(c) More seeds** — run seeds 0-4 for variance estimates on the fixed refit.
   - **(d) Analysis-paper pivot** — document the FM refit mechanism and the T1/T2
     tradeoff as a controllable bias-variance knob in functional-memory methods.
2. **Launch the full multi-backbone grid on a rented GPU.** Code + grid are ready
   (commits `8557ac6`/`d76ad3d`/`170f81a`, pushed). The default `bash
   scripts/run_grid_multigpu.sh` is now the 873-job sweep (LayoutLMv3 + all 14 methods ×
   {lilt,bros,bert}). Local RTX 2060 can't fit it → use `scripts/setup_remote.sh` +
   `docs/MULTI_MACHINE_RUN.md` / `RUNBOOK.md`. Preview first with `DRY_RUN=1`. Resume-safe
   (257 runs already `.done`). Watch `results/logs/progress.json` + the heartbeat.
3. **Analyze → thesis.** When the sweep finishes: `analyze_results.py --source local`
   (or remote) → `all_runs.csv`, `pivot_*.csv`, `table_*.tex`, figures →
   `ingest_to_thesis.py` → build the secondary-backbone **generalization tables**
   (does the forgetting locus + the proposed method transfer across LiLT/BROS/BERT?).
4. **Let the local `dil_lexslot_seed42` run finish** (in progress at session end) before
   starting anything else heavy on the local box — it fits exactly one job.

## Deferred
- Widening the sweep to long-horizon CIL variants / extra seeds — only if the
  generalization tables need more statistical power; current 3 seeds is the standard.
- The HGT/CUBER and LCA/DocMERGE NO-GO findings feed the analysis-paper framing
  ("nothing transfers on the head in doc-IE; only replay works") — revisit when writing
  that section, not before the grid results land. See STATE.md history.

## Done (this session, 2026-06-27)
- All CL methods made backbone-agnostic (3 LayoutLMv3 couplings fixed); full
  method × {LiLT,BROS,BERT} × {CIL,DIL} e2e matrix added and verified green.
- Grid defaulted to the full multi-backbone sweep. See [[clue-backbone-agnostic-audit]].
