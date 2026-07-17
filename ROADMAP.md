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

### RESUME HERE (2026-07-17): RCA COMPLETE — verdicts in; read-side chain running
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
Next session: (1) read read-side results (`results/gate0_knn_probe.json`, colar_knn λ runs
vs CoLaR 87.6/[89.2,76.3,97.2], colar_meta m1 control BEFORE m3); (2) 3-seed conservation
verdict is IN (base 87.8 = bal 87.8 > kc 87.0; no lever beats base at any seed) — fold into
STATE/paper as Finding 3c; (3) re-brainstorm method design against the RCA; (4) consider
the H4 falsification quickies (frozen-trunk naive; logit-prior correction).

### RUNNING (relaunched 2026-07-17 after RCA): READ-SIDE memory chain
`scripts/run_readside_gate01.sh` (log `results/readside_gate01.log`): Gate 0 probe →
colar_knn λ={0.3,1.0} k4/d50/r128 s42 → colar_meta m1/m3 k8/d50/r128. If R1 signals
(λ>0 beats λ=0 by real margin), build R3 `colar_mbpa` per the plan
(`/home/thanh/.claude/plans/let-s-find-a-way-humming-flurry.md`).

### SUPERSEDED (2026-07-14 KT, killed 2026-07-16 am): CoLaR + LexSlot integration
Design A (slots-on-CoLaR) killed by the redistribution law — CoLaR has no slack to harvest
(STATE.md "PRIOR ACTIVE"). Handoff kept for reference: `docs/KT_2026-07-14_colar_lexslot.md`.
LARM FAILED (all variants 32–66 AA); do NOT rebuild it — read KT §4 for why.

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
3. **Write-up.** Fold Finding 3b + the consistency-law single-variable figure (4 whole docs 63.8 vs
   4 decoupled carriers 36.7) into the thesis/paper falsification chapter. Position CoLaR as the
   constructive control (proves negatives weren't effort-limited), PLaR as the bounded-negative
   (coverage-limited proxy replay).
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
