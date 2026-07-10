# ROADMAP

## DIRECTION (2026-07-10): method-led "SLR" pivot CLOSED → back to diagnostic + falsification paper

The 2026-07-10 method-led pivot (SLR = subspace-targeted latent replay) was tested to Gate 0 and
**FALSIFIED** — buffer-free feature replay does not work for doc-IE. See `STATE.md` "GATE-0 FINAL"
block and `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` Finding 3b for the full 5-run ablation. The
paper reverts to the pre-pivot **diagnostic + falsification** framing, now sharper: SLR/AGLR/Coreset
are the new *terminal level* of the falsification chain, with a clean single-variable mechanism
figure (consistency ablation: 4 whole docs AA 63.8 vs 4 decoupled carriers AA 36.7).

### Next Up (diagnostic-paper writing + evidence hygiene)
1. **Write-up DONE** (2026-07-10): Finding 3b added to the findings doc; SLR idea page marked
   falsified; STATE/ROADMAP reverted. Next: fold Finding 3b into the thesis/paper falsification
   section when drafting that chapter.
2. **Fair-ladder re-runs (before the paper table):** latent_replay @ 5ep docs=5 (the on-disk
   `dil_latent_replay_seed42` is stale 1-epoch AA 64.9, not the converged 87.3); regenerate the
   rm'd `dil_coreset_memory_seed42` (4-carrier) artifact. All cheap; do on Vast.ai with the grid.
3. **Multi-backbone generalization grid** (the pre-pivot priority, still open): does the
   forgetting locus + the negative result transfer across LiLT/BROS/BERT? `run_grid_multigpu.sh`
   on a rented GPU; the local box is too slow (~90 min/dil-run confirmed this session).
4. **Optional (only if a method angle is ever revived):** the ONE finding that would reopen a
   method is that whole-doc *consistency* is required — a "consistency-preserving compressed
   memory" (store few whole real docs, compressed losslessly) is untested, but that is ≈ raw
   replay with fewer docs, not a novel method. Park unless a reviewer asks.

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
