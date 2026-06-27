# ROADMAP

## Next Up
1. **Launch the full multi-backbone grid on a rented GPU.** Code + grid are ready
   (commits `8557ac6`/`d76ad3d`/`170f81a`, pushed). The default `bash
   scripts/run_grid_multigpu.sh` is now the 873-job sweep (LayoutLMv3 + all 14 methods ×
   {lilt,bros,bert}). Local RTX 2060 can't fit it → use `scripts/setup_remote.sh` +
   `docs/MULTI_MACHINE_RUN.md` / `RUNBOOK.md`. Preview first with `DRY_RUN=1`. Resume-safe
   (257 runs already `.done`). Watch `results/logs/progress.json` + the heartbeat.
2. **Analyze → thesis.** When the sweep finishes: `analyze_results.py --source local`
   (or remote) → `all_runs.csv`, `pivot_*.csv`, `table_*.tex`, figures →
   `ingest_to_thesis.py` → build the secondary-backbone **generalization tables**
   (does the forgetting locus + the proposed method transfer across LiLT/BROS/BERT?).
3. **Let the local `dil_lexslot_seed42` run finish** (in progress at session end) before
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
