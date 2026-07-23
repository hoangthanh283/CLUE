# B+ generalization grid — audited execution plan (2026-07-18)

Requirement (ROADMAP critical path): headline rows at 3 seeds (42/7/123) ×
{cil_cord, dil, mixed} × {LayoutLMv3, LiLT, BROS, BERT}. Plan audited against the live
repo (DRY_RUN + .done set-diff by an independent agent); the empty-override script bug it
found (`${VAR:-}` vs `${VAR-}`) is FIXED in `run_grid_multigpu.sh` + `setup_remote.sh`
(commit this doc's commit), so the commands below now plan exactly what they say. RE-VERIFIED LIVE 2026-07-23: planned 387 unique cells, 32 already .done on disk (23 LiLT + 9 BERT pilots — auto-skipped), 355 missing confirmed. Second bug layer fixed same day: CORE_BEFORE_DOCCL/CORE_AFTER_DOCCL had independent hardcoded defaults (lines 322-323) ignoring CORE_METHODS; commands below now pass them explicitly.

## Missing cells (vs 263 `.done` markers on disk)

| backbone | missing cells |
|---|---|
| LayoutLMv3 (lexslot gap only) | 9 |
| LiLT | 103 |
| BROS | 126 (zero coverage today) |
| BERT | 117 |
| **total** | **355** |

14-method headline set (verified vs METHOD_REGISTRY): naive joint ewc lwf er der_pp l2p
dualprompt coda_prompt o_lora cl_lora er_cflat doccl lexslot.

**Estimate: ≈293 GPU-hours ⇒ ~$88–147 at $0.30–0.50/hr** (soft number; single 24 GB-class
card at batch 8 + AMP; halves wall-clock with 2 GPUs via `GPUS="0 1"`).

## Commands (paste on the rented box after `scripts/setup_remote.sh` bootstrap)

```bash
# Step 1 — LayoutLMv3 lexslot gap (small)
SCENARIOS="cil_cord dil mixed" RUN_LEXSLOT=1 RUN_DOCCL=0 RUN_ABLATION=0 RUN_BERT=0 \
  RUN_SINGLETASK=0 CORE_METHODS="" CORE_BEFORE_DOCCL="" CORE_AFTER_DOCCL="" PROMPT_METHODS="" CURRENCY_METHODS="" BACKBONES="" \
  GPUS="0" JOBS_PER_GPU=2 BATCH_SIZE=8 AMP=1 \
  bash scripts/run_grid_multigpu.sh

# Step 2 — secondary backbones. SPLIT DECISION 2026-07-23: BERT slice (117 cells)
# runs LOCALLY (launched, results/bert_slice_grid.log) — rented box does LiLT+BROS
# ONLY (229 cells, ~190 GPU-h) to avoid double-running BERT. Both sync .done state
# through the R2 bucket (durable resume), but do NOT rely on live dedup — keep the
# backbone split disjoint.
SCENARIOS="cil_cord dil mixed" BACKBONES="lilt bros" \
  BACKBONE_METHODS="naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora cl_lora er_cflat doccl lexslot" \
  RUN_SINGLETASK=0 RUN_BERT=0 RUN_DOCCL=0 RUN_LEXSLOT=0 RUN_ABLATION=0 \
  CORE_METHODS="" CORE_BEFORE_DOCCL="" CORE_AFTER_DOCCL="" PROMPT_METHODS="" CURRENCY_METHODS="" \
  GPUS="0 1" JOBS_PER_GPU=2 BATCH_SIZE=8 NUM_WORKERS=2 AMP=1 \
  bash scripts/run_grid_multigpu.sh

# Preview a plan without running:  DRY_RUN=1 <same env> bash scripts/run_grid_multigpu.sh
```

Notes from the audit: resume-safety is real (`.done` skip in the dispatch loop — safe to
relaunch after crashes); DRY_RUN prints the full plan, not the remaining delta — use
`.done` diffing for true remaining counts; LiLT tokenizer + XFUND RAM traps do not apply
to this scenario set; `PYTORCH_CUDA_ALLOC_CONF` is set inside the script. Afterwards:
`scripts/analyze_results.py` → `table_backbone_*` regenerate, then `ingest_to_thesis.py`.

**Blocked on:** renting the box (user action). Local 6 GB box is NOT suitable (~90 min/run
⇒ ~530 h serial).
