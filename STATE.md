# STATE

## All methods made BACKBONE-AGNOSTIC (2026-06-27) — verified green on 4 backbones
Audited every CL method for LayoutLMv3 coupling (2 parallel audit agents: methods +
data/encoder). Data/encoder path was already clean. Fixed 3 method-layer couplings:
(1) ported `token_features` to `TokenClassificationWrapper` (+ BERT text-only override
that drops `bbox`) → unblocks `lca`/`hgt`; (2) `prompt_base.py` hard `batch["pixel_values"]`
→ `.get()` → unblocks `l2p`/`dualprompt`/`coda_prompt`/`hrp`; (3) ported the PEFT
`ModulesToSaveWrapper` head-unwrap from `LayoutLMv3Wrapper` into the base wrapper →
unblocks `o_lora`/`cl_lora` on CIL. Closed the TEST GAP: e2e method-lifecycle tests ran
LayoutLMv3-only; added `test_method_{cil,dil}_lifecycle_secondary_backbone` = full method
set × {LiLT,BROS,BERT} × {CIL,DIL}. Test-data gotcha fixed: `_TinyKIEDataset` is now
backbone-aware (LiLT needs ordered boxes x0<=x1/y0<=y1; BROS needs [0,1] floats; only
LayoutLMv3 gets pixel_values). **Verified: LayoutLMv3 12/12, BERT 42/42, BROS 42/42, LiLT
0-fail; fast not-slow lane 0-fail; ruff/black clean.** Commits `8557ac6` (fixes) +
`d76ad3d` (full method set on secondaries) + `170f81a` (default `BACKBONES="lilt bros bert"`)
— all pushed to `origin/doccl`. **Grid default is now the full 873-job multi-backbone
sweep**: bare `bash scripts/run_grid_multigpu.sh` runs LayoutLMv3 main grid + all 14 methods
× {lilt,bros,bert} × 5 scenarios × 3 seeds (incl. doccl depth + lexslot slot ablations);
`BACKBONES=""` opts back to LayoutLMv3-only. Resume-safe (skips `results/<run>/.done`; 257
already done). DRY_RUN verified (free, ~30ms): 873 jobs = 243 LayoutLMv3 + 210 each
lilt/bros/bert. See [[clue-backbone-agnostic-audit]].
**BLOCKED: the 873-job sweep needs a rented GPU** — local RTX 2060 (6 GB) can't fit the
heavy methods (der_pp ~35 GiB, er ~27) and is currently busy training `dil_lexslot_seed42`.
NEXT SESSION: spin up the remote GPU (`scripts/setup_remote.sh` + `docs/MULTI_MACHINE_RUN.md`),
launch the sweep, then `analyze_results.py` → generalization tables → ingest into thesis.

## HGT method BUILT + TESTED + run (2026-06-26) — ⚠️ NO-GO verdict (analysis-paper pivot)
**HGT (Head-localized Gradient-subspace Transfer)** + CUBER baseline implemented via
subagent-driven-development (5 tasks, all reviewed clean): `doccl/methods/grad_subspace.py`
(b075dc5), `hgt.py` (c45354d), `cuber.py` (9846303), e2e (5e9fd05), runner cf2aa07. Spec
`docs/superpowers/specs/2026-06-26-head-gradient-subspace-transfer-design.md`; plan
`docs/superpowers/plans/2026-06-26-head-gradient-subspace-transfer.md`; ledger
`.superpowers/sdd/progress.md`. See [[clue-headgrad-direction]].
**α-ablation on dil (5ep, results/hgt_ablation/) — VERDICT: NO-GO on positive-BWT.**
AA/BWT: naive 38.7/−75.2 | er 86.2/−3.3 | hgt α0 24.8/−7.9 | hgt α0.5 25.3/−21.1 | hgt α1
24.9/−23.2 | **cuber 39.8/−73.0**. **CUBER (whole-network) ALSO fails to give positive BWT**
(−73 ≈ naive −75; learns all at-learning diag [86,82,97] but forgets all) → the aligned-gradient
transfer mechanism does NOT manifest on doc-IE head-only OR whole-network. Only REPLAY (er)
works, needs a buffer. **α-BWT monotonically WORSE (−7.9→−21.1→−23.2)** while
at-learning rises (diag 10.8→24.9→28.5), AA flat ~25 → α is a STABILITY↔PLASTICITY dial, NOT
transfer; no positive BWT anywhere. All HGT AA~25 ≪ er 86 (frozen-backbone cap). Integration
fully validated (hooks survived real training, no crash). **→ analysis-paper framing** (the
sharp finding: head-gradient steering trades stability/plasticity, doesn't transfer; with
merge-fails + LCA-fails = the "nothing transfers on the head in doc-IE" story). CODE correct
regardless. PENDING: cuber result, then FINAL whole-branch review + finishing-the-branch.

## Brainstorm (2026-06-26) — lexical-similarity-coupling CL method (DIRECTION, pre-spec)
After DocMERGE-merge falsified + LCA-align would re-skin the baseline, brainstorming a NEW
A*-tier direction built on our ONE validated asset (sparse-OCR lexical routing, drift-immune).
**Core idea:** the OCR lexical-similarity matrix S between tasks gates TWO transfer channels —
(rep/prompt channel → FWT: warm-start a new task's prompts from lexically-similar prior tasks;
head channel → BWT: gated GRADIENT FLOW [not merge] couples related tasks' head rows so they
refine each other, isolating unrelated ones). Gate = drift-immune OCR signature cosine.
**S VALIDATED (computed from raw OCR, scratchpad/compute_task_similarity_S.py):** dil task-sim
matrix has the needed CONTRAST — SROIE↔CORD=0.311 (receipts cluster, shared top-words
total/cash/tax = LABEL-BEARING tokens) vs FUNSD↔{SROIE,CORD}=0.10–0.15 (form is the outlier).
~3× contrast → not flat → real signal to gate on. **Key insight:** shared vocab IS the
label-bearing vocab → S measures "shared extraction rules", not generic affinity (uniquely a
structured-IE signal; no image-CL analogue = the moat).
**Open forks (user deciding):** (1) full-vocab S vs label-token-only S (lean: label-token =
"shared rules" framing); (2) run a cheap SROIE↔CORD transfer-sanity probe first?
**Warnings surfaced:** FWT metric is BROKEN for CIL (measures zero-shot-on-unseen-labels ≈ −85
for everyone, NOT real transfer) → need a learning-curve/few-shot FWT metric (a 2nd
contribution); cil_cord routing COLLAPSES to 0.22 (5 sessions = same CORD dataset → signatures
overlap) → lexical routing needs DISTINCT-vocab tasks (dil's 0.92), fails on same-dataset CIL.
dil is the proving ground; may be a one-scenario result — be clear-eyed.

## Active Work (2026-06-26) — LCA baseline (ICLR 2026) IMPLEMENTED + queued to run
**LCA: Local Classifier Alignment** (Tran/Vargas/**Khoat Than**, ICLR 2026; arXiv 2603.09888,
repo tungts1101/LCA) — faithfully ported to doc-IE and committed (d86a53a + 1231978 on
`doccl`). See memory [[clue-lca-paper]]. **Method `lca`** (`doccl/methods/lca.py`): per task,
SGD+cosine finetune backbone+head → per-BIO-class feature Gaussians (μ,Σ over TOKEN features)
→ **TIES-merge** backbone task-vectors (`doccl/methods/ties_merge.py`, verbatim port of LCA
`helper.merge`) → **classifier ALIGN** on sampled N(μ,Σ) features with CE + robust·(intra-class
loss-variance) + entropy·(entropy), skip task0. Added `LayoutLMv3Wrapper.token_features`.
Registered in METHOD_REGISTRY + `_STD_FORWARD`; config `configs/method/lca.yaml`. **9 unit +
2 e2e (CIL/DIL real LayoutLMv3) pass; ruff+black clean.**
- **KEY INSIGHT:** LCA = TIES-merge (which DocMERGE has, and which ALONE failed) + a post-merge
  classifier re-grounding via GENERATIVE Gaussian feature replay (which DocMERGE LACKS). LCA
  independently CONFIRMS the DocMERGE RCA: merging alone is insufficient; you must re-align the
  head. Their align is buffer-free (μ/Σ, not exemplars) = the "generative head replay" floated
  in the DocMERGE brainstorm and skipped. **Strong candidate to salvage DocMERGE.**
- **LCA VERDICT (dil seed42 5ep, 4/5 done; der_pp running but ~88 known): LCA does NOT port to
  doc-IE.** Table AA/BWT: naive 38.7/−75.2 (floor) | lca_merge_noalign 40.2/−44.9 | **lca (full)
  41.9/−23.8** | der_pp ~88/~−3. (lca_nomerge_noalign 21.8/−4.8 = ablation bug, merge_coef=0
  RESETS backbone — IGNORE.) **(1) TIES backbone merge WORKS** (BWT −75→−45) — merging a TRAINED
  backbone is LMC-valid (real task-vectors), the mirror of why DocMERGE's frozen-backbone HEAD
  merge cancelled. **(2) ALIGN trades plasticity↔stability**: BWT −45→−24 (regrounds head) but
  damages at-learning (merge-only task1 diag 48.7 → full-LCA 8.3) — single-Gaussian-per-class
  replay over-regularizes per-token feats (+'O' dominance). **(3) LCA AA 41.9 ≈ naive 38.7, ≪
  replay 88** → ICLR'26 image-CIL SoTA lands at HALF of replay on doc token-IE. Clean
  "published SoTA fails on doc-IE" result. See [[clue-lca-paper]]. `scripts/run_lca.sh`;
  SELF-MATCH-PROOF watcher
  (LCAGUARD_a94f) correctly fired when cil_cord freed the GPU. Output →
  `results/lca_eval/lca_run.log`. NOTE: LCA finetunes the FULL backbone (heavy, ~like der_pp).

## Active Work (2026-06-25) — DocMERGE method (NEW, implemented + queued to run)
**New CL method `doc_merge`** (committed 7185a6b + 35f5376 on `doccl`): diagnosis-guided
head-merging + drift-immune lexical memory. Born from a scientific-brainstorm + brutal
lit-check (NOVEL: head-only-merge-by-diagnosis, merge+KV-memory, BM25/TF-IDF-over-OCR
addressing, positive-BWT-via-LMC; framing rules + must-cite baselines MagMax/AMD-Proj
recorded in plan `~/.claude/plans/zazzy-brewing-popcorn.md`).
- **Files:** `doccl/methods/doc_merge.py` (method), `doccl/methods/head_merge.py`
  (plain/ties/fisher merge), `configs/method/doc_merge.yaml`, registered in
  `scripts/train.py`. Reuses HRP's `HybridPromptPool`/`sparse_doc_vectors`/routing.json
  + the Fisher displacement API (re-validates head-locus per run → `diag.json`).
- **Ablation axis:** `consolidate ∈ {merge, memory, both}` × `merge_rule ∈ {plain, ties,
  fisher}` — one method, config-toggled. `merge`=head-merge only (no memory read);
  `memory`=HRP-without-replay; `both`=full method. Backbone FROZEN (LMC holds → merge safe).
- **Tests GREEN:** 9 unit (`tests/methods/test_doc_merge.py`) + 6 e2e (3 modes × CIL/DIL).
  ruff + black clean. Fixed one real bug: CIL head-growth made the Fisher-displacement
  diagnostic mismatch widths (5 vs 7); fix = paired same-moment (fisher_old, params_old)
  snapshots per boundary.
- **Feasibility DONE (dil, seed42, 3ep) — ❌ NO-GO, head-merge hypothesis FALSIFIED.**
  Full bake-off finished 03:47 (results/docmerge_feas/<tag>/dil_*_seed42/). Table (AA / BWT
  / routing):
  | variant | AA | BWT | route |
  | der_pp (baseline) | **88.0** | -1.5 | — |
  | memory (HRP no-replay) | **21.9** | -10.2 | 0.915 |
  | merge_plain | 19.1 | -7.9 | — |
  | merge_ties | 21.6 | -5.4 | — |
  | merge_fisher | 14.3 | **-0.19** | — |
  | both_plain | 19.3 | -6.9 | 0.915 |
  | both_ties | 20.5 | -3.8 | 0.915 |
  | both_fisher | 12.9 | -0.41 | 0.915 |
  **Verdict against the criteria:** (1) `both` does NOT beat its parts — best `both` (ties
  20.5) < `memory` (21.9); flagship `both_fisher` (12.9) is the WORST. (2) BWT NEVER goes
  positive (best -0.19). (3) `fisher` HURTS AA (opposite of the bonus). Routing is fine
  (0.915, matches HRP).
  **RCA (3 layered root causes, confirmed via offline merge-arithmetic probe
  `scratchpad/rca_merge.py` + retention matrices):**
  1. **1/T SHRINKAGE BUG (implementation, fixable).** `merge_head_deltas` does a coordinate
     MEAN over all T task-deltas → every task's learned update is scaled by 1/T (=1/3),
     *even on rows the other tasks never touched* (proved: CIL disjoint-rows survival
     `[0.33,0.33,0.33]`, same as DIL). A head needing update |δ| gets |δ|/3 → systematic
     underfit on CIL *and* DIL. I used model-soup-style MEAN; task-arithmetic SUMS task
     vectors precisely to avoid this. → fix: sum / count-aware mean (full magnitude on a
     task's own rows).
  2. **DIL ROW-CONFLICT (structural, dil-specific).** dil = fixed label space → head does
     NOT grow → all tasks fine-tune the SAME rows toward different domain optima →
     averaging conflicting solutions CANCELS them: `||Σδ||/Σ||δ|| = 0.57`, merged-head norm
     28.5 vs 50 each task wants (43% too small). Averaging weights ≠ finding a jointly-good
     point when directions conflict — which is exactly why head-REPLAY (real gradients,
     HRP +5.7) worked and head-MERGE (arithmetic mean) didn't.
  3. **LOPSIDED FISHER (why `fisher` is worst).** task0 (FUNSD) converges first → dominant
     head-Fisher mass → fisher-merge ≈ 99% task0 head (`cos(merged,d0)=0.99` vs d1/d2
     ~0.09) → tasks 1-2 DROWNED (matrix: merge_fisher task1 F1=2.0). The "BWT~0" of
     fisher is a flat-line (it pins task0, never learns the rest), not transfer.
  **Net:** best merge (`ties`, which trims+sign-elects → dodges cancellation, |merged|/|δ|
  =1.27) only reaches AA 21.6 ≈ no-merge `memory` 21.9 → **merging adds NOTHING positive**;
  the LMC-shared-basin premise fails for per-task heads writing shared rows. Compounded by a
  hard UNDERFIT FLOOR (3ep frozen-backbone caps even at-learning F1 at ~21 FUNSD / ~9 SROIE
  / ~55 CORD vs der_pp 88/82/97).
  **Diagnostic caveat:** `diag.json` head/backbone ratio = `inf` for ALL variants — backbone
  is FROZEN so its displacement is 0 → ratio vacuous. The Step-A premise check as designed
  does NOT apply to a frozen-backbone method (real design oversight; would only be meaningful
  with a trainable backbone).
  **UPDATE (06:37): step (1) DONE + step (2) RUNNING on cil_cord.** Count-aware merge fix
  committed (f900089/amended + 0b5c927: `merge_count_aware` default True; +count-aware unit
  tests, 11 pass; `method.merge_count_aware` config knob wired). **KEY: proved the fix is a
  mathematical NO-OP on dil** (dense shared-label rows → divisor T either way; merged head
  byte-identical, `max|aware−classic|=0.000`). So the dil failure is purely structural
  (#2 row-conflict + #3 lopsided Fisher), NOT the 1/T bug. The fix only bites on DISJOINT
  rows → re-running on **cil_cord** instead (5 sessions, growing head = disjoint logit rows;
  RCA predicts merge *could* work there). `scripts/run_docmerge_cil.sh` RUNNING (Task 1/5,
  1.2GB VRAM): memory + merge×{plain,ties,fisher} + both_plain + a plain count_aware=false
  CONTROL. Decisive comparison: merge_plain(fixed) vs merge_plain_classic(pre-fix) =the fix
  effect; best-merge vs memory = does merge add anything on CIL. Output →
  `results/docmerge_cil/cil_feasibility.log`. ~1.5–2.5h.
  **RCA-INFORMED NEXT (original plan, for reference):**
  (1) ~~Fix the 1/T shrinkage → re-run~~ DONE (no-op on dil → running on cil_cord instead).
  (2) If merge still fails after the fix → **the negative is real and reportable**: "weight
     MERGING cannot consolidate a shared classifier head across conflicting domain optima;
     gradient REPLAY can (HRP +5.7)" — a clean mechanism contrast for the thesis.
  (3) **Strongest salvage:** drop merge, restore HRP head-REPLAY as DocMERGE's consolidation
     (replay already gave AA 31.7 / BWT +5.7 on this exact setting). Merge was the wrong swap.
  (4) Orthogonal: the UNDERFIT FLOOR (~21 at-learning) caps everything → higher epochs
     (10–15) and/or trainable backbone would lift the floor AND make the diag non-vacuous;
     but that doesn't rescue the merge MECHANISM, only the operating point.
  Code/tests stay valid (method runs; the merge idea loses). Saga (Hydra epochs bug fixed
  c33d899; self-matching-pgrep watcher bug; BROS grid race) recorded below for completeness.
  - Saga: first watcher launch (18:47) crashed on `training.epochs` Hydra error (fixed
    c33d899 → `method.epochs`). Re-armed watcher had a SELF-MATCH bug (`pgrep -f
    "doccl.pilot.run_pilot"` matched its OWN bash body). Moot: user stopped the BROS grid;
    I launched the bake-off DIRECTLY after confirming GPU idle + retiring the broken watcher.
- **NEXT (decide):** the clean negative is itself reportable (could be a "what doesn't work"
  contrast: replay grounds the head, merging can't). To salvage the METHOD: (a) re-run at
  higher epochs (10–15) — does the merge stop underfitting? (b) drop merge, keep the
  HRP head-REPLAY (which DID give +5.7) as the consolidation; merge was the wrong swap.
  (c) try a TRAINABLE backbone so LMC/merge has a real basin + the diag isn't vacuous.
  ⚠️ BROS BASELINE grid belongs on A6000/L40, NOT local (per below) — do NOT auto-launch here.

## Active Work (2026-06-25) — backbone-generalization of the forgetting analysis
**Task: make the forgetting analysis comprehensive across ALL 4 backbones**
(LayoutLMv3 primary + BERT + LiLT + BROS). Found it was NOT: the pilot localizer
ran only LayoutLMv3+BERT; the grid had LiLT=24/BERT=15/**BROS=0** runs; and
analyze_results silently merged LiLT runs into LayoutLMv3 rows. All three fixed:

1. **Analysis fix (commit db07d58).** `analyze_results.py` now carries `model_family`
   as a real column; main/ablation/compute/forgetting/baseline tables scoped to the
   primary backbone (`_primary_only`, keeps the bert_textonly row); new
   `write_backbone_table` → `table_backbone_<metric>.tex` (LayoutLMv3 reference +
   LiLT/BROS). First result: replay-dominant ordering (ER/DER++ ≫ EWC > Naive)
   **transfers from LayoutLMv3 to LiLT** on dil/mixed. 4 tests (test_backbone_analysis).

2. **Pilot extension (commit 49dda2f).** Added `cl_lilt` + `cr_bros` conditions to
   run_pilot (CKA+Fisher+displacement). LiLT uses the English checkpoint
   (`SCUT-DLVCLab/lilt-roberta-en-base`) with the `roberta-base` tokenizer (AutoTokenizer
   resolves the English LiLT repo to a box-demanding LayoutLMv3 tokenizer; RoBERTa vocab
   is identical+box-free, and XLM-R's 250k embedding OOMs the 2060). Fixes:
   `add_prefix_space=True` in encoders (RoBERTa/XLM-R need it); CKA hook unwraps nested
   tuples (LiLT two-stream layers); gradient-ckpt warn+continue for BROS (commit d3388d4).
   **Smoke-validated on the 2060 (1ep): head-dominant forgetting REPRODUCED on BOTH** —
   LiLT final-Fisher classifier=0.034 vs others ≤0.005, displacement at head only;
   BROS displacement at head=4e-7, all other depths 0.

3. **RUNNING NOW (detached, local 2060):** full pilot sweep `cl_lilt`+`cr_bros` × seeds
   {42,123,7} × 10ep (matches existing LayoutLMv3/BERT pilot for comparability).
   Runner: `scratchpad/run_pilot_secondary.sh`; resume-safe (skips existing
   `results/pilot/<cond>_seed<seed>.json`). ~4h. seed42+seed123 DONE. The sweep spawns a
   FRESH PID per (cond,seed) run — don't track a single PID to detect "pilot done".
   ⚠️ **DocMERGE watcher near-miss:** the docmerge feasibility watcher tracked only the
   FIRST pilot PID (1890344=seed42); when that exited it fired at 18:47 while THIS sweep
   was still running seed123 — but its run crashed instantly on a Hydra error
   (`training.epochs` not overridable — needs `+training.epochs=`) before any dataset/GPU
   use, so NO collision. docmerge feasibility is BROKEN+needs that config fix; rc=0 was a
   swallowed error. No watcher now lingers.

**Pilot sweep RESULTS (5/6 done, all head-localized — generalization CONFIRMED):**
| cond  | seed | AA   | BWT   | disp head/non-head |
| cl_lilt | 42  | 27.4 | -86.8 | 67x |
| cl_lilt | 123 | 26.8 | -86.6 | 52x |
| cl_lilt | 7   | 27.2 | -86.8 | head leads (seed7 non-head higher) |
| cr_bros | 42  | 28.5 | -89.6 | 33x |
| cr_bros | 123 | 29.2 | -88.5 | 201x |
| cr_bros | 7   | 28.4 | -88.6 | 104x |
**ALL 6/6 PILOT DONE.** BROS CKA depth-gradient (seed42, FUNSD→CORD): 0.99 emb → 0.95 L0
→ 0.61 L6 → 0.15 L11 → 0.11 head — clean monotonic drift-with-depth, max at head.
Cross-backbone displacement-by-depth (mean over seeds, head vs max-non-head):
c4_full 1.6e-5/1e-7, cb_bert 7e-6/0, cl_lilt 2.6e-4/9.2e-6, cr_bros 5.4e-5/1e-7 — ALL
head-dominant. Pilot-analyze location test: cl_lilt p=0.38, cr_bros p=0.51 vs c4_full
(NOT different → same locus). Head/depth-dominant forgetting REPRODUCES on LiLT+BROS =
diagnosis is NOT architecture-specific. Pilot figures regenerated (7 conditions, 22:09).

**SCOPE (corrected): this session = forgetting ANALYSIS + PLOTS across backbones to
confirm the head-locus hypothesis. NOT baselines.** Baseline metric grids (AA/BWT
comparison runs for LiLT/BROS) belong on the A6000/L40 — NOT the local box. A BROS
baseline grid was mistakenly started here, then STOPPED and its partial dir removed.

**DONE — forgetting analysis COMPLETE for all backbones + FOLDED INTO THESIS:**
- 6/6 pilot localizer runs (LiLT + BROS × 3 seeds) finished.
- 4 diagnostic figures regenerated with all 7 conditions, ingested into thesis/figures/.
- Hypothesis CONFIRMED: Fisher-weighted displacement head-dominant on every backbone
  (head/non-head ratio: BERT 1589×, LayoutLMv3 masks 246–9368×, LiLT 28×, BROS 699×);
  CKA monotonic depth-gradient max at head on all 4; location permutation test p=0.38
  (LiLT), 0.51 (BROS) vs c4_full → SAME locus. Architecture-general.
- Honest caveats recorded: |BWT| magnitude test underpowered at n=3 (claim shared
  LOCATION not amount); BROS position-embeddings are a 2nd mobility/importance locus
  (head-only necessary-but-not-fully-sufficient on BROS); use canonical-order n=3
  c4_full (AA 28.3 / BWT −89.8), NOT the n=6 pool that mixes _ord210 reverse-order runs.
- THESIS UPDATED (commits d93ecbd + acda6d4, pushed): Ch6 sec:results-layer new
  'across architectures' paragraph + 4-backbone tab:bert-contrast + refreshed
  tab:cka-layer/captions; Ch6 limitations (iv)/(vi) reframed (perm test now reported,
  diagnosis on 4 backbones, only full CL benchmark stays LayoutLMv3-scoped); Ch7
  limitation reframed. Builds clean (XeLaTeX, 115pp, no undefined refs).

**Note:** `analyze_results.py` backbone-table fix (model_family column + table_backbone_*)
is committed and will surface LiLT/BROS metrics WHEN those baseline runs land on the
A6000/L40 — but running those baselines is a LATER, off-box task, not this session.
The other session's docmerge watcher is self-deadlocked (`while pgrep -f
"doccl.pilot.run_pilot"` matches its own cmdline) — their bug to fix, harmless here.

**NEXT (off-box / later):**
- Run LiLT+BROS baseline metric grids on A6000/L40 → then re-run analyze_results so
  table_backbone_*.tex gains the BROS column (LiLT already there).
- Update thesis Ch6 limitation (vi): reframe LiLT/BROS forgetting-localization from
  "planned" to DONE (the analysis/plots in this session); keep baseline-table cells
  pending until the off-box runs land.

## Prior Active Work
**HRP feasibility method implemented (2026-06-24, commit b047672).** New CL method
`hrp` (`doccl/methods/hybrid_routed_prompt.py`): L2P-family, frozen backbone, task-pinned
prompt pool with a **hybrid dense+sparse router** — sparse = BM25/TF-IDF over OCR tokens
(`batch["input_ids"]`), fused with the dense CLS query by RRF. `method.router ∈
{dense,sparse,hybrid}` is the routing ablation in one method; emits
`results/<run>/routing.json` (per-task routing hit-rate = the diagnostic claim). 6 unit
tests + 2 e2e lifecycle tests pass. **Reframed the user's idea**: NOT "no forgetting" (it
moves into the router) and NOT acceleration (≤40 slots) — the contribution is *measured
routing accuracy*. Spec: `docs/superpowers/specs/2026-06-24-hybrid-routed-prompt-design.md`.
- **One-command runner:** `scripts/run_hrp_feasibility.sh` (GPU/dep check + 3-router
  ablation + der_pp/doccl + prints routing hit-rate & AA/BWT). Small-VRAM recipe:
  `GRAD_CKPT=1 BATCH_SIZE=2 NUM_WORKERS=0`.
- **BUG FOUND + FIXED (commit 518e5bb).** First L40 run gave AA~18 on dil (diagonal
  [8,0,47]): free top-k routing while keys were random → the active task's own block was
  rarely selected → its prompts never trained (L2P cold-start collapse). Fix: **task-pin
  every doc to the active block during TRAINING** (DualPrompt recipe; trains the block +
  pulls its dense key toward the task's queries); route freely only at EVAL where the
  hit-rate is measured. routing.json now writes (added writer regression test). 8 tests
  pass.
- **FEASIBILITY RESULT (dil, seed42, 3ep, local 2060) — IDEA VALIDATED + reframed.**
  Final per-task routing hit-rate (router sends doc to its own task's block):
  | router | task0 | task1 | task2 | overall | AA |
  | dense  | 0.00  | 0.13  | 1.00  | **0.294** | 20.81 |
  | sparse | 0.74  | 0.92  | 1.00  | **0.915** | 20.87 |
  | hybrid | 0.00  | 0.91  | 1.00  | **0.839** | 20.82 |
  **Sparse OCR-token routing = 3.1× better than dense (0.29→0.92).** Dense suffers total
  recency collapse (routes everything to the last-trained block); sparse signatures are
  accumulated+frozen per task → no recency bias → recovers task0/task1. The core idea
  works. **BUT AA is flat (~20.8) for both** — better routing did NOT lift accuracy
  because the bottleneck is **head drift** ([[clue-prompt-family-head-drift]]), not
  routing: correct prompts feed an already-drifted shared classifier head. Routing is
  necessary but not sufficient.
- **SECOND FINDING — naive RRF hybrid (0.839) < pure sparse (0.915), task0 back to 0.00.**
  Fixed 50/50 Reciprocal Rank Fusion lets the *recency-collapsed* dense signal drag the
  strong sparse signal down. RRF assumes both rankers are individually useful; here dense
  is actively harmful. Full method should use **confidence-weighted / learned fusion**
  (down-weight dense when its keys collapse), not fixed RRF — or just use sparse routing.
- **HEAD-REPLAY ADDED (commit 78c1421) → IT WORKS.** A/B on dil (sparse router both,
  ONLY head_replay_weight differs): a small reservoir of past examples, each pinned to
  its OWN frozen block, with a CE loss to ground the shared head.
  | variant | task0 | task1 | task2 | AA | BWT |
  | sparse, head-replay OFF (w=0) | 6.97 | 3.67 | 51.81 | **20.87** | -10.12 |
  | sparse, head-replay ON  (w=1) | 29.52| 11.72| 53.79 | **31.68** | **+5.65** |
  **AA +10.8 (+52%); BWT goes POSITIVE (-10.1→+5.7).** Head protection stops the task-0
  collapse (21→0.8 became 21→29.5) — old tasks now *improve* as new ones train. This is
  the full thesis realised: routing keeps the right PROMPTS firing; head-replay keeps the
  HEAD stable; together → no forgetting (matches the user's original "no forgetting"
  intuition, but via the correct mechanism). 9 unit + 2 e2e tests pass. w=0 reduces
  exactly to the router-only loop (ablation-safe).
- **CONTROL CONFIRMED:** w=0 run reproduced AA=20.87 BWT=-10.12 *byte-identical* to the
  original sparse run → the +10.8 AA / +15.8 BWT gain is causally head-replay, not noise
  (same router/data/seed, only the weight toggled). Clean controlled ablation.
- **STILL TO DO:** (1) ~~confirm w=0 control~~ DONE. (2) absolute
  AA still well below replay (der_pp ~88) — head-replay at w=1/buffer200/3ep is a first
  cut; tune weight/buffer/epochs. (3) confirm on **cil_cord** (the routing stress test).
  (4) fix the RRF fusion (sparse-only currently best). (5) the win is the *mechanism +
  positive BWT*; for A* it needs to close more of the gap to replay or win on a metric
  replay can't (param-efficiency: frozen backbone, tiny prompt+head footprint).

---

**Grid harvested (167 runs) → thesis updated (2026-06-18).** Full R2 pull = 167 completed runs
across **9 scenarios** (incl. new `cil_wildreceipt`, `dil_xlingual`) × ~15 methods × 3 seeds.
Regenerated the whole analysis pipeline and updated thesis Ch.1/6/7 with measured numbers,
new figures, and a new Extended-Scenarios section. Thesis builds clean (113 pp).

### Key measured findings (now in thesis)
- DIL oracle = **88.7** (was 85.1). DocCL DIL 84.7 **closes most of the gap but does NOT reach
  the oracle**; replay (ER 87.9 / DER++ 88.0) matches it. DocCL competitive-with-but-below replay.
- Pilot: head displacement ~1e-4 vs late ~1e-8 (≈4 orders); **BERT (cb_bert) shows the same
  head-dominant pattern** → output-side forgetting is NOT vision-specific (BERT contrast now
  real: AA 29.9, BWT -70; previously thought degenerate/excluded).
- **XFUND cross-lingual = headline new result**: near-ZERO forgetting under pure language shift
  (naive BWT only -6.6) — fixed label space ⇒ head needn't relearn ⇒ minimal forgetting.
  Strong corroboration of the output-side/head-locus thesis.
- **WildReceipt (scale)**: severe forgetting (BWT ≈-85), replay can't rescue the growing head.

### Genuinely unmeasured (flagged honestly, NOT fabricated)
- `er_cflat` / C-Flat++ runs FAILED (no metrics.json) → "not available".
- DocCL ablation variants (head_only/late_only/uniform) did NOT complete → future work.
- DocCL / 2025 / prompt-LoRA NOT run on the two new scenarios (compute) → classical + oracle only.

### Generators widened (committed)
analyze_results (5 scenarios + full method set + per-scenario ablation/compute), build_result_figures
(full method set + bench_newscenarios_aa), pilot/analyze (drops all-zero dead runs), ingest_to_thesis
(new pilot figs + single-task baselines incl WildReceipt/XFUND).

### bd tracker: local DONE, dolt-REMOTE push blocked (environment)
bd issues are updated LOCALLY: **CLUE-4ry CLOSED** (this task), **CLUE-0cv FILED** (LiLT bug:
`LiLTWrapper` padding_idx IndexError in tests/test_backbones.py forward/expand + prompt-injection,
'index out of range in self', embedding padding_idx > num_embeddings — pre-existing, fix before LiLT
secondary-backbone runs). Mutations were slow (a stray `dolt sql-server` PID 942698, up 8+ days, on
127.0.0.1:46149 holds the embeddeddolt lock; `dolt_mode: embedded` contends with it) but they DID
commit to the embedded DB. **`bd dolt push` times out** against that lock, so the bd remote is NOT
synced. TODO for user: stop that dolt server (or `bd init --server`), then `bd dolt push`. See
[[clue-bd-dolt-lock]]. All CODE + THESIS work is committed and **git pushed** (origin/doccl up to
date); only the beads dolt-remote sync is pending.

**NEW datasets added (infrastructure only — grid runs LATER on the powerful machine):**
- **XFUND** (`nnul/xfund-multilingual`) → cross-lingual **domain-IL** scenario `dil_xlingual`
  (de→es→fr→it→zh, fixed label space → pure language-shift drift). The headline new experiment:
  does output-side forgetting hold under language shift?
- **WildReceipt** (`kaydee/wildreceipt`) → **class-IL** scenario `cil_wildreceipt` (24 classes,
  49 BIO tags, 4 growing-head sessions). Adds scale + richer receipt schema.
- Both loaders validated against real HF schemas; 19 fast tests pass. ~66 new grid runs pending.

## Last Decision
- Train-to-convergence via **val-F1 early stopping** (was fixed 3ep — under-trained EWC).
- New datasets: XFUND=domain-IL (language axis), WildReceipt=class-IL (scale axis). Task-IL stays
  deferred. Run the new ~66 runs on the powerful machine (docker/), not the laptop.
- **num_workers=0** mandatory: num_workers=4 forked the dataset into ~5×10 GB and OOM-crashed
  the machine. See memory [[clue-resource-limits]].
- Hard ceilings: RAM < 14 GB, VRAM < 5 GB. Watchdog hard-aborts if crossed.

## Fixed this session (13 bugs/issues, all committed+pushed)
1. c4b9ff9 — class-IL label filter: mask out-of-session tags (was: issubset → 0 examples).
2. ba87040 — dataset RAM: lazy image decode + shared HF handle per split (11 GB → 3 GB).
3. f4c61d1 — CIL label remapper: native ids → head-index space (was: CUDA assert).
4. 3ab7008 — expand_classifier: handle both LayoutLMv3 head types.
6. cc5fc7c — EWC penalty: slice to old head shape (was [25]vs[13] crash after expansion).
7. cc5fc7c — DER++ buffer: pad mixed-width cached logits before stack (was [512,13]vs[512,25] crash).
5. 358385e — force Linear head: MLP head (≥10 labels) collapsed to all-O after CIL expansion (tanh saturation killed new-class gradients). Was producing F1=0 on every CIL task after task 0. Fixed + validated (task1 F1=53 post-expand).
8. 2d984cb — Joint trained on CIL-MASKED per-session datasets concatenated: same receipt appeared 3x with conflicting labels (menu labeled in one copy, masked to O in another) -> Joint learned only task 0 (F1=[83,0,0,0,0], AA=16.6 ~= naive). Fixed: Joint now trains on a full-label dedup pool (joint_train_datasets) where each doc appears ONCE with ALL labels. DIL falls back (disjoint docs). Validated: joint learns all 5 tasks now.
Validated: cil_cord_naive 1ep reached task0 F1=88.24, expanded head, advanced to task 2 — clean.

## EWC fix saga (now resolved, validated 3 tasks GPU)
EWC crashed 3 ways, each fixed: (a) Fisher penalty p vs theta_star shape [cc5fc7c], (b) OOM without
checkpointing — reverted to ckpt ON [45fd650], (c) Fisher ACCUMULATION in after_task summed 13+25
across head growth + penalty fisher_val mismatch [6eb903b: pad old fisher to new shape, slice all 3
to common min]. Verified EWC runs 3 tasks (head 13->25->37) at 4.7GB VRAM. LwF next — may OOM (frozen
teacher = 2nd model); will GPU-smoke-test before trusting.

## LwF fix (validated before grid reached it)
LwF teacher (deepcopy of model) had a device mismatch: position_ids buffer left on wrong device
after deepcopy+expand. Fixed [85f1d53]: teacher.to(self.device) each train_task + disable its
checkpointing. GPU teacher fits at 3GB (no OOM — the CPU-teacher detour was unneeded). Validated
2 tasks, KD active. er/der_pp next (lighter, replay; der buffer fix already in).

## Method-fix loading
Grid runs naive->joint->ewc->lwf->er->der_pp. EWC+DER fixes are in the tree; the driver spawns each train.py fresh, so ewc/der_pp runs (hours away) load the fixed code — no restart needed. joint/er/lwf already safe.

## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Blocked On
Nothing — fully autonomous. PHASE 5 (doccl) uses DocCL_A placeholder (user accepted re-point risk).
