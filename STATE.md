# STATE

## Active Work
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
- **STILL TO DO:** (1) confirm w=0 control reproduces ~20.8 (in progress). (2) absolute
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
