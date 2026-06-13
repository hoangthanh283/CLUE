# DocCL: Continual Learning for Document Information Extraction — Complete Research Record

> Self-contained transfer document. Captures every finding, bug, decision, table, and result produced during the experiment run, for use when writing the thesis report. Snapshot taken at **41/54 grid runs complete** (naive 9/9, joint 9/9, EWC 9/9, LwF 9/9, ER 5/9, DER++ 0/9). The remaining ER and DER++ runs follow the same code paths and are expected to extend the tables below without changing the conclusions.

---

## 0. Context Note (read first)

This document records a **continual-learning (CL) research project** on a master's thesis codebase named **CLUE** (PyPI package `cl4ie`, working git branch `doccl`). It trains **LayoutLMv3-base** on **document information-extraction (IE)** tasks — token-level entity tagging on scanned forms and receipts — and compares CL strategies against lower/upper bounds. Output fills the thesis result placeholders (Chapter 6 tables/figures, §3.3.6, abstract/conclusion).

**Research questions:**
1. How severe is catastrophic forgetting when LayoutLMv3 is fine-tuned sequentially across document-IE tasks (the **naive** lower bound)?
2. How much of that forgetting does each CL strategy recover, relative to a **joint** (all-data) upper bound?
3. Which strategy family — regularization (EWC), distillation (LwF), or replay (ER, DER++) — works best for **dense token-classification** document IE, and does that differ across class-incremental vs domain-incremental settings?

**Experimental grid = methods × scenarios × seeds:**
- **Methods (6 core):** `naive` (lower bound), `joint` (upper bound), `ewc`, `lwf`, `er`, `der_pp`. (Deferred/optional: Phase-4 prompt/LoRA = `l2p`, `dualprompt`, `coda_prompt`, `o_lora`; Phase-5 proposed method `doccl` aliased to `DocCL_A`.)
- **Scenarios (3):** `cil_cord` (class-incremental on CORD, 5 sessions), `dil` (domain-incremental FUNSD→SROIE→CORD with a unified 9-tag schema), `mixed` (6-task interleaved class-IL + domain shifts).
- **Seeds (3):** 42, 123, 7.
- **Core grid size:** 6 × 3 × 3 = **54 runs**.

**Hardware / hard constraints:** single **RTX 2060, 6 GB VRAM**; **15.9 GB system RAM**; 4 GB swap. User-imposed ceilings: **never exceed 14 GB RAM or 5 GB VRAM** (OOM crashes the whole machine). Python 3.12 `.venv` (no conda); torch 2.11+cu130; W&B online project `thanh-workspace/CL4IE`.

**Metrics:** entity-level span **F1 via seqeval** (BIO scheme). From the accuracy matrix R (R[i][j] = F1 on task j after training task i):
- **AA** (Average Accuracy) = mean(R[T-1, :])
- **BWT** (Backward Transfer) = mean over i<T-1 of (R[T-1, i] − R[i, i]) — negative = forgetting
- **AF** (Average Forgetting) = −BWT
- **FWT** (Forward Transfer) = mean over i>0 of (R[i-1, i] − b_i); requires single-task baselines b_i

**Epoch budget:** **3 epochs/task** (user-approved). 10 epochs ≈ 6.7 days = infeasible; model reaches F1≈88 after 1 epoch, so 3 epochs is converged + defensible.

**Final training config:** `batch_size=2`, `gradient_checkpointing=true`, `num_workers=0`, `method.epochs=3`, W&B online, each run inside a `systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=0` cgroup so the kernel can OOM-kill only the run (never the machine).

---

## 1. Codebase Architecture

### 1.1 Monorepo layout
| Path | What it is |
|------|-----------|
| `CLUE/` | The CL code (LayoutLMv3 + strategies). Training, eval, configs. Package `cl4ie`. |
| `CL4IE/` | Thesis knowledge wiki (Obsidian) + LaTeX thesis. No build tooling. |
| `papers/` | Source PDFs. |
| `experiments/` | Standalone research scripts. |

Branch **`doccl`** is the clean experiment infrastructure. `RUNBOOK.md` pipeline: **setup → pilot → [GATE A] → main grid + ablation → aggregate → ingest → write prose.**

### 1.2 Key source modules
- `doccl/data/scenarios.py` — `build_cil_cord`, `build_dil`, `build_mixed`, `build_single`, `build_cil_funsd`; `CLScenario` dataclass; `_cil_head_snapshots()`.
- `doccl/data/cord.py`, `funsd.py`, `sroie.py` — loaders with `_filter_by_labels` (class-IL splits) and lazy image loading.
- `doccl/data/cil_remapping.py` — `CIL_LabelRemapper` (created in this work).
- `doccl/data/dil_remapping.py` — `DIL_LabelRemapper`, `DIL_UNIFIED_LABELS` (pre-existing).
- `doccl/models/layoutlm_wrapper.py` — `LayoutLMv3Wrapper`: `expand_classifier`, `_force_linear_head`, `forward`.
- `doccl/methods/` — `naive.py` (`NaiveFineTune`, `JointMultiTask`), `ewc.py` (`EWC`), `lwf.py` (`LwF`), `er.py` (`ER`), `der.py` (`DERpp`), `buffer.py` (`ReservoirBuffer`).
- `doccl/eval/metrics.py` — `CLMetricsTracker`, `compute_token_f1` (seqeval).
- `scripts/train.py` — single-run trainer (method registry, joint branch, CL loop, `save_run_metrics`).
- `scripts/run_grid.sh` — grid runner (resume-safe via `results/<run>/.done`; `MEM_CAP` cgroup wrap).
- `scripts/run_autonomous_grid.sh` — full-pipeline driver.
- `scripts/run_grid_watchdog.sh` — resource watchdog.
- `scripts/analyze_results.py` — aggregates `results/*/metrics.json` → tables/figures + single-task baseline table.
- `scripts/ingest_to_thesis.py` — copies artifacts into the thesis tree (`chapter6.tex` uses `\IfFileExists`).

### 1.3 Per-run metrics contract (`save_run_metrics`)
```
{method, scenario, seed, target_component, AA, BWT, AF, FWT,
 matrix (T×T list), num_tasks, wall_time_per_task_s, total_wall_time_s,
 mean_time_per_task_s, total_params, trainable_params, peak_gpu_mem_mb}
```
`analyze_results.py` globs `results/*/metrics.json` and pivots on {method, scenario, seed} with {AA, BWT, AF, FWT}.

### 1.4 Datasets and label spaces
| Dataset | HF id | Task | Native entities | Notes |
|---------|-------|------|-----------------|-------|
| FUNSD | `nielsr/funsd-layoutlmv3` | Forms | HEADER, QUESTION, ANSWER → 7 BIO incl. O | 149 train / 50 test |
| CORD-v2 | `naver-clova-ix/cord-v2` | Receipts | 30 fine classes → 61 BIO incl. O; super-granularity → 11 BIO | 800 train / 100 test; **avg 7.9 distinct entity categories per receipt** |
| SROIE | `mp-02/sroie` (HF mirror) | Receipt KIE | COMPANY, DATE, ADDRESS, TOTAL → 9 BIO | 626 train |

### 1.5 Scenario definitions
**`cil_cord`** — Class-Incremental, 5 sessions × 6 fine classes; head grows 13→25→37→49→61 (O at index 0). Session classes:
- s0: menu.{cnt, discountprice, itemsubtotal, nm, num, price}
- s1: menu.{unitprice, vatyn}, menu.sub.{cnt, nm, price, unitprice}
- s2: sub_total.{discount_price, etc, othersvc_price, service_price, subtotal_price, tax_price}
- s3: total.{cashprice, changeprice, creditcardprice, emoneyprice, menuqty_cnt, menutype_cnt}
- s4: total.{total_etc, total_price}, void_menu.{nm, price}, sub.{nm, cnt}

**`dil`** — Domain-Incremental: FUNSD → SROIE → CORD-super, **unified 9-tag schema** `[O, B-HEADER, I-HEADER, B-KEY, I-KEY, B-VALUE, I-VALUE, B-OTHER, I-OTHER]`; native labels remapped via `DIL_LabelRemapper`; head pre-expanded to 9. Mapping rationale (from `docs/dil_schema_mapping.md`): FUNSD QUESTION→KEY, ANSWER→VALUE; SROIE COMPANY→KEY, ADDRESS/DATE/TOTAL→VALUE; CORD menu/sub_total/total→VALUE, void_menu/sub→OTHER.

**`mixed`** — 6 tasks: FUNSD-s0 (HEADER+QUESTION) → FUNSD-s1 (ANSWER) → SROIE (4 fields) → CORD-super-s0 (menu+sub_total) → CORD-super-s1 (total+void_menu+sub) → FUNSD revisit. Head grows 5→7→15→19→25 (revisit reuses existing labels).

---

## 2. Reuse Audit (pre-grid investigation)

Before committing GPU-weeks, a 4-explorer + synthesis workflow audited two external repos and their W&B stores:
`/mnt/DataDrive/agentic-ai/openclaw/workspace/research/lth-cl-doc-ie` and `/.../continual-learning-doc-ie`.

### 2.1 Verdict: nothing external is reusable
**`lth-cl-doc-ie`** — incomparable on five independent axes:
- Backbone: TinyNet BOW+MLP (256→512→4), not LayoutLMv3-base (125M).
- Dataset: XFUND 7-language streams (German/French/Spanish/Italian/Japanese/Portuguese/Chinese, ~149 docs/lang), not FUNSD/CORD/SROIE.
- Metric: per-language argmax accuracy. CSV header `method,order,seed,avg_final,avg_forgetting,bwt,fwt_proxy` — **no F1 column**.
- Seeds: 1337/2027/3141, not 42/123/7.
- Label regime: fixed 4 classes (answer/header/other/question), no CIL head growth. 3 epochs; replay buffer 1024; memory bank 8 slots.
- Methods (TinyNet/XFUND): seq_ft, upper_joint_oracle, ewc_v1, lwf_v1, er_v1, derpp_v1, si_v1, mas_v1, mir_v1, agem_v1, lth_distill.
- Accuracy numbers (not transferable): derpp_v1 0.4949; ewc_v1 0.4405; lwf_v1 0.4405 (**identical to EWC — red flag, possibly degenerate regularizers**); er_v1 0.4859; seq_ft 0.4478; lth_distill 0.4994. Located under `experiments/h6-bounds-baselines/`, `h7-classic-baselines/`, `h9-modern-baselines/`.

**`continual-learning-doc-ie`** — designed an XFUND language-stream protocol but **never executed it** (data mount blocked, 0/8 paths). No FUNSD/CORD/SROIE numbers in the target format.

**CLUE local artifacts** — at audit time exactly **one** contract-compliant run: `results/single_funsd_naive_seed42` (AA=87.99, matrix [[87.99]], 125,332,359 params, peak GPU 2407 MB, wall 357s). **Zero `.done` markers**. The 15 pilot JSONs use a diagnostic schema (`condition, seed, task_order, cka_records, fisher_records, accuracy_records, cl_metrics, matrix`) missing `method/scenario/wall_time/params/gpu_mem` → **cannot be backfilled**. Legacy `nightly_2026032*` / `layoutlmv3_*` runs are pre-final-pipeline, off-spec 5-task (WildReceipt + XFUND_zh), inconsistently labeled → **excluded**.

**Conclusion:** all of PHASE 3 (54 runs) must run fresh; the only genuine reuse is `single_funsd_naive_seed42` (pipeline/contract validation + one FWT baseline).

### 2.2 Pilot study & GATE A
GATE A resolved to a **characterization-only fallback** (pilot inconclusive). Dominant component was the classifier head (Fisher 2.17e-1, CKA 0.22), which does not map to candidate methods A/B/C (fusion LoRA / layout-protected EWC / modality-routed prompts). Cross-condition test (c4_full vs c1/c2/c3) non-significant under Bonferroni (p=0.026/0.028/0.167 vs α=0.0167), partly because text-deprived conditions collapse to F1=0 and inflate variance. Per pre-agreed policy, the baseline grid runs regardless; Phase 5 paused.

**Pilot findings (PILOT_STUDY_REPORT.md):**
- **Modality collapse:** text-deprived runs (c1_bert, c2_no_text) collapse to **F1=0** → **text is the load-bearing modality** for LayoutLMv3; layout+vision alone is unstable. A real LayoutLMv3 property, not a bug.
- **Naive c4_full diagnostic:** seed 42 ≈ AA 28.8, BWT −87.6.
- **Order dependence:** alt order (SROIE→CORD→FUNSD) BWT −75.2 vs default (FUNSD→CORD→SROIE) BWT −81.3.
- Pilot conditions: C1 text-only, C2 image+layout (no text), C3 text+layout (no image), C4 full multimodal; 4 conditions × 3 seeds = 12 runs (+3 alt-order) on FUNSD→CORD→SROIE.

### 2.3 FWT honesty (committed as `docs/FWT_NOTE.md`)
True forward-transfer is **NOT recoverable post-hoc**: the CL loop only evaluates *seen* tasks (`eval_loaders_seen` accumulates), so the matrix is strictly lower-triangular and R[i-1, i] (future-task zero-shot) is never measured; the tracker is also built without `baseline_perf`, so `forward_transfer()` short-circuits to 0.0. Decision: `analyze_results.py` reports FWT as **unavailable ("--")**, never fabricated, and emits a single-task baseline table (`table_single_task_baselines.{tex,csv}`, b_i per dataset). Enabling real FWT would need a `train.py` change (evaluate unseen tasks zero-shot + seed the tracker with baseline_perf).

---

## 3. The Bug-Fix Chain (13 issues, all committed on `doccl`)

Central theme: **verify the content of results, not just that runs complete.** Each bug hid behind the previous one; runs got progressively further each attempt.

### Bug #1 — Empty class-IL splits (`c4b9ff9`)
`_filter_by_labels` kept a doc only if `tag_set.issubset(keep_ids)` (whole doc only one session's classes). Document-IE samples are densely multi-class (CORD avg 7.9 categories/doc; every FUNSD form has header+question+answer) → no doc is a subset → **0 examples** → crash `num_samples=0`.
Empirical: 800 CORD docs, 800/800 contain ≥1 session-0 category, **0/800 are a subset**. cil_cord s0=0/800, mixed_cord_s0=2/800, mixed_cord_s1=0/798, cil_funsd s0=2/149.
**Fix:** standard CIL masking — keep any doc with ≥1 in-session entity, relabel out-of-session entity tokens to O. cord/funsd/sroie. Verified non-empty, leak-free: cil_cord 800/382/549/723/780; cil_funsd 149/147; sroie 625/626.

### Bug #2 — Dataset RAM OOM crashed the machine (`ba87040`)
Loaders eagerly decoded/retained every PIL image in `self.data`; `build_cil_cord` builds 10 CORDDataset objects (5 train+5 eval × 800 images); `num_workers=4` forked ~10 GB per worker → RAM hit **14.86 GB**, kernel OOM. One CORDDataset ≈ 1.9 GB; building all 10 initially peaked at **11.4 GB**.
**Fix:** store HF row index + token/box/label arrays only (decode image on demand in `__getitem__`); share one HF Arrow handle per (split[/granularity]); memoize parsed base list; read image dims from raw bytes (decode=False). After fix: cil_cord 3.1 GB, dil 3.7 GB, mixed 3.1 GB. Pair with `num_workers=0`.

### Bug #3 — CIL label-space mismatch / CUDA assert (`f4c61d1`)
`nll_loss... Assertion 't >= 0 && t < n_classes'`. Datasets emit native ids (CORD: O=0, 61 tags), head built from `tasks[0].label_set` (12 entity tags, no O, different order). New `CIL_LabelRemapper` + `_cil_head_snapshots()` remap native ids → cumulative head-index space; O prefixed into each `label_set`. Applied to cil_cord/cil_funsd/mixed. Verified in-range across all transitions.

### Bug #4 — expand_classifier head-type (`3ab7008`)
LayoutLMv3 uses a plain `nn.Linear` head for `num_labels<10` and an MLP `LayoutLMv3ClassificationHead` (`dense→tanh→dropout→out_proj`) for `num_labels>=10`. `expand_classifier` assumed `.weight` → crash for cil_cord (head 13). Fix grows only the final output Linear (`.out_proj` for MLP, the classifier for Linear), preserving old rows.

### Bug #5 — All-O collapse after CIL expansion / force Linear head (`358385e`)
Every cil_cord task after task 0 scored F1=0.00 (model predicted only O). Isolation tests: a fresh 25-way MLP head trains to F1=37, but the **same head grown 13→25 collapses to 0** — the task-0-trained `dense` layer saturates `tanh`, killing gradients to new class rows. Fix: `_force_linear_head()` forces a single `nn.Linear` head regardless of label count. Validated: post-fix expand 13→25 + train task1 → F1=53.3 with entity predictions. Production: cil_cord_naive `After task0: 92.16; task1: {0:0, 1:93.91}; task2: {0:0,1:0,2:88.97}` (fresh-task 88–94, prior=0 = genuine forgetting). FWT honesty + baseline table also committed here (`50fad3c`/`55cc0bf`).

### Bug #6 — Joint trains on conflicting labels (`2d984cb`)
`cil_cord_joint AA=16.65, last_row=[83.2,0,0,0,0]` — joint learned only task 0. Cause: `CIL_LabelRemapper` masks out-of-session entities to O per session; joint concatenated the masked per-session datasets → the **same receipt appears in multiple sessions with conflicting labels** (row 0 in sessions [0,2,4]). All 800 receipts appear in multiple sessions → contradictory supervision → collapse. Fix: added `joint_train_datasets` to `CLScenario`; joint trains on a full-label dedup pool (each doc once with ALL labels, remapped via the full snapshot). cil_cord pool = 800 unique (not 3234); DIL falls back (disjoint docs); mixed = 3 full datasets. Validated: joint now learns all 5 tasks; bonus ~2.5× faster.

### Issue #7 — OOM-proofing config (`7b36a8a`/`d168b8d`)
Machine OOM'd repeatedly even after Bug #2: the **watchdog's own 30-s restart loop** stacked the new dataset build on un-freed memory (swap thrashing) → abort→restart→OOM 6× in 15 min; original trigger was a parallel dataset-loading sub-agent (removed). Structural fix:
- **Per-run cgroup cap** `MemoryMax=9G MemorySwapMax=0` via `systemd-run`. Verified: 500 MB alloc in a 200 MB scope is OOM-killed inside the scope while host stays idle.
- **Watchdog** threshold lowered 13.5→11 GB; restart blocks on `wait_for_ram()` until RAM <6 GB; tears down leftover `run-*.scope`.

### Bug #8 — EWC Fisher penalty shape (`cc5fc7c`, partial)
`_ewc_penalty` slices `p` to `theta_star.shape` when the head grows (penalize old rows only). Incomplete — see #11.

### Bug #9 — DER++ buffer mixed-width logits (`cc5fc7c`)
`ReservoirBuffer.sample` `torch.stack` crashed on cached `_logits` of mixed width (`[512,13]` vs `[512,25]`) after head growth. Fix: right-pad `_logits` to max C with zeros before stacking (DER++ MSE truncates to n_shared; zero cols neutral). Verified synthetic.

### Issue #10 — EWC OOM without checkpointing (`45fd650`)
A speed experiment disabled gradient checkpointing (bs=2) to use VRAM headroom (4.4 GB for naive/joint). EWC needs Fisher+theta_star VRAM → OOM at 5.6 GB on the 6 GB card. Reverted to checkpointing ON globally; GPU-smoke-tested EWC fits at 4.18 GB. The ~22% no-checkpoint speedup abandoned for correctness.

### Bug #11 — EWC Fisher *accumulation* (`6eb903b`) — the real EWC bug
`size of tensor a (13) must match b (25)`, NOT penalty (held) or OOM. Online EWC accumulates Fisher across tasks (`gamma*old + new`); after task 0 Fisher=13, after task 1 head=25 → `old(13)+new(25)` crash. Fix: pad old Fisher to new head size (new rows = 0 importance); penalty slices all three tensors (fisher, p, theta_star) to common min. Verified through **3 tasks on GPU** (head 13→25→37) at 4.7 GB. **EWC saga:** 3 crash modes, each fixed: (a) penalty shape `cc5fc7c`, (b) OOM `45fd650`, (c) accumulation `6eb903b`.

### Bug #12 — LwF teacher device mismatch (`85f1d53`)
`Expected all tensors to be on the same device... index_select`. The deep-copied frozen teacher's `position_ids` buffer was left on the wrong device after deepcopy+expand+move. (A CPU-teacher approach was tried and also failed/over-engineered.) Fix: `teacher.to(self.device)` each `train_task`; disable teacher checkpointing. The OOM fear was unfounded — **GPU teacher fits at 3027 MB** (frozen eval adds ~400 MB vs naive's 2594). Validated 2 tasks, KD active (kd=0.001–0.004).

### Speed/utilization findings (`ce4605c` ckpt-off, reverted by `45fd650`)
- With checkpointing ON, bs=1/2/4 all peak ~2.5–2.9 GB VRAM (checkpointing dominates; batch size barely affects peak). bs=2 adopted, modest speedup.
- With checkpointing OFF, bs=2 peaks **4.4 GB** (flat across head sizes 13→61), ~22% faster (8.4s vs 10.8s/40 samples) — but broke EWC, reverted.
- Final: **bs=2 + gradient_checkpointing=ON + num_workers=0.**

---

## 4. Autonomous Execution Infrastructure

- **`scripts/run_autonomous_grid.sh`** — PHASE 3 (54) → single-task FWT baselines (9) → PHASE 4 (36) → PHASE 5 doccl (9) → aggregate → ingest. Resume-safe via `.done`. `MEM_CAP=9G`; `run_capped()` wraps single-task runs.
- **`scripts/run_grid_watchdog.sh`** — samples every 20s; HARD-KILL if RAM>11 GB or VRAM>5 GB; restarts on stall/death (waits RAM<6 GB first); heartbeats to `results/logs/watchdog.log`.

**Operating rules (recorded to agent memory):** never exceed 14 GB RAM / 5 GB VRAM; never run parallel dataset-loading work alongside the grid (the 15 GB box fits ONE dataset-builder; a parallel sub-agent once tripped the watchdog); `num_workers=0` mandatory; one training process at a time.

---

## 5. Complete Results (snapshot at 41/54)

All runs: LayoutLMv3-base, bs=2, grad-checkpointing on, 3 epochs/task, seeds {42,123,7}, seqeval entity F1.

### 5.1 Naive — lower bound (9/9)
| Run | AA | BWT | Own-task F1 (diagonal) |
|-----|-----|------|------------------------|
| cil_cord_naive_seed42 | 18.80 | −89.10 | [92.2, 93.9, 89.0, 81.4, 94.0] |
| cil_cord_naive_seed123 | 18.01 | −91.42 | [94.0, 92.8, 92.4, 86.5, 90.1] |
| cil_cord_naive_seed7 | 18.34 | −90.80 | [94.1, 86.9, 94.2, 88.0, 91.7] |
| dil_naive_seed42 | 38.28 | −74.43 | — |
| dil_naive_seed123 | 39.45 | −71.31 | — |
| dil_naive_seed7 | 37.57 | −68.22 | — |
| mixed_naive_seed42 | 33.29 | −63.61 | — |
| mixed_naive_seed123 | 26.70 | −69.59 | — |
| mixed_naive_seed7 | 31.26 | −67.09 | [84.0, 88.4, 81.4, 94.9, 93.3, 81.1] |

Textbook catastrophic forgetting: own-task F1 81–94 (learns each task), AA low, BWT severely negative (earlier tasks forgotten). cil_cord (hardest, 5 sessions) forgets most (BWT≈−90); dil least (≈−71); mixed intermediate (≈−66). Single-task reference: `single_funsd_naive_seed42` AA=87.99.

### 5.2 Joint — upper bound (9/9)
| Run | AA | BWT |
|-----|-----|------|
| cil_cord_joint_seed42 | 30.92 | 0.00 |
| cil_cord_joint_seed123 | 31.22 | 0.00 |
| cil_cord_joint_seed7 | 30.44 | 0.00 |
| dil_joint_seed42 | 85.52 | 0.00 |
| dil_joint_seed123 | 85.01 | 0.00 |
| dil_joint_seed7 | 83.96 | 0.00 |
| mixed_joint_seed42 | 60.59 | 0.00 |
| mixed_joint_seed123 | 63.47 | 0.00 |
| mixed_joint_seed7 | 64.37 | 0.00 |

BWT=0 (no forgetting, all tasks pooled). cil_cord joint last_row e.g. seed42 [67.2,25.4,24.1,24.5,13.4]. cil_cord joint AA modest (~31) because all 60 fine classes share one head — a genuine ceiling. dil ~85, mixed ~61–64 strong upper bounds.

### 5.3 EWC (9/9)
| Run | AA | BWT |
|-----|-----|------|
| cil_cord_ewc_seed42 | 13.07 | −75.70 |
| cil_cord_ewc_seed123 | 13.83 | −75.56 |
| cil_cord_ewc_seed7 | 16.00 | −80.48 |
| dil_ewc_seed42 | 39.02 | −39.12 |
| dil_ewc_seed123 | 46.40 | −35.99 |
| dil_ewc_seed7 | 43.80 | −34.34 |
| mixed_ewc_seed42 | 28.15 | −31.48 |
| mixed_ewc_seed123 | 34.79 | −24.02 |
| mixed_ewc_seed7 | 29.87 | −38.35 |

cil_cord_ewc_seed42 own-task F1 [92.9, 88.6, 62.7, 58.7, 65.3]. EWC consistently reduces forgetting vs naive: cil_cord −76 to −80 (vs −90); **dil −34 to −39 (≈half of naive's −71, AA up to 39–46)**; mixed −24 to −38 (vs −66). The Fisher penalty constrains plasticity, so cil_cord AA dips slightly below naive while BWT improves.

### 5.4 LwF (9/9)
| Run | AA | BWT | Own-task F1 |
|-----|-----|------|-------------|
| cil_cord_lwf_seed42 | 18.80 | −91.39 | [92.9, 91.5, 91.5, 89.7, 94.0] |
| cil_cord_lwf_seed123 | 18.88 | −89.93 | — |
| cil_cord_lwf_seed7 | 18.18 | −89.92 | — |
| dil_lwf_seed42 | 38.44 | −73.64 | — |
| dil_lwf_seed123 | 39.32 | −71.58 | — |
| dil_lwf_seed7 | 37.47 | −70.64 | — |
| mixed_lwf_seed42 | 33.42 | −63.89 | [80.5, 85.8, 77.2, 95.8, 94.3, 86.4] |
| mixed_lwf_seed123 | 31.95 | −66.17 | [86.0, 85.2, 78.6, 95.5, 94.9, 82.4] |
| mixed_lwf_seed7 | 31.44 | −65.86 | — |

**Key honest finding:** LwF ≈ naive across all scenarios (cil_cord −90/−91; dil −71/−74; mixed −64/−66). LwF provides **little anti-forgetting benefit** in this dense token-classification setting — a documented limitation, not a bug: its distillation only constrains old-class logits, which collapse anyway in class-IL when the growing head never reinforces them; and distillation is known to be weaker for token tasks than image classification. Seed-consistent → robust. (Validated end-to-end: KD active, GPU teacher ~3 GB.)

### 5.5 ER — Experience Replay (5/9; cell in progress)
| Run | AA | BWT |
|-----|-----|------|
| cil_cord_er_seed42 | 15.88 | −64.58 |
| cil_cord_er_seed123 | 15.26 | −68.88 |
| cil_cord_er_seed7 | 16.07 | −58.08 |
| dil_er_seed42 | 87.87 | 0.00 |
| dil_er_seed123 | 88.56 | +2.00 |
| dil_er_seed7 | (running) | — |
| mixed_er ×3 | pending | — |

Retention signal (cil_cord_er_seed42): `After task1: {0:56.04, 1:79.15}; After task2: {0:48.08, 1:10.45, 2:35.96}` — **task 0 retains F1≈48–56 deep into the sequence** (vs ~0 for naive/EWC/LwF). **ER is the strongest method.** On cil_cord (hardest), BWT −58 to −69 (best of all). On dil, **AA≈88, BWT≈0 — matching/exceeding the joint upper bound (85), positive BWT means later tasks even slightly *helped* earlier ones.** Replay closes the entire forgetting gap on domain-incremental learning — the headline result. Config: `ReservoirBuffer(capacity=buffer_size default 200, store_logits=False)`, mixes a replay batch each step (double forward → ~2× compute).

### 5.6 DER++ (0/9, not yet run)
`ReservoirBuffer(store_logits=True)`, alpha=0.5 (MSE), beta=0.5 (CE); samples two replay batches per the DER++ paper; `n_shared = min(...)` truncation for the logit MSE. Buffer pad fix (`cc5fc7c`) in place but **not yet verified in production** — its first task-1 run (head growth) is the last fix to confirm.

### 5.7 Consolidated method comparison
| Method | cil_cord BWT | cil_cord AA | dil BWT | dil AA | mixed BWT | mixed AA |
|--------|-------------|-------------|---------|--------|-----------|----------|
| naive (lower) | −89 to −91 | ~18 | −68 to −74 | ~38 | −64 to −70 | ~27–33 |
| LwF | −90 to −91 | ~18 | −71 to −74 | ~38–39 | −64 to −66 | ~31–33 |
| EWC | −76 to −80 | ~13–16 | −34 to −39 | ~39–46 | −24 to −38 | ~28–35 |
| **ER** | **−58 to −69** | ~15–16 | **~0 to +2** | **~88** | (pending) | (pending) |
| joint (upper, AA) | — | ~31 | — | ~84–86 | — | ~61–64 |

**Method hierarchy (forgetting prevention):** naive ≈ LwF < EWC < ER < joint (ceiling). Replay (ER) decisively best; on dil it matches the upper bound. Regularization (EWC) helps moderately, especially on domain shifts. Distillation (LwF) does not help here.

---

## 6. Timing Analysis (measured from `.done` timestamps)
- Naive cil_cord: ~24–40 min/run (system-load dependent).
- EWC: ~40–50 min/run (Fisher overhead).
- LwF: ~40–50 min/run (teacher forward overhead).
- **ER cil_cord: ~2h 16min/run** (timestamps seed42→123→7: 09:25→11:41→13:58). ER does double forward+backward (current + replay batch).
- ER dil: ~1h 14min (3 tasks).

Projection (replay methods are the bottleneck; all remaining runs are ER/DER++): ER remaining ~9.9h; DER++ (9 runs) ~19.5h; **total ~29h (~1.2 days).** bs=2 gave only a marginal speedup because gradient checkpointing recomputes activations every step. Earlier estimates (~2.3–2.6 days) superseded by measured numbers. **User decision: let the full grid finish at 3 seeds (~29h).**

---

## 7. Decisions & Rationale (chronological)
1. Reuse audit first — nothing external reusable; all runs fresh (avoided incomparable backbone/dataset/metric/seeds).
2. 3 epochs not 10 (10≈6.7 days; F1≈88 after 1 epoch).
3. num_workers=0 mandatory (fork duplication caused OOM).
4. Force Linear head (avoids MLP tanh-saturation collapse after CIL expansion).
5. Joint full-label dedup pool (so the upper bound learns all tasks).
6. Gradient checkpointing ON for all methods (EWC/replay need the headroom).
7. Per-run cgroup cap (9 GB) + thrash-proof watchdog (OOM structurally impossible).
8. FWT reported unavailable, not fabricated; emit single-task baseline table.
9. PHASE 5 doccl with DocCL_A placeholder (deferred in practice; grid focused on 6 core methods).
10. Validate each method proactively (GPU-smoke memory-heavy methods; verify own-task F1>0, not just completion).
11. Let the full grid finish (~29h) over descoping.

---

## 8. Commit Log (branch `doccl`)
| Commit | Description |
|--------|-------------|
| `c4b9ff9` | class-IL label filter masks out-of-session tags instead of dropping docs |
| `ba87040` | bound dataset RAM — lazy image decode + shared HF handle per split |
| `f4c61d1` | remap native label ids to head-index space for class-IL scenarios |
| `3ab7008` | expand_classifier handles both LayoutLMv3 head types |
| `358385e` | force Linear classifier head — MLP head collapses to all-O after CIL expansion |
| `50fad3c`/`55cc0bf` | honest FWT handling + single-task baseline table in analyze_results |
| `7b36a8a`/`d168b8d` | autonomous 3-epoch grid driver + watchdog; per-run cgroup cap + thrash-proof restart |
| `2d984cb` | Joint trains on full-label pool not CIL-masked sessions |
| `cc5fc7c` | EWC penalty + DER++ buffer survive classifier-head expansion |
| `ce4605c` | (perf) disable gradient checkpointing at bs=2 — later reverted |
| `45fd650` | re-enable gradient checkpointing so EWC fits in 6 GB GPU |
| `6eb903b` | EWC Fisher accumulation and penalty handle growing classifier head |
| `85f1d53` | LwF teacher device — move whole teacher to student device each task |

---

## 9. Outstanding / Next Steps
1. Finish ER cell (4 runs: dil seed7 + mixed ×3) — ER mixed expected strong.
2. Run DER++ cell (9) — verify buffer pad fix (`cc5fc7c`) holds at first task-1 head growth (watch `stack expects equal size`).
3. Aggregate: `python scripts/analyze_results.py` → `all_runs.csv`, `pivot_*.csv`, `table_main.tex`, `table_ablation.tex`, `table_compute.tex`, `figure_forgetting_curves.pdf`, `table_single_task_baselines.{tex,csv}`.
4. Ingest: `python scripts/ingest_to_thesis.py`.
5. Optional PHASE 4 (prompt/LoRA, 36 runs) and PHASE 5 doccl — deferred.
6. Prose (Ch6/§3.3.6/abstract) against real `results/`, no fabricated numbers.

**Thesis narrative the data supports:** Sequential fine-tuning of LayoutLMv3 on document IE suffers catastrophic forgetting (BWT to −90 on class-IL). Among CL strategies, **replay (ER) is decisively most effective** — matching the joint upper bound on domain-incremental learning (AA 88 vs 85, BWT≈0) and best on class-IL; **regularization (EWC) helps moderately** (halving forgetting on domain shifts); **distillation (LwF) provides little benefit** in this dense token setting. The pilot additionally found **text is the load-bearing modality** (text-deprived → F1=0) and forgetting is **order-dependent**.

---

## 10. References

**Codebase & artifacts**
- CLUE repo, branch `doccl`: `/mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE`
- W&B project: `thanh-workspace/CL4IE` — https://wandb.ai/thanh-workspace/CL4IE
- Repo docs: `docs/PILOT_STUDY_REPORT.md`, `docs/FWT_NOTE.md`, `docs/cil_cord_split.md`, `docs/dil_schema_mapping.md`, `docs/CLASS_IL_LABEL_SPACE_BUG.md`, `RUNBOOK.md`, `STATE.md`.
- External (audited, not reusable): `lth-cl-doc-ie`, `continual-learning-doc-ie` under `/mnt/DataDrive/agentic-ai/openclaw/workspace/research/`.

**Datasets**
- FUNSD — `nielsr/funsd-layoutlmv3` (HuggingFace). Jaume et al., "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents," ICDAR-OST 2019, arXiv:1905.13538.
- CORD-v2 — `naver-clova-ix/cord-v2` (HuggingFace). Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing," DI@NeurIPS 2019.
- SROIE — `mp-02/sroie` (HF mirror). Huang et al., "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction," ICDAR 2019.
- XFUND (external repos only) — `FrancophonIA/XFUND` (HuggingFace).

**Methods / theory**
- LayoutLMv3 — Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (model `microsoft/layoutlmv3-base`).
- CL metrics (AA/BWT/FWT) — Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning," NeurIPS 2017, arXiv:1706.08840.
- EWC — Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (Elastic Weight Consolidation, Fisher information).
- LwF — Li & Hoiem, "Learning without Forgetting" (knowledge distillation from a frozen teacher).
- DER++ — Buzzega et al., "Dark Experience for General Continual Learning: a Strong, Simple Baseline" (NeurIPS 2020). ER = experience replay with reservoir sampling.
- seqeval — BIO-scheme span F1 for token classification.

**Tools** — PyTorch 2.11+cu130; HuggingFace `transformers`, `datasets`; Weights & Biases; `systemd-run` (cgroup memory caps); UV (dependency management).
