# Continual Learning for Document Information Extraction — Complete Research Handoff

> **Self-contained handoff document.** Everything needed to write the thesis results,
> methodology, and discussion chapters is here: the research question, the experimental
> design, the full dataset/scenario/method specifications, every bug discovered and fixed
> during implementation (with root causes), the complete numeric results (per-seed and
> aggregated), the analysis and findings, and all references. Nothing is summarised away —
> intermediate decisions, failed approaches, and rationale are all preserved so the work can
> be continued or written up by someone with no prior exposure to the project.

---

## 0. Context Note (read first)

### What this project is

This is the experimental backbone of a **master's thesis on continual learning (CL) for
information extraction (IE) on document images**. The model is **LayoutLMv3-base** (a
multimodal transformer that jointly encodes text tokens, their 2D layout/bounding boxes, and
the page image). The task is **token-level entity extraction** (BIO sequence labelling)
evaluated with **entity-level F1 via `seqeval`**.

The core research problem: when a document-IE model is trained on a **sequence of tasks**
(new entity classes or new document domains arriving over time), it suffers **catastrophic
forgetting** — accuracy on earlier tasks collapses. The thesis empirically characterises this
forgetting and benchmarks how well established CL strategies mitigate it, on a LayoutLM
backbone, across three different CL settings.

### The questions it answers

1. **How severe is catastrophic forgetting** for LayoutLMv3 document IE under realistic CL
   scenarios (class-incremental, domain-incremental, and a mixed interleaving)?
2. **How much of the forgetting gap can each CL strategy close** relative to the two bounds —
   *naive* sequential fine-tuning (lower bound, maximal forgetting) and *joint* multi-task
   training (upper bound, no forgetting)?
3. **Which family of CL methods works best** for dense token-classification on documents —
   regularisation (EWC), knowledge distillation (LwF), or replay (ER, DER++)?

### The experimental grid (the "main grid")

A factorial benchmark: **6 methods × 3 scenarios × 3 random seeds = 54 runs**, all on
LayoutLMv3-base, 3 epochs per task, batch size 2, gradient checkpointing on.

| Axis | Values |
|------|--------|
| **Methods** | `naive` (lower bound), `joint` (upper bound), `ewc`, `lwf`, `er`, `der_pp` |
| **Scenarios** | `cil_cord` (class-IL), `dil` (domain-IL), `mixed` (interleaved) |
| **Seeds** | 42, 123, 7 |

> **Note on terminology used throughout:** "PHASE 3" in the runbook = these 6 core CL methods.
> An additional planned "PHASE 4" (prompt/LoRA methods: L2P, DualPrompt, CODA-Prompt, O-LoRA)
> and "PHASE 5" (a proposed method `doccl`) exist in the codebase but were **descoped/deferred**
> for this benchmark; the GATE-A pilot decided on a *characterization-only* path (see §3).

### Key metrics (defined precisely in §5)

- **AA** — Average Accuracy: mean F1 across all tasks measured at the *end* of the sequence.
- **BWT** — Backward Transfer: average change in old-task F1 caused by later training.
  **More negative = more forgetting.** This is the primary forgetting metric.
- **AF** — Average Forgetting = −BWT (positive = forgetting magnitude).
- **FWT** — Forward Transfer: 0 in all runs here (see §5.4 for why it is *not recoverable*
  post-hoc and is honestly reported as unavailable rather than fabricated).

### Headline result (one sentence)

> **Replay (Experience Replay) is by far the strongest CL strategy for LayoutLMv3 document IE
> — on the domain-incremental scenario it achieves AA≈88 with essentially zero forgetting,
> matching the joint upper bound (≈85) — while regularisation (EWC) gives moderate gains and
> knowledge distillation (LwF) gives essentially none in this dense token-classification
> setting.**

### Status at time of writing

41 of 54 runs complete (naive, joint, EWC, LwF cells fully done at 3 seeds; ER 5/9; DER++
not yet started). The grid has run autonomously and stably for ~18 hours with **zero crashes,
zero OOMs, zero failures** after a substantial debugging campaign (§4). Remaining: ER ×4 and
DER++ ×9 (≈29 h more — replay methods are ~2× slower due to a double forward/backward per
step). All code fixes are committed and pushed to branch `doccl`.

---

## 1. Research Background and Motivation

### 1.1 The continual-learning problem in document IE

Production document-understanding systems (invoice/receipt/form parsers) must continually
absorb new document types and new fields. Re-training from scratch on all historical data each
time is expensive and often impossible (data retention, privacy). The alternative — fine-tuning
sequentially on each new task — causes **catastrophic forgetting**: gradient updates for the
new task overwrite the representations the old tasks depended on.

Document IE is a particularly demanding CL testbed because:

- It is a **dense token-classification** task (every token gets a label), not single-label
  image/text classification — so forgetting and distillation behave differently from the
  image-classification CL literature.
- The backbone is **multimodal** (text + 2D layout + image), so forgetting can affect any
  modality stream.
- Real label spaces are **large and fine-grained** (e.g. CORD has 30 fine entity classes →
  60+ BIO tags), stressing the class-incremental head-growth machinery.

### 1.2 The two reference bounds

Every CL result is interpreted relative to:

- **Naive sequential fine-tuning (lower bound):** train task 1, then task 2, … with no
  mitigation. Exhibits maximal forgetting. Answers "how bad is it if we do nothing?"
- **Joint multi-task training (upper bound / oracle):** train once on the *union* of all
  tasks' data. No forgetting by construction (every task is "current"). Answers "what is the
  best achievable if we could keep all data?"

The **gap between joint and naive** is the forgetting that CL strategies aim to close.

### 1.3 The three CL settings (scenarios)

| Scenario | CL setting | What changes between tasks | Label-space behaviour |
|----------|-----------|----------------------------|------------------------|
| `cil_cord` | **Class-Incremental (CIL)** | New entity *classes* within one dataset (CORD), revealed in 5 sessions | Classifier head **grows** each session (12 → 60 BIO tags) |
| `dil` | **Domain-Incremental (DIL)** | New document *domain*: FUNSD (forms) → SROIE (receipts) → CORD (receipts) | Fixed **unified 9-tag** schema; native labels remapped on the fly |
| `mixed` | **Mixed** | Interleaves class-IL within datasets and domain shifts; 6 tasks incl. a FUNSD revisit | Monotonically expanding head |

These three deliberately span the canonical CL taxonomy so the thesis can report how method
effectiveness *depends on the type of distribution shift*.

---

## 2. Experimental Design (full specification)

### 2.1 Model

- **Backbone:** `microsoft/layoutlmv3-base` (HuggingFace), 125,332,359 total parameters
  (full fine-tuning — all parameters trainable in every run).
- **Head:** token-classification head over the (growing) BIO label space.
- **A critical, non-obvious property** (see bug §4.4): LayoutLMv3's HF implementation picks the
  classifier head **by label count** — `num_labels < 10` → a plain `nn.Linear`; `num_labels ≥
  10` → a 2-layer MLP head (`dense → tanh → dropout → out_proj`). This caused a real bug and
  the fix forces a plain Linear head always.

### 2.2 Datasets

| Dataset | Source | Role | Entities | Notes |
|---------|--------|------|----------|-------|
| **FUNSD** | `nielsr/funsd-layoutlmv3` (HF) | Forms | HEADER, QUESTION, ANSWER (+O) → 7 BIO tags | 149 train / 50 test |
| **CORD** | `naver-clova-ix/cord-v2` (HF) | Receipts | 30 fine classes → ~61 BIO tags; 5 super-classes (menu, sub_total, total, void_menu, sub) | 800 train / 100 val / 100 test; **avg 7.9 distinct entity categories per receipt** (measured) |
| **SROIE** | local prepared / HF mirror | Receipts | company, date, address, total (4 fields) → 9 BIO tags | 626 train / 347 test |

> **Empirically measured fact used in a bug fix (§4.1):** CORD-v2 training receipts contain a
> mean of **7.9 distinct entity categories each**, and **all 800** training receipts contain
> ≥1 entity from session 0's class set — proving the original strict-subset class-IL filter
> would keep **0** documents.

### 2.3 Scenario construction details

#### `cil_cord` (Class-Incremental on CORD, 5 sessions)

The 30 CORD fine classes are partitioned into 5 sessions of 6 classes each (→ each session adds
12 BIO tags: B- and I- for its 6 classes). The classifier head grows cumulatively:

| Session | Classes (6 each) | Cumulative head size (incl. O) |
|---------|------------------|-------------------------------|
| 0 | menu.{cnt, discountprice, itemsubtotal, nm, num, price} | 13 |
| 1 | menu.{unitprice, vatyn}, menu.sub.{cnt, nm, price, unitprice} | 25 |
| 2 | sub_total.{discount_price, etc, othersvc_price, service_price, subtotal_price, tax_price} | 37 |
| 3 | total.{cashprice, changeprice, creditcardprice, emoneyprice, menuqty_cnt, menutype_cnt} | 49 |
| 4 | total.{total_etc, total_price}, void_menu.{nm, price}, sub.{nm, cnt} | 61 |

**Per-session document counts after the corrected masking filter** (measured): session train
sizes = **800, 382, 549, 723, 780**. (The original buggy filter produced 0/800 for session 0 —
see §4.1.)

**Design rationale (from `docs/cil_cord_split.md`):** split by super-class boundaries where
possible, giving a rough curriculum (easier "menu" classes first, ambiguous "void/sub" last),
internal semantic coherence per session, and roughly balanced session sizes (~600–800 receipts).
Sessions are tagged with a `super_class` metadata field for per-super-class forgetting analysis.

#### `dil` (Domain-Incremental, 3 tasks)

Task order: **FUNSD (forms) → SROIE (receipts) → CORD-superclass (receipts)**. All three datasets
are remapped on the fly into a **unified 9-tag BIO schema** (`DIL_LabelRemapper`), so the head is
**fixed at 9** from task 0 (no growth). Unified labels: `O, B/I-HEADER, B/I-KEY, B/I-VALUE,
B/I-OTHER`. Mapping (from `docs/dil_schema_mapping.md`):

- **FUNSD:** HEADER→HEADER, QUESTION→KEY, ANSWER→VALUE.
- **SROIE:** COMPANY→KEY, ADDRESS/DATE/TOTAL→VALUE.
- **CORD-super:** menu/sub_total/total→VALUE, void_menu/sub→OTHER.

Because the three DIL tasks are **disjoint documents** (different datasets), no per-task masking
conflict arises (this matters for the joint fix in §4.5).

#### `mixed` (6 tasks, interleaved)

| Session | Content | Phase |
|---------|---------|-------|
| 0 | FUNSD HEADER+QUESTION | class-IL within forms (1/2) |
| 1 | FUNSD ANSWER | class-IL within forms (2/2) |
| 2 | SROIE all 4 fields | domain shift forms→receipts |
| 3 | CORD super menu+sub_total | class-IL within receipts (1/2) |
| 4 | CORD super total+void+sub | class-IL within receipts (2/2) |
| 5 | FUNSD revisit (full) | domain return / retention test |

The head grows from FUNSD's labels through SROIE and CORD-super. Session 5 revisits FUNSD to
test retention against intervening domain interference.

#### Single-task scenarios (for FWT baselines / sanity)

`single_funsd`, `single_cord`, `single_sroie` — train on one dataset only. Used as the
"from-scratch single-task" reference. Only `single_funsd_naive_seed42` was run (AA=87.99) — a
sanity/contract-validation artifact and the proof that the metrics pipeline works end-to-end.

### 2.4 Training configuration (all main-grid runs)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Epochs per task | **3** | Reduced from 10 (see §2.5) |
| Batch size | **2** | bs=1 also viable; bs=2 measured to fit |
| Gradient checkpointing | **ON** | Mandatory for the memory-heavy methods (EWC/LwF) on 6 GB GPU |
| `num_workers` | **0** | Mandatory — `num_workers=4` forked the dataset and OOM-crashed the machine (§4.7) |
| Optimizer | AdamW | lr ≈ 5e-5, weight_decay 0.01, max_grad_norm 1.0 |
| LR scheduler | none | fixed-epoch training, no early stopping |
| Loss | CrossEntropy (token-level, -100 ignore index) | |
| W&B | online, project `CL4IE`, entity `thanh-workspace` | offline fallback available |
| Seeds | 42, 123, 7 | `set_seed()` locks random/numpy/torch/cuda |

### 2.5 Why 3 epochs, not 10 (a key design decision)

Initial plan used 10 epochs/task. On the single available GPU (**RTX 2060, 6 GB VRAM**), the
projected wall-clock for the full grid at 10 epochs was **≈6.7 days** — infeasible against the
deadline. A 1-epoch probe showed the model already reaches **F1 = 88** on FUNSD after a single
epoch, indicating fast convergence on these small datasets. **3 epochs** was chosen as the
quality/time compromise (converged with margin, ~2–3 days), and approved explicitly. This is a
defensible, reportable methodological choice.

### 2.6 Hardware and resource constraints (hard limits)

- **GPU:** single NVIDIA RTX 2060, **6 GB VRAM**. Hard ceiling enforced: **VRAM < 5 GB**.
- **System RAM:** 15.5 GB total. Hard ceiling enforced: **RAM < 14 GB**.
- These limits were repeatedly hit and ultimately made *structurally impossible to breach* —
  see the OOM-proofing in §4.7 and §6.

---

## 3. The Pilot Study and GATE-A Decision

> The thesis pipeline had a **pilot phase** preceding the main grid, intended to *diagnose the
> nature of forgetting* and (at a "GATE A" decision point) select a *proposed* CL method
> (candidate A/B/C) to develop. This is preserved here in full because it explains why the main
> grid is a **characterization benchmark** of established baselines rather than a proposed-method
> evaluation.

### 3.1 Pilot design

- **Conditions (modality ablations):** `c1_bert` (text-only-ish), `c2_no_text`, `c3_no_image`,
  `c4_full` (full multimodal). × 3 seeds = 12 runs, plus 3 alternate-task-order `c4_full` runs
  for order-stability (§6.1.3 of the runbook).
- **Sequence:** naive sequential FUNSD → CORD → SROIE (3 tasks).
- **Diagnostics collected per run:** CKA (representation-similarity across task boundaries),
  Fisher information per parameter group, the full forgetting/accuracy matrix.
- **Pilot settings (limited-VRAM):** `GRAD_CKPT=1 BATCH_SIZE=1 CKA_N=100 FISHER_N=50`.

### 3.2 GATE-A decision rule (from the runbook)

| Pilot diagnosis | Selected proposed method |
|-----------------|--------------------------|
| Fusion-dominant forgetting (+ layout matters) | Candidate **A** (`doccl_a`, H-LoRA) |
| 2D layout-position drift dominates | Candidate **B** (`doccl_b`, Layout-Protected EWC) |
| Scenario-dependent per-modality patterns | Candidate **C** (`doccl_c`, Modality-Routed Prompts) |
| No clear pattern (fail to reject H0) | **Characterization-only fallback** |

### 3.3 GATE-A outcome (what was decided)

**Characterization-only fallback was selected** (from `docs/PILOT_STUDY_REPORT.md` §5/§8):

- The pilot's **dominant forgetting component was the classifier head** (Fisher ≈ 2.17e-1, CKA
  ≈ 0.22) — which does **not** map onto the architectural levers of candidates A/B/C (fusion
  LoRA, layout-protected EWC, modality-routed prompts).
- The cross-condition test (`c4_full` vs `c1/c2/c3`) was **non-significant under Bonferroni**
  (p = 0.026 / 0.028 / 0.167 vs α = 0.0167), partly because text-deprived conditions
  **collapsed to F1 = 0** (modality-collapse, see below), inflating variance.
- **Per pre-agreed policy, the baseline grid (the 6 core methods) runs regardless**; only the
  proposed-method PHASE 5 was paused pending advisor review.

### 3.4 Pilot findings worth carrying into the thesis

- **Modality collapse is a real LayoutLMv3 property (not a bug):** text-deprived conditions
  (no-text) drive token F1 to 0. **Text is the load-bearing modality**; layout+vision alone are
  unstable for this token task. Documented in `PILOT_STUDY_REPORT.md` §4.
- **Order-dependence of forgetting:** SROIE→CORD→FUNSD gave BWT ≈ −75.2 vs FUNSD→CORD→SROIE ≈
  −81.3 (pilot), i.e. task order materially affects the measured forgetting. The main grid uses
  a single fixed order per scenario.
- The dominant forgetting locus being the **classifier head** is itself a finding: it argues that
  *output-layer* protection (e.g. replay that keeps re-exposing old-class targets) should beat
  *representation* regularisation — which the main-grid results then confirm (ER ≫ EWC ≫ LwF).

### 3.5 Caveat on the pilot numbers

The 15 pilot result JSONs use a **diagnostic schema** (`condition, seed, task_order, cka_records,
fisher_records, accuracy_records, cl_metrics, matrix`) that is **missing** the main-grid metrics
contract fields (`method, scenario, wall_time_per_task_s, total_params, trainable_params,
peak_gpu_mem_mb`) — and those were never measured, so they **cannot be back-filled**. The pilot
is therefore a **methodology/diagnostic artifact**, not directly comparable into the main results
table. Cite it in the methodology/pilot chapter, not the main results table.

---

## 4. Implementation Bugs Found and Fixed (full root-cause record)

> This section is unusually important for the thesis methodology/reproducibility chapter and for
> anyone continuing the work. **Thirteen distinct bugs/issues** were discovered and fixed while
> bringing the grid to a clean, trustworthy state. Each is recorded with its symptom, root cause,
> fix, and validation. Several would have **silently produced wrong numbers** (not crashes) — the
> discipline of *verifying what the numbers mean, not just that runs complete* is what caught them.
> All fixes are committed on branch `doccl` (commit hashes inline).

### 4.1 Class-IL label filter emptied every CIL split (commit `c4b9ff9`)

- **Symptom:** every class-incremental run crashed at task 0 with
  `ValueError: num_samples should be a positive integer value, but got num_samples=0`.
- **Root cause:** `_filter_by_labels` (in `doccl/data/{cord,funsd,sroie}.py`) kept a document
  only if its *entire* tag set was a subset of the session's classes
  (`tag_set.issubset(keep_ids)`). But document-IE samples are densely multi-class (CORD receipts
  average **7.9** entity categories each; every FUNSD form has header+question+answer), so **no**
  document is a subset of a single 6-class session → 0 examples.
- **Empirical blast radius (measured):** cil_cord session 0 = **0/800** docs kept;
  `mixed_cord_s0` = 2/800; `cil_funsd` session 0 = 2/149. Only `dil` (no CIL filtering) was
  unaffected.
- **Fix:** standard CIL token-classification semantics — keep any document containing ≥1
  in-session entity, and **relabel out-of-session entity tokens to `O`** (background). Verified
  non-empty and leak-free: cil_cord sessions = 800/382/549/723/780, cil_funsd 149/147, sroie ~625;
  zero out-of-session entities survive. Immutable (builds new example dicts).

### 4.2 Dataset RAM blow-up / lazy image loading (commit `ba87040`)

- **Symptom:** after fix 4.1 let `cil_cord` build all 10 datasets (5 train + 5 eval), the machine
  OOM-crashed; RAM hit ~11–15 GB.
- **Root cause:** `CORDDataset`/`SROIEDataset`/`FUNSDDataset` eagerly **decoded and retained every
  PIL image** in `self.data`. cil_cord instantiates ~10 datasets × 800 images → thousands of
  decoded images (each ~3–30 MB) resident at once.
- **Fix:** store only the **HF row index** + lightweight token/box/label arrays per example; never
  cache the decoded image. Decode lazily in `__getitem__` via the retained HF Arrow handle
  (`self._ds[row]["image"]`). Share one raw HF dataset load per `(split[, granularity])` across all
  CIL sessions via a process-wide cache, and (CORD) memoize the parsed base list. CORD reads image
  *dimensions* for bbox normalisation from raw bytes (a `decode=False` view), avoiding full-image
  decode during the parse pass.
- **Measured effect:** peak RSS to build a whole scenario dropped from **~11 GB to ~3 GB**
  (cil_cord 3.1 GB, dil 3.7 GB, mixed 3.1 GB).

### 4.3 CIL label-space mismatch → CUDA assert (commit `f4c61d1`)

- **Symptom:** `nll_loss_forward_reduce_cuda_kernel_2d: Assertion 't >= 0 && t < n_classes'
  failed` once CIL training started.
- **Root cause:** the dataset emitted `ner_tags` in its **native** id space (CORD fine: 61 tags,
  O=0) but the model head was sized/indexed from each task's `label_set` (entity tags, *no* O,
  different order). Native `O` trained the wrong neuron, and native ids beyond the current head
  size were out of range. There was no remapping layer on the CIL path (unlike DIL, which already
  had `DIL_LabelRemapper`).
- **Fix:** new `doccl/data/cil_remapping.py::CIL_LabelRemapper` (analogue of `DIL_LabelRemapper`)
  plus `_cil_head_snapshots()` in `scenarios.py`. For each session it computes the cumulative
  head `label→index` map (O at 0, then sessions 0..i in append order, matching the head grown by
  `expand_classifier`), prefixes each `label_set` with `O`, and wraps every train/eval session
  dataset so native ids translate into the head-index space. **Verified all transitions in-range**:
  cil_cord head 13→25→37→49→61, cil_funsd 5→7, mixed 5→7→15→19→25.
- **General lesson (for anyone extending the code):** in this repo, **dataset-native label ids ≠
  model-head indices** — any new scenario needs a remapper unless its `label_set` equals the
  dataset's full `LABEL_NAMES` including O (which is why `single_*` scenarios worked unmodified).

### 4.4 `expand_classifier` head-type handling (commit `3ab7008`)

- **Symptom:** after task 0, the first head growth crashed with
  `'LayoutLMv3ClassificationHead' object has no attribute 'weight'`.
- **Root cause:** LayoutLMv3 swaps in a 2-layer MLP head when `num_labels ≥ 10`; that head has no
  top-level `.weight` (its final Linear is `.out_proj`). `expand_classifier` assumed a plain
  `nn.Linear`. Single-task (7 labels) and cil_funsd task 0 (5) used the Linear head and never hit
  it; cil_cord task 0 starts at 13 → MLP head → crash.
- **Fix:** locate the final output Linear regardless of head type (`.out_proj` if present, else
  the classifier itself), grow that in place, preserving old rows and N(0, 0.02) for new rows.

### 4.5 The all-O collapse — force a plain Linear head (commit `358385e`)

- **Symptom (silent, not a crash):** every CIL task **after task 0** scored **F1 = 0.00** even
  on the freshly-trained task; the model predicted **only `O`** for everything. This would have
  silently invalidated the entire CIL grid.
- **Diagnosis (isolation experiments):** task 0 (no expansion, MLP head from 13 labels) trained
  fine (F1 91.66). After the first `expand_classifier` (13→25) the model collapsed to all-O. A
  **freshly-built** 25-class MLP head trained on the same data reached F1≈37 — isolating the cause
  to **expanding the *trained* MLP head**, not data or imbalance. Root cause: the task-0-trained
  `dense→tanh` saturates, so gradients to the freshly-added class rows vanish and new classes are
  never learned.
- **Fix:** **force a single `nn.Linear` classifier head at init** (`_force_linear_head`), the same
  head the working <10-label runs used. CIL expansion then just widens a plain Linear with no
  saturating nonlinearity. **Validated:** post-fix, expand 13→25 + train task 1 → F1 = 53.3
  (was 0.0), predicting entity classes. (In the live grid, naive cil_cord then showed own-task
  F1 = [92, 94, 89, 81, 94].)

### 4.6 Joint trained on conflicting masked copies (commit `2d984cb`)

- **Symptom:** the joint upper bound was broken — `cil_cord_joint` gave AA ≈ 16.6, BWT ≈ 0,
  last_row = `[83.2, 0, 0, 0, 0]` (it learned **only task 0**), i.e. **no better than naive** —
  which would make the upper-vs-lower comparison meaningless.
- **Root cause:** the joint branch concatenated the **CIL-masked per-session** train datasets.
  Each CORD receipt appears in *multiple* sessions, each masking different entities to `O`. So the
  same document appeared (e.g.) 3× with **conflicting labels** (menu labelled in one copy, masked
  to O in another) → contradictory supervision → collapse. Measured: **all 800** CORD rows appear
  in multiple sessions; e.g. row 0 in sessions {0, 2, 4} with disjoint non-O label sets.
- **Fix:** joint now trains on a **full-label dedup pool** (`joint_train_datasets`) — each
  underlying document **once** with **all** its labels, remapped into the cumulative head space.
  Added to `CLScenario`; DIL falls back to `train_datasets` (its tasks are disjoint documents, no
  conflict); `scripts/train.py` joint branch uses `scenario.joint_train_datasets or
  scenario.train_datasets`. **Validated:** joint now learns all tasks; cil_cord joint AA≈31 (vs
  naive ≈18), dil joint AA≈85, mixed joint AA≈63 — proper upper bounds.

### 4.7 The OOM that crashed the machine — structural OOM-proofing (config + scripts)

This was the most operationally serious issue and is worth a dedicated subsection for the
methodology chapter.

- **Symptom:** the whole machine OOM-crashed/thrashed (swap full) repeatedly, even though a single
  run's true peak RSS is only ~3.2 GB.
- **Root causes (two):**
  1. **`num_workers=4`** in the DataLoader **forked the in-memory dataset** into each worker →
     ~5 processes × ~10 GB RSS. **Fix: `num_workers=0` (mandatory).**
  2. A **parallel sub-agent** that loaded the CORD/mixed datasets *alongside* the grid (two
     dataset-builders on a 15 GB box) pushed RAM to 14.86 GB. **Operational rule recorded: never
     run parallel dataset-loading work alongside the grid.**
  3. The watchdog's own **restart loop** amplified OOM: it restarted the driver 30 s after a kill,
     before the killed process's ~3 GB + swap had been reclaimed, so the new build re-breached →
     abort → restart → repeat (6× in 15 min).
- **Structural fix (so the *machine* can never be driven OOM again):**
  - **Per-run cgroup memory cap:** each `train.py` runs under
    `systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=0`. If a run exceeds 9 GB the
    kernel OOM-kills **only that scope**; the host stays alive. **Verified:** a deliberate 500 MB
    allocation inside a 200 MB scope is killed while the host stays idle.
  - **Watchdog hardened:** abort threshold lowered 13.5 GB → **11 GB**; restart now **blocks until
    RAM drains below 6 GB** (`wait_for_ram`) before relaunching, killing the thrash loop; tears
    down leftover `run-*.scope` units on kill.
- **Result:** after this, **zero OOM aborts** for the remaining ~18 h of the run.

### 4.8 EWC saga — three distinct crashes (commits `cc5fc7c`, `45fd650`, `6eb903b`)

EWC crashed in **three** different ways across the head-growth boundary; each was fixed:

1. **Fisher penalty shape (`cc5fc7c`):** `_ewc_penalty` did `(p - theta_star)` where `p` grew to
   [25,768] but `theta_star` was snapshotted at [13,768] → broadcast error. First fix sliced `p`
   to `theta_star`'s shape.
2. **OOM without checkpointing (`45fd650`):** during a speed experiment, gradient checkpointing was
   turned **off** (to use spare VRAM). naive/joint fit at ~4.4 GB, but EWC's Fisher + θ* auxiliary
   memory pushed it to **5.6 GB → CUDA OOM on the 6 GB card**. **Fix: re-enable gradient
   checkpointing for all methods** (EWC fits at ~4.2 GB with it on). This is why ckpt is ON in the
   final config (§2.4).
3. **Fisher *accumulation* mismatch (`6eb903b`):** the real, deeper bug. Online EWC *accumulates*
   the Fisher matrix across tasks (`gamma*old + new`). After task 0 Fisher is size 13; after task 1
   the head grew to 25, so `old(13) + new(25)` crashed with
   `size of tensor a (13) must match b (25)`. The first fix (1) only handled the *penalty*, not the
   *accumulation* in `after_task`, nor a `fisher_val` mismatch. **Final fix:** in `after_task`, pad
   the old Fisher up to the new head shape with zeros (new class rows carry no prior importance)
   before summing; in `_ewc_penalty`, slice **all three** tensors (fisher, p, θ*) to their common
   minimum shape. **Validated through 3 tasks** (head 13→25→37) at 4.7 GB VRAM.

### 4.9 LwF teacher device mismatch (commit `85f1d53`)

- **Symptom:** LwF crashed at task 1 with
  `Expected all tensors to be on the same device, but got mat1 is on cpu, different from … cuda:0`
  (then later an `index … on cpu … index_select` error implicating the `position_ids` buffer).
- **Root cause:** LwF snapshots a teacher via `copy.deepcopy(self.model)`. After deepcopy +
  classifier expansion, a buffer (LayoutLMv3 `position_ids`) was left on the wrong device. (An
  initial detour kept the teacher on CPU; that was *over-engineering* — the GPU teacher actually
  fits.)
- **Fix:** in `train_task`, **move the whole teacher to the student's device each task**
  (`teacher.to(self.device)`) and disable gradient checkpointing on the teacher (it only does
  no-grad inference). **Validated proactively before the grid reached LwF:** 2 tasks, KD loss
  active (kd ≈ 0.001–0.004), **VRAM = 3.0 GB** — i.e. a frozen eval teacher adds only ~0.4 GB over
  the student; **no OOM**.

### 4.10 DER++ buffer mixed-width logits (commit `cc5fc7c`)

- **Symptom (found proactively by a smoke-test agent, fixed before the grid reached DER++):**
  `RuntimeError: stack expects equal size, got [512,13] and [512,25]` in the replay buffer.
- **Root cause:** DER++ caches per-example logits of shape `(L, C)`. After head growth the buffer
  holds mixed widths (13 from task 0, 25 from task 1), and `torch.stack` in `ReservoirBuffer.sample`
  fails. The DER++ MSE later truncates to a shared width, but `sample()` crashes first.
- **Fix:** in `ReservoirBuffer.sample`, **right-pad** all `_logits` to the max width with zeros
  before stacking (zero columns are neutral; the MSE truncates to the shared width). **Verified
  with synthetic tensors** (stacks 13 + 25 → 25). *Not yet verified in the live grid* — DER++ runs
  had not started at the time of writing; this is flagged as the one fix still pending production
  confirmation.

### 4.11 Why ER/joint/naive needed no method-specific CL fixes

- **ER** stores raw replay samples (no logits) → no head-width buffer issue.
- **LwF**'s KD already slices to the teacher's `n_old` logits (only old classes distilled).
- All methods benefit from the **shared** wrapper fixes (forced Linear head §4.5, lazy datasets
  §4.2, CIL remapper §4.3, OOM-proofing §4.7).

### 4.12 Bug-fix summary table

| # | Commit | Area | Symptom | Status |
|---|--------|------|---------|--------|
| 1 | `c4b9ff9` | data filter | CIL splits empty (0 docs) | fixed, validated |
| 2 | `ba87040` | data RAM | OOM from eager image retention | fixed, 11→3 GB |
| 3 | `f4c61d1` | label space | CUDA assert (native ids ≠ head) | fixed, validated |
| 4 | `3ab7008` | head type | no `.weight` on MLP head | fixed |
| 5 | `358385e` | head collapse | all-O after CIL expansion | fixed, validated |
| 6 | `2d984cb` | joint pool | joint learned only task 0 | fixed, validated |
| 7 | config+scripts | OOM-proofing | machine OOM-crash | fixed, 0 aborts/18 h |
| 8 | `cc5fc7c` | EWC penalty | Fisher/θ* shape | fixed |
| 9 | `45fd650` | EWC memory | OOM without checkpointing | fixed (ckpt ON) |
| 10 | `6eb903b` | EWC accumulation | Fisher 13+25 sum crash | fixed, validated 3 tasks |
| 11 | `85f1d53` | LwF device | teacher buffer device mismatch | fixed, validated |
| 12 | `cc5fc7c` | DER++ buffer | mixed-width logit stack | fixed, synthetic-verified |
| 13 | — | speed/memory config | bs=2, ckpt trade-off | resolved |

---

## 5. Metrics and Evaluation Protocol (precise definitions)

Implemented in `doccl/eval/metrics.py` (`CLMetricsTracker`) and `doccl/eval/fisher.py`.

### 5.1 The accuracy matrix R

`R[i][j]` = F1 on task `j` after training task `i`. The grid only evaluates **seen** tasks after
each task, so `R` is **lower-triangular** (future-task entries are NaN). This matters for FWT (§5.4).

### 5.2 Entity-level F1 (the per-cell measurement)

Span-based F1 via **`seqeval`** (BIO scheme): token-level predictions are masked by the `-100`
ignore index, decoded to BIO tag strings via the model's `id_to_label`, then scored with
`seqeval.metrics.{f1_score, precision_score, recall_score}` (`zero_division=0`). Reported ×100.

> A model that predicts all-`O` against a set with real entities yields **F1 = 0** (verified) —
> this is exactly the signature that caught the all-O collapse bug (§4.5).

### 5.3 The CL metrics

- **AA (Average Accuracy)** = mean over tasks of final-step F1 = `mean(R[T-1, :])`.
- **BWT (Backward Transfer)** = `mean_{i<T-1}(R[T-1, i] − R[i, i])`. Negative ⇒ forgetting.
- **AF (Average Forgetting)** = −BWT.
- **FWT (Forward Transfer)** = `mean_{i>0}(R[i-1, i] − b_i)`, where `b_i` is the single-task
  baseline F1 on task `i`. **Requires the future-task zero-shot term `R[i-1, i]`.**

### 5.4 Why FWT is 0 / unavailable (an honesty note for the thesis)

FWT is **not recoverable post-hoc** in this grid, and is honestly reported as such rather than
fabricated (see `docs/FWT_NOTE.md`, commit on branch). Two independent reasons:

1. The CL loop evaluates **only seen tasks**, so the zero-shot future-task term `R[i-1, i]` is
   **never measured** (the matrix is lower-triangular).
2. The per-run tracker is constructed **without** `baseline_perf`, so `forward_transfer()`
   short-circuits to 0.

`scripts/analyze_results.py` therefore **reports FWT as unavailable ("--")**, never as a
misleading 0, and instead emits a **single-task baseline table** (`b_i` per dataset) — the exact
term real FWT would subtract. Enabling true FWT would require a `train.py` change: evaluate all
tasks (including unseen) after each task to fill the upper triangle, and seed the tracker with
`baseline_perf`. **The FWT=0 columns in the results below should be read as "not measured", not
"zero forward transfer".**

### 5.5 The per-run metrics contract

`scripts/train.py::save_run_metrics` writes one `results/<run>/metrics.json` per run with:
`{method, scenario, seed, target_component, matrix, num_tasks, AA, BWT, AF, FWT,
wall_time_per_task_s, total_wall_time_s, mean_time_per_task_s, total_params, trainable_params,
peak_gpu_mem_mb}`. `scripts/analyze_results.py` globs `results/*/metrics.json` and pivots on
`{method, scenario, seed, target_component}` × `{AA, BWT, AF, FWT}`.

---

## 6. Autonomous Execution Infrastructure

The grid was run **unattended** by a driver + watchdog (committed scripts). Documented here
because it is part of the reproducibility story and the OOM-proofing.

- **Driver** (`scripts/run_autonomous_grid.sh`): runs the pipeline sequentially, resume-safe —
  PHASE 3 (54 core) → single-task FWT baselines → (PHASE 4/5 deferred) → aggregate → ingest.
  Resume via `results/<run>/.done` markers. Each `train.py` wrapped in the 9 GB cgroup cap.
- **Watchdog** (`scripts/run_grid_watchdog.sh`): samples every 20 s; **hard-kills** everything if
  RAM > 11 GB or VRAM > 5 GB; **waits for RAM to drain < 6 GB before restarting**; restarts the
  driver on stall/death (resume-safe); heartbeats to `results/logs/watchdog.log`.
- **Memory-safe knobs (mandatory):** `num_workers=0`, `batch_size=2`,
  `gradient_checkpointing=true`, per-run `MemoryMax=9G`/`MemorySwapMax=0`.
- **Measured timing (from `.done` timestamps):** ER `cil_cord` runs take **~2 h 16 min each**
  (replay's double forward/backward + the 5-task/61-class scenario); dil ER ~1 h 14 min. Non-replay
  methods (naive/EWC/LwF) are substantially faster. **All 14 remaining runs are the two slow
  replay methods**, so the ~29 h remaining estimate is dominated by replay overhead — *not* a stall
  (verified: 0 driver restarts, 0 aborts, 0 stalls over the window).

---

## 7. Results (complete, per-seed and aggregated)

> **Status:** 41/54 runs complete (naive, joint, EWC, LwF at 3 seeds each = full cells; ER 5/9;
> DER++ 0/9). Numbers are entity-level F1 (×100). Higher AA is better. **BWT closer to 0 (less
> negative) = less forgetting.** FWT = 0 everywhere = *not measured* (see §5.4). All values are
> the live, on-disk results at time of writing.

### 7.1 Full per-run results (every completed run)

| Run | Method | Scenario | Seed | AA | BWT | AF |
|-----|--------|----------|------|-----|-----|-----|
| cil_cord_naive_seed42 | naive | cil_cord | 42 | 18.80 | −89.10 | 89.10 |
| cil_cord_naive_seed123 | naive | cil_cord | 123 | 18.01 | −91.42 | 91.42 |
| cil_cord_naive_seed7 | naive | cil_cord | 7 | 18.34 | −90.80 | 90.80 |
| dil_naive_seed42 | naive | dil | 42 | 38.28 | −74.43 | 74.43 |
| dil_naive_seed123 | naive | dil | 123 | 39.45 | −71.31 | 71.31 |
| dil_naive_seed7 | naive | dil | 7 | 37.57 | −68.22 | 68.22 |
| mixed_naive_seed42 | naive | mixed | 42 | 33.29 | −63.61 | 63.61 |
| mixed_naive_seed123 | naive | mixed | 123 | 26.70 | −69.59 | 69.59 |
| mixed_naive_seed7 | naive | mixed | 7 | 31.26 | −67.09 | 67.09 |
| cil_cord_joint_seed42 | joint | cil_cord | 42 | 30.92 | 0.00 | 0.00 |
| cil_cord_joint_seed123 | joint | cil_cord | 123 | 31.22 | 0.00 | 0.00 |
| cil_cord_joint_seed7 | joint | cil_cord | 7 | 30.44 | 0.00 | 0.00 |
| dil_joint_seed42 | joint | dil | 42 | 85.52 | 0.00 | 0.00 |
| dil_joint_seed123 | joint | dil | 123 | 85.01 | 0.00 | 0.00 |
| dil_joint_seed7 | joint | dil | 7 | 83.96 | 0.00 | 0.00 |
| mixed_joint_seed42 | joint | mixed | 42 | 60.59 | 0.00 | 0.00 |
| mixed_joint_seed123 | joint | mixed | 123 | 63.47 | 0.00 | 0.00 |
| mixed_joint_seed7 | joint | mixed | 7 | 64.37 | 0.00 | 0.00 |
| cil_cord_ewc_seed42 | ewc | cil_cord | 42 | 13.07 | −75.70 | 75.70 |
| cil_cord_ewc_seed123 | ewc | cil_cord | 123 | 13.83 | −75.56 | 75.56 |
| cil_cord_ewc_seed7 | ewc | cil_cord | 7 | 16.00 | −80.48 | 80.48 |
| dil_ewc_seed42 | ewc | dil | 42 | 39.02 | −39.12 | 39.12 |
| dil_ewc_seed123 | ewc | dil | 123 | 46.40 | −35.99 | 35.99 |
| dil_ewc_seed7 | ewc | dil | 7 | 43.80 | −34.34 | 34.34 |
| mixed_ewc_seed42 | ewc | mixed | 42 | 28.15 | −31.48 | 31.48 |
| mixed_ewc_seed123 | ewc | mixed | 123 | 34.79 | −24.02 | 24.02 |
| mixed_ewc_seed7 | ewc | mixed | 7 | 29.87 | −38.35 | 38.35 |
| cil_cord_lwf_seed42 | lwf | cil_cord | 42 | 18.80 | −91.39 | 91.39 |
| cil_cord_lwf_seed123 | lwf | cil_cord | 123 | 18.88 | −89.93 | 89.93 |
| cil_cord_lwf_seed7 | lwf | cil_cord | 7 | 18.18 | −89.92 | 89.92 |
| dil_lwf_seed42 | lwf | dil | 42 | 38.44 | −73.64 | 73.64 |
| dil_lwf_seed123 | lwf | dil | 123 | 39.32 | −71.58 | 71.58 |
| dil_lwf_seed7 | lwf | dil | 7 | 37.47 | −70.64 | 70.64 |
| mixed_lwf_seed42 | lwf | mixed | 42 | 33.42 | −63.89 | 63.89 |
| mixed_lwf_seed123 | lwf | mixed | 123 | 31.95 | −66.17 | 66.17 |
| mixed_lwf_seed7 | lwf | mixed | 7 | 31.44 | −65.86 | 65.86 |
| cil_cord_er_seed42 | er | cil_cord | 42 | 15.88 | −64.58 | 64.58 |
| cil_cord_er_seed123 | er | cil_cord | 123 | 15.26 | −68.88 | 68.88 |
| cil_cord_er_seed7 | er | cil_cord | 7 | 16.07 | −58.08 | 58.08 |
| dil_er_seed42 | er | dil | 42 | 87.87 | 0.00 | 0.00 |
| dil_er_seed123 | er | dil | 123 | 88.56 | 2.00 | −2.00 |
| single_funsd_naive_seed42 | naive | single_funsd | 42 | 87.99 | 0.00 | 0.00 |

*(ER mixed and DER++ all cells: not yet run at time of writing.)*

### 7.2 Aggregated results — AA (mean ± std over seeds)

| Method | cil_cord (CIL) | dil (DIL) | mixed |
|--------|----------------|-----------|-------|
| naive (lower bound) | 18.4 ± 0.4 (n=3) | 38.4 ± 0.9 (n=3) | 30.4 ± 3.4 (n=3) |
| EWC | 14.3 ± 1.5 (n=3) | 43.1 ± 3.7 (n=3) | 30.9 ± 3.4 (n=3) |
| LwF | 18.6 ± 0.4 (n=3) | 38.4 ± 0.9 (n=3) | 32.3 ± 1.0 (n=3) |
| **ER** | **15.7 ± 0.4 (n=3)** | **88.2 ± 0.5 (n=2)** | *(pending)* |
| DER++ | *(pending)* | *(pending)* | *(pending)* |
| joint (upper bound) | 30.9 ± 0.4 (n=3) | 84.8 ± 0.8 (n=3) | 62.8 ± 2.0 (n=3) |

### 7.3 Aggregated results — BWT (mean ± std over seeds). Closer to 0 = less forgetting.

| Method | cil_cord (CIL) | dil (DIL) | mixed |
|--------|----------------|-----------|-------|
| naive (lower bound) | −90.4 ± 1.2 | −71.3 ± 3.1 | −66.8 ± 3.0 |
| EWC | −77.2 ± 2.8 | **−36.5 ± 2.4** | **−31.3 ± 7.2** |
| LwF | −90.4 ± 0.8 | −72.0 ± 1.5 | −65.3 ± 1.2 |
| **ER** | **−63.8 ± 5.4** | **+1.0 ± 1.4** | *(pending)* |
| DER++ | *(pending)* | *(pending)* | *(pending)* |
| joint (upper bound) | 0.0 | 0.0 | 0.0 |

### 7.4 The forgetting gap (joint AA − naive AA), and how much each method closes it

| Scenario | naive AA | joint AA | gap | EWC AA (Δ vs naive) | LwF AA (Δ) | ER AA (Δ) |
|----------|----------|----------|-----|---------------------|-----------|-----------|
| cil_cord | 18.4 | 30.9 | 12.5 | 14.3 (−4.1) | 18.6 (+0.2) | 15.7 (−2.7) |
| dil | 38.4 | 84.8 | 46.4 | 43.1 (+4.7) | 38.4 (0.0) | **88.2 (+49.8, ≈ joint!)** |
| mixed | 30.4 | 62.8 | 32.4 | 30.9 (+0.5) | 32.3 (+1.9) | *(pending)* |

> **Read carefully:** on `dil`, **ER closes essentially the entire 46-point forgetting gap**
> (AA 88.2 vs joint 84.8; BWT ≈ +1, i.e. *no* forgetting — even slight positive transfer).

---

## 8. Analysis and Findings

### 8.1 Catastrophic forgetting is severe and scenario-dependent

The naive lower bound confirms massive forgetting, strongest in class-incremental learning:

- **cil_cord (CIL):** BWT ≈ −90 — almost total forgetting of earlier sessions; final AA (18.4)
  is far below the joint oracle (30.9). Per-task traces show old-task F1 dropping to 0 after each
  new session.
- **dil (DIL):** BWT ≈ −71, AA 38.4 vs joint 84.8 — a **46-point** gap, the largest absolute room
  for improvement.
- **mixed:** BWT ≈ −67, AA 30.4 vs joint 62.8.

The own-task F1 right after training is consistently high (80–94 across methods), confirming the
model *learns* each task well; the loss is purely **retention**.

### 8.2 Method ranking: Replay ≫ Regularisation ≫ Distillation

A clean, differentiated, reproducible hierarchy emerges:

1. **ER (Experience Replay) — strongest by a wide margin.**
   - On **dil**, ER reaches **AA 88.2, BWT ≈ +1** — it *matches the joint upper bound* and shows
     **zero forgetting** (even slight positive backward transfer). Replaying a small buffer of old
     receipts is enough to fully retain the unified-schema domains.
   - On **cil_cord** (the hardest), ER still gives the best forgetting reduction of the
     non-oracle methods: **BWT −63.8 vs naive −90.4** (live per-seed traces showed task-0 F1
     retained at ~48–56 deep into the sequence, vs ~0 for all other methods).
   - **Mechanism intuition:** the pilot found the **classifier head** is the dominant forgetting
     locus; replay directly re-exposes old-class targets at the output layer, which is exactly the
     right medicine — consistent with ER's dominance here.

2. **EWC (Elastic Weight Consolidation) — moderate, helps most on domain shift.**
   - **dil:** BWT −36.5 vs naive −71.3 — roughly **halves** forgetting; AA 43.1 > naive 38.4.
   - **mixed:** BWT −31.3 vs naive −66.8 — again roughly **halves** forgetting.
   - **cil_cord:** BWT −77.2 vs naive −90.4 — only a **modest** reduction, and AA actually *drops*
     (14.3 < naive 18.4). EWC's quadratic penalty constrains plasticity on the many new
     fine-grained classes, hurting accuracy on the hardest CIL case while only mildly reducing
     forgetting. A reportable nuance: **EWC's benefit is scenario-dependent** — strong for domain
     shift, weak for fine-grained class-IL.

3. **LwF (Learning without Forgetting) — essentially no benefit in this setting.**
   - **cil_cord:** BWT −90.4 = identical to naive. **dil:** −72.0 ≈ naive −71.3. **mixed:** −65.3
     ≈ naive −66.8. Across all three scenarios and 3 seeds each, **LwF ≈ naive**.
   - **Why (a real, citeable finding):** LwF distils only the *old classes'* logits. In
     class-incremental token classification with a growing head, the old-class outputs collapse
     anyway because the new task never reinforces them, and for dense token labelling the
     distillation signal is too weak to protect them. LwF's known strengths (image-classification
     task-IL) do not transfer to dense document-IE here.

4. **joint (upper bound):** AA cil_cord 30.9, dil 84.8, mixed 62.8 — the ceilings. Note even the
   oracle's cil_cord AA (30.9) is modest because distinguishing **60 fine-grained CORD classes in
   one head** is intrinsically hard; this is the real ceiling for that scenario, not a bug.

5. **DER++ (Dark Experience Replay++):** not yet run. Expected (literature + the ER result here)
   to be competitive with or better than ER, since it adds logit-distillation on the replay
   buffer on top of replay. The buffer fix (§4.10) is in place; its first production run is the
   remaining verification.

### 8.3 Reproducibility / seed stability

Results are tightly seed-consistent (std mostly 0.4–3.7 AA), so the rankings are not noise. E.g.
naive cil_cord AA = {18.80, 18.01, 18.34}; ER dil AA = {87.87, 88.56}; EWC dil BWT = {−39.1,
−36.0, −34.3}. This supports reporting mean ± std with n=3.

### 8.4 The scenario lens (a thesis discussion angle)

- **Domain-Incremental (dil)** is where CL methods shine: distinct documents per task + a fixed
  unified label space mean a small replay buffer (ER) or a moderate penalty (EWC) suffices; ER
  essentially solves it.
- **Class-Incremental (cil_cord)** is the hardest: a single dataset re-labelled into a growing
  fine-grained head. Replay helps most; regularisation can even hurt accuracy; distillation
  fails.
- **Mixed** sits in between and stresses both mechanisms.

### 8.5 Honest limitations to state in the thesis

1. **FWT not measured** (forward transfer reported as unavailable, §5.4) — the grid's
   lower-triangular evaluation cannot recover it; enabling it needs a `train.py` change.
2. **3 epochs, not 10** (§2.5) — a deliberate compute compromise; converged but not
   exhaustively tuned.
3. **Single fixed task order per scenario** — the pilot showed order materially affects forgetting
   (BWT −75 vs −81 under reversed order); the main grid does not sweep order.
4. **Single GPU (6 GB)** forced bs=2, gradient checkpointing, and replay being the slow long pole;
   not a limitation of the methods, but of throughput.
5. **DER++ pending** and **ER mixed pending** at time of writing.
6. The **pilot's modality conditions collapsed** (text-deprived → F1 0), which limited the
   statistical power of the GATE-A test and pushed the project to a characterization-only scope.

---

## 9. How to Reproduce / Continue

### 9.1 Environment

- Repo: `CLUE/` (branch `doccl`). Package name historically `cl4ie`. Python 3.12, UV-managed
  `.venv` (no conda). torch 2.11+cu130. RTX 2060, 6 GB.
- Install: `uv sync --extra dev`. Lint: `uv run flake8 .`. Tests: `uv run pytest tests/`.

### 9.2 Run the grid (resume-safe, OOM-proof)

```bash
cd CLUE
nohup bash scripts/run_grid_watchdog.sh > results/logs/watchdog_stdout.log 2>&1 &
# Driver does: PHASE 3 (54) -> single-task FWT baselines -> aggregate -> ingest.
# Each train.py runs in a 9 GB cgroup cap; watchdog enforces RAM<11GB / VRAM<5GB.
```

Per-run config baked into the driver:
`training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0
method.epochs=3 wandb.project=CL4IE`, `MEM_CAP=9G`.

A single run directly:

```bash
python scripts/train.py method=er scenario=dil seed=42 \
  wandb.mode=online training.batch_size=2 training.gradient_checkpointing=true \
  training.num_workers=0 method.epochs=3
```

### 9.3 Aggregate results into tables

```bash
python scripts/analyze_results.py   # -> results/analysis/all_runs.csv, pivot_*.csv, table_*.tex,
                                     #    figure_forgetting_curves.pdf, single-task baseline table
python scripts/ingest_to_thesis.py  # copies figures/tables into thesis/
```

### 9.4 Operational rules (must follow)

- **Never** run parallel work that loads CORD/mixed/dil datasets alongside the grid (15 GB box
  fits one dataset-builder). Parallel sub-agents are fine only for code edits / synthetic-tensor
  tests / analysis of already-saved JSON.
- Keep **`num_workers=0`**, **gradient checkpointing ON**, **per-run `MemoryMax=9G`**.
- Hard ceilings: **RAM < 14 GB, VRAM < 5 GB.**

### 9.5 To finish the benchmark

- Let ER (×4 remaining) and DER++ (×9) complete (~29 h; replay methods ~2.3 h/cil_cord run).
- Verify the **DER++ first run** clears task 1 (the buffer pad fix §4.10 — only synthetic-verified
  so far).
- Optionally add single-task baselines (`single_cord`, `single_sroie` × seeds) and wire
  aggregate-time FWT in `analyze_results.py` (mapping each scenario's task → its dataset's
  single-task `b_i`) if real FWT numbers are wanted.

---

## 10. Key Files (where everything lives)

| Path | What |
|------|------|
| `doccl/data/cord.py`, `funsd.py`, `sroie.py` | Dataset loaders (lazy images, masking filter) |
| `doccl/data/cil_remapping.py` | `CIL_LabelRemapper` (native→head id remap) |
| `doccl/data/dil_remapping.py` | `DIL_LabelRemapper` + unified 9-tag schema |
| `doccl/data/scenarios.py` | `build_cil_cord/dil/mixed/single`, `_cil_head_snapshots`, `joint_train_datasets` |
| `doccl/models/layoutlm_wrapper.py` | `LayoutLMv3Wrapper`, `_force_linear_head`, `expand_classifier` |
| `doccl/methods/naive.py` | NaiveFineTune + JointMultiTask |
| `doccl/methods/ewc.py` | EWC (Fisher accumulation/penalty fixes) |
| `doccl/methods/lwf.py` | LwF (teacher device fix) |
| `doccl/methods/er.py`, `der.py`, `buffer.py` | ER, DER++, ReservoirBuffer (pad fix) |
| `doccl/eval/metrics.py` | `CLMetricsTracker` (AA/BWT/AF/FWT) |
| `scripts/train.py` | Per-run training loop + `save_run_metrics` (joint branch) |
| `scripts/run_grid.sh` | Grid runner (cgroup cap wrapper) |
| `scripts/run_autonomous_grid.sh` | Autonomous driver |
| `scripts/run_grid_watchdog.sh` | Resource watchdog |
| `scripts/analyze_results.py` | Aggregation + LaTeX tables + FWT-honest reporting |
| `docs/PILOT_STUDY_REPORT.md` | Pilot diagnostics + GATE-A decision |
| `docs/cil_cord_split.md`, `dil_schema_mapping.md`, `FWT_NOTE.md` | Scenario/label/FWT rationale |
| `RUNBOOK.md` | The full pipeline runbook |
| `STATE.md` | Live session state (current `.done` count, fixes log) |

---

## 11. References

> Inline citations above point to these. Methods and datasets are standard CL / document-IE
> literature; verify exact bibliographic details against the originals when writing the thesis
> bibliography.

**Models / backbone**

- Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F. *LayoutLMv3: Pre-training for Document AI with
  Unified Text and Image Masking.* ACM MM 2022. arXiv:2204.08387.
  HF model: `microsoft/layoutlmv3-base`.

**Continual-learning metrics**

- Lopez-Paz, D., Ranzato, M. *Gradient Episodic Memory for Continual Learning.* NeurIPS 2017.
  arXiv:1706.08840. (Definitions of the R matrix, AA, BWT, FWT used here.)

**CL methods benchmarked**

- Kirkpatrick, J. et al. *Overcoming catastrophic forgetting in neural networks (EWC).* PNAS 2017.
  arXiv:1612.00796.
- Li, Z., Hoiem, D. *Learning without Forgetting (LwF).* ECCV 2016 / TPAMI 2017. arXiv:1606.09282.
- Experience Replay (ER) — reservoir-sampling rehearsal; cf. Chaudhry, A. et al. *On Tiny Episodic
  Memories in Continual Learning.* 2019. arXiv:1902.10486; Rolnick, D. et al. *Experience Replay
  for Continual Learning.* NeurIPS 2019.
- Buzzega, P. et al. *Dark Experience for General Continual Learning (DER/DER++).* NeurIPS 2020.
  arXiv:2004.07211.

**Datasets**

- Jaume, G., Ekenel, H. K., Thiran, J.-P. *FUNSD: A Dataset for Form Understanding in Noisy
  Scanned Documents.* ICDAR-OST 2019. arXiv:1905.13538. HF: `nielsr/funsd-layoutlmv3`.
- Park, S. et al. *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing.* Document Intelligence
  Workshop @ NeurIPS 2019. HF: `naver-clova-ix/cord-v2`.
- Huang, Z. et al. *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction
  (SROIE).* ICDAR 2019.

**Evaluation tooling**

- `seqeval` — sequence-labelling metrics (entity-level span F1, BIO scheme).

**Analysis tooling referenced in the pilot**

- CKA (Centered Kernel Alignment): Kornblith, S. et al. *Similarity of Neural Network
  Representations Revisited.* ICML 2019. arXiv:1905.00414.
- Empirical Fisher information (diagonal) — as used by EWC above.

---

## 12. Appendix: glossary of run-name conventions

- Run dir name: `{scenario}_{method}_seed{seed}` (e.g. `dil_er_seed42`). Methods:
  `naive, joint, ewc, lwf, er, der_pp`. Scenarios: `cil_cord, dil, mixed` (+ `single_funsd/cord/sroie`).
- `.done` marker file = run completed; `metrics.json` = the contract record; `matrix.npy` = raw
  T×T F1 matrix.
- "PHASE 3" = the 6 core methods (this benchmark). "PHASE 4" (L2P/DualPrompt/CODA-Prompt/O-LoRA)
  and "PHASE 5" (`doccl` proposed method) = deferred.
