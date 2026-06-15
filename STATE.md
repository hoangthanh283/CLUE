# STATE

## Active Work
**Converged grid re-running** with val-F1 early stopping (patience=2, cap=100ep) — replaces the
earlier fixed 3-epoch budget that under-trained EWC. Order: single-task baselines (9, DONE) →
DocCL (running) → Core(54)+Prompt(36) as local fallback (offload to powerful machine via docker/).
Per-run **FWT now real** (zero-shot future-task term recorded + tracker seeded with b_i).

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
