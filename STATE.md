# STATE

## Active Work
Autonomous baseline-grid execution (user out, full permission granted 2026-06-12 ~06:30).
Running PHASE 3 (54 core-baseline runs) at **3 epochs** (deadline budget, user-approved) on
the single 6 GB RTX 2060. Then chains: single-task FWT baselines → PHASE 4 (36 prompt/LoRA) →
PHASE 5 (9 doccl, DocCL_A placeholder) → aggregate → ingest → thesis prose.

Launcher: `/tmp/launch_phase3.sh` (PATH=venv, num_workers=0, bs=1+grad_ckpt, epochs=3,
WANDB online CL4IE). Resume-safe via `results/<run>/.done`. Log: `results/logs/grid_phase3.log`.

## Last Decision
- **3 epochs** not 10 (10 = ~6.7 days, infeasible). Model hit F1=88 after 1 epoch, so 3 is
  converged + defensible. ~2 days for PHASE 3.
- **num_workers=0** mandatory: num_workers=4 forked the dataset into ~5×10 GB and OOM-crashed
  the machine. See memory [[clue-resource-limits]].
- Hard ceilings: RAM < 14 GB, VRAM < 5 GB. Watchdog hard-aborts if crossed.

## Fixed this session (8 bugs, all committed+pushed on branch doccl)
1. c4b9ff9 — class-IL label filter: mask out-of-session tags (was: issubset → 0 examples).
2. ba87040 — dataset RAM: lazy image decode + shared HF handle per split (11 GB → 3 GB).
3. f4c61d1 — CIL label remapper: native ids → head-index space (was: CUDA assert).
4. 3ab7008 — expand_classifier: handle both LayoutLMv3 head types.
6. cc5fc7c — EWC penalty: slice to old head shape (was [25]vs[13] crash after expansion).
7. cc5fc7c — DER++ buffer: pad mixed-width cached logits before stack (was [512,13]vs[512,25] crash).
5. 358385e — force Linear head: MLP head (≥10 labels) collapsed to all-O after CIL expansion (tanh saturation killed new-class gradients). Was producing F1=0 on every CIL task after task 0. Fixed + validated (task1 F1=53 post-expand).
Validated: cil_cord_naive 1ep reached task0 F1=88.24, expanded head, advanced to task 2 — clean.

## Method-fix loading
Grid runs naive->joint->ewc->lwf->er->der_pp. EWC+DER fixes are in the tree; the driver spawns each train.py fresh, so ewc/der_pp runs (hours away) load the fixed code — no restart needed. joint/er/lwf already safe.

## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Blocked On
Nothing — fully autonomous. PHASE 5 (doccl) uses DocCL_A placeholder (user accepted re-point risk).
