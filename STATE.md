# STATE

## Active Work
Autonomous baseline grid running (bs=2, 3 epochs, full scope per user 2026-06-12). 3/54 done
(cil_cord_naive x3 seeds, all valid: AA~18, BWT~-90, own-task F1 80-94, seed-consistent). On
dil_naive now. OOM-proof: each run in 9GB cgroup cap, watchdog 11GB+drain-before-restart. Real
pace ~39min/run -> ~2.6 days total. No aborts since OOM-proofing relaunch (10:12).

## Last Decision
- **3 epochs** not 10 (10 = ~6.7 days, infeasible). Model hit F1=88 after 1 epoch, so 3 is
  converged + defensible. ~2 days for PHASE 3.
- **num_workers=0** mandatory: num_workers=4 forked the dataset into ~5×10 GB and OOM-crashed
  the machine. See memory [[clue-resource-limits]].
- Hard ceilings: RAM < 14 GB, VRAM < 5 GB. Watchdog hard-aborts if crossed.

## Fixed this session (11 bugs/issues, all committed+pushed)
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

## Method-fix loading
Grid runs naive->joint->ewc->lwf->er->der_pp. EWC+DER fixes are in the tree; the driver spawns each train.py fresh, so ewc/der_pp runs (hours away) load the fixed code — no restart needed. joint/er/lwf already safe.

## CRITICAL OPS RULE
NEVER run parallel work that loads CORD/mixed/dil datasets alongside the grid — the 15GB box fits ONE dataset-builder. A parallel smoke-test agent tripped the 14GB watchdog (it recovered, resume-safe). Parallel sub-agents OK only for non-dataset work (code edits, synthetic-tensor tests, doc/analysis of saved JSON).

## Blocked On
Nothing — fully autonomous. PHASE 5 (doccl) uses DocCL_A placeholder (user accepted re-point risk).
