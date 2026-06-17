# Multi-Machine Experiment Distribution

Three machines run in parallel with **zero overlap** by partitioning the grid on
`SCENARIOS` (the lock-free R2 resume means disjoint partitions = no wasted duplicate
compute). The two 48 GB cards split the measured grid; the 6 GB local card runs the
formal forgetting diagnosis (pilot).

| Machine | Script | Owns |
|---|---|---|
| **RTX 6000 Ada 48 GB** | `setup_remote.sh` | scenarios `cil_cord dil` + DocCL depth-ablation (cil_cord) |
| **L40 48 GB (vast.ai)** | `setup_vastai.sh` | scenarios `mixed dil_xlingual cil_wildreceipt` + all single-task baselines |
| **RTX 2060 6 GB (local)** | `run_pilot.sh` | pilot diagnosis (FUNSD→CORD→SROIE, 5 conditions) |

Each remote runs the **full method set** on its scenarios: classical (naive joint ewc
lwf er der_pp) + BERT text-only + prompt/LoRA (l2p dualprompt coda_prompt o_lora) +
DocCL + currency (er_cflat cl_lora). Verified: the two remotes' DRY_RUN job lists are
fully disjoint.

## 1. RTX 6000 Ada — `setup_remote.sh`

```bash
git clone <repo> && cd CLUE && git checkout doccl
SCENARIOS="cil_cord dil" ABLATION_SCENARIOS="cil_cord" RUN_ABLATION=1 \
  RUN_SINGLETASK=0 BATCH_SIZE=16 JOBS_PER_GPU=3 AMP=1 \
  bash scripts/setup_remote.sh
```
`RUN_SINGLETASK=0` → the L40 owns the (scenario-independent) single-task baselines, so
they run on exactly one box. The script prompts for W&B + R2 creds (held in memory,
never written to disk).

## 2. L40 (vast.ai) — `setup_vastai.sh`

**Instance:** image `pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime` (match `cuda12.x` to
the host driver — setup_remote.sh hard-stops on a mismatch and prints the fix). Pass
secrets as instance ENV at creation (NOT a `.env` on disk):
```
-e WANDB_API_KEY=...  -e WANDB_PROJECT=CL4IE
-e R2_ACCESS_KEY_ID=...  -e R2_SECRET_ACCESS_KEY=...  -e R2_ENDPOINT=...  -e R2_BUCKET=doccl-results
```
**onstart (or first SSH command):**
```bash
git clone <repo> && cd CLUE && git checkout doccl && bash scripts/setup_vastai.sh
```
`setup_vastai.sh` bakes in `SCENARIOS="mixed dil_xlingual cil_wildreceipt"`,
`RUN_ABLATION=0`, and L40 sizing, then execs `setup_remote.sh`. Use the **same R2
bucket** as the Ada — disjoint scenarios keep the work apart; shared `.done` markers just
let either box skip anything already finished.

## 3. RTX 2060 (local) — pilot diagnosis

The pilot is independent of the grid R2 bucket (writes to `results/pilot/`), so it can't
collide with the remotes. Run a gradient-checkpointed, small-batch subset that fits 6 GB
(respect the 14 GB RAM / ~5.8 GB VRAM ceiling):
```bash
EPOCHS=10 BATCH_SIZE=2 GRAD_CKPT=1 \
  CONDITIONS="c4_full c1_text c2_no_text c3_no_image cb_bert" \
  SEEDS="42 123 7" \
  bash scripts/run_pilot.sh
```
> The local box uses **UV**; `run_pilot.sh` calls bare `python`. Run it inside the venv
> (`source .venv/bin/activate` first) or set the interpreter so `python -m doccl.pilot.run_pilot`
> resolves to the project env.

Outputs `results/pilot/<condition>_seed<seed>.json` (CKA / Fisher / displacement /
accuracy) — the inputs to the Ch.6 per-component diagnosis. The pilot defaults to 5 seeds
for the Mann–Whitney significance floor; seeds 42/123/7 first, add 1/2 later if the
significance test needs all five. Then `python -m doccl.pilot.analyze --pilot_dir results/pilot`.

## Coordination & aggregation

- **Disjoint partitions = the coordination.** No distributed lock needed; a box that
  dies mid-run resumes from its own `.done` markers by re-running the same script.
- **Watchdog** (`run_grid_watchdog.sh`, 5.8 GB VRAM cap) is for the **6 GB card only** —
  do NOT run it on the 48 GB boxes.
- All three log to W&B project `CL4IE`; runs aggregate in one dashboard.
- When scenarios finish, pull R2 and run `scripts/analyze_results.py` +
  `scripts/build_all_metrics_table.py` to emit the Ch.6 tables (the new method display
  names er_cflat / cl_lora / bert_textonly are already wired).

## Verify before launching (on each box)

```bash
# Ada — should list ONLY cil_cord/dil jobs (+ cil_cord ablation), NO singles:
SCENARIOS="cil_cord dil" RUN_ABLATION=1 ABLATION_SCENARIOS="cil_cord" RUN_SINGLETASK=0 \
  DRY_RUN=1 bash scripts/run_grid_multigpu.sh | less

# L40 — should list ONLY mixed/dil_xlingual/cil_wildreceipt + single_* jobs, NO ablation:
DRY_RUN=1 bash scripts/setup_vastai.sh   # (exec'd setup_remote.sh prints the job count)
```
