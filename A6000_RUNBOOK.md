# A6000 grid runbook — LexSlot + full grid

**Job counts (verified via `DRY_RUN=1`). ALL backbones (LayoutLMv3 + LiLT + BROS + BERT) run by
default:**
- **FULL grid = 873 jobs** (5 scenarios × 3 seeds × ~14 methods × 4 backbones + ablations +
  single-task). At 8-way parallel, ~30-45 min/job → **~2-3 days** on one A6000.
- **LexSlot contribution alone = 69 jobs** across all 4 backbones (24 primary: 15 main + 9
  cil_cord ablation; + 45 secondary = 15 × LiLT/BROS/BERT) → **~5-7h** on an A6000 at 8-way.
  (Primary backbone only = 24 jobs, ~2h — set `BACKBONES=""`.)

**Two launchers:**
- `scripts/run_grid_multigpu.sh` — the FULL grid (all methods + baselines). Use for the
  complete thesis table.
- `scripts/run_lexslot_grid.sh` — **LexSlot ONLY** across all scenarios/seeds/backbones (no
  baseline re-runs). Use when you just want the contribution numbers fast. All backbones by
  default; resume-safe (skips `metrics.json`/`.done`); self-parallel via `JOBS`.

The grid is **resume-safe** (skips any run whose `results/<run>/.done` exists) and already
wired for LexSlot (`RUN_LEXSLOT=1`, `lexslot` in `BACKBONE_METHODS`). **Decide scope below
before launching** — the full 873-job grid re-runs every baseline (the local box's existing
baseline results are gitignored, so a fresh A6000 won't have them).

Branch `doccl` on `git@github.com:hoangthanh283/CLUE.git` already has all the LexSlot code +
fixes (verified up to date with origin). The local `dil` LexSlot results are NOT on the remote
(`results/` is gitignored) — the A6000 will re-run them (cheap there), unless you copy them over
(Step 0, optional).

---

## Step 1 — Rent the box

Provider: Vast.ai / RunPod / Lambda. Pick:
- **GPU:** 1× RTX A6000 (48 GB).
- **Template/image:** a `pytorch/pytorch:*-cuda*-runtime` image (torch + CUDA pre-installed →
  `setup_remote.sh` skips reinstalling torch). On Vast.ai search the "PyTorch" template.
- **Disk:** ≥ 60 GB (datasets + HF cache + results).
- SSH in.

## Step 2 — Clone + bootstrap (one command)

```bash
git clone git@github.com:hoangthanh283/CLUE.git && cd CLUE && git checkout doccl
# (if the box has no SSH key for GitHub, use HTTPS:)
# git clone https://github.com/hoangthanh283/CLUE.git && cd CLUE && git checkout doccl
```

`setup_remote.sh` installs deps (reuses the template torch), checks the GPU, prompts for W&B
(optional — answer blank / set `WANDB_MODE=offline` to skip), then launches the grid. Pass the
A6000 knobs inline so it packs jobs and skips the small-GPU recipe:

```bash
GPUS=0 \
JOBS_PER_GPU=8 \
BATCH_SIZE=16 \
GRAD_CKPT=false \
NUM_WORKERS=4 \
AMP=1 \
GPU_VRAM_BUDGET_GB=46 \
WANDB_MODE=offline \
bash scripts/setup_remote.sh
```

- `JOBS_PER_GPU=8` + `GPU_VRAM_BUDGET_GB=46` → the VRAM-aware scheduler packs light jobs
  (lexslot/doccl ~5 GB) up to ~8-up while heavy replay jobs (der_pp ~35 GB) stay 1-2 — no OOM.
- `WANDB_MODE=offline` avoids needing a W&B login; metrics still land in `results/` for analysis.

## Step 3 — Watch it

```bash
tail -f results/logs/multigpu_grid.log         # scheduler heartbeat + per-job status
watch -n5 nvidia-smi                            # confirm ~8 jobs packed, VRAM < 48 GB
ls results/*/.done | wc -l                      # completed-run counter
```

If the box dies / you re-provision: re-run the **same** Step-2 command — finished runs are
skipped (via `.done`). For durable cross-restart resume on ephemeral instances, configure
`SYNC_REMOTE` + R2 creds (see `docs/MULTI_MACHINE_RUN.md`); not needed for a single ~7h session.

## Step 4 — Pull results back here, then analyze locally

From THIS local box (not the A6000), copy the results dir back:

```bash
# replace user@HOST:PORT and the remote path with your instance's
scp -P <PORT> -r user@<HOST>:~/CLUE/results/  /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE/results_a6000/
# then merge the new run dirs into local results/ (only dirs with a .done):
rsync -a --include='*/' --include='*/.done' --include='*/metrics.json' --include='*/matrix.npy' \
      --include='*/per_class_f1.json' --include='*/tb/***' --exclude='*' \
      results_a6000/ results/
```

Then on the local box:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/Thesis/CLUE
uv run python scripts/analyze_results.py --source local   # -> all_runs.csv, pivot_*, table_*.tex, figures
uv run python scripts/ingest_to_thesis.py                  # copy tables/figures into thesis/
```

---

## What the default grid runs (this is the FULL grid)

With the Step-2 command (all `RUN_*` default to 1, `BACKBONES="lilt bros bert"`):

- **Scenarios:** `cil_cord dil mixed dil_xlingual cil_wildreceipt`
- **Seeds:** `42 123 7`
- **Primary backbone (LayoutLMv3):** single-task baselines, classical (naive/joint/ewc/lwf/er/der_pp),
  prompt/LoRA (l2p/dualprompt/coda_prompt/o_lora), currency (er_cflat/cl_lora), **doccl**, **lexslot**
  (+ the cil_cord ablation: doccl depth targets, lexslot slot_depth/sharing).
- **Secondary backbones (LiLT/BROS/BERT):** the full method set incl. **lexslot**.

### LexSlot ONLY (all backbones, no baseline re-runs) — the fast path (~5-7h, 69 jobs)

After `setup_remote.sh` has installed deps once (or run it then Ctrl-C before the grid, or just
run the line below directly if deps are ready):

```bash
JOBS=8 bash scripts/run_lexslot_grid.sh                       # all 4 backbones (default)
JOBS=8 BACKBONES="" bash scripts/run_lexslot_grid.sh          # primary LayoutLMv3 only (24 jobs, ~2h)
DRY_RUN=1 bash scripts/run_lexslot_grid.sh                    # print plan + count, don't run
```

### To run LESS of the FULL grid, restrict via env on the Step-2 command:

```bash
# LexSlot + its direct comparison baselines, ALL backbones, all scenarios:
CORE_METHODS="naive joint er der_pp" PROMPT_METHODS="" CURRENCY_METHODS="" RUN_BERT=0 \
GPUS=0 JOBS_PER_GPU=8 BATCH_SIZE=16 GRAD_CKPT=false GPU_VRAM_BUDGET_GB=46 WANDB_MODE=offline \
bash scripts/setup_remote.sh
```

`DRY_RUN=1 ... bash scripts/run_grid_multigpu.sh` (after setup) prints the job plan + count
without running — use it to confirm scope/cost before committing paid GPU hours.

## Cost sanity

A6000 ≈ $0.50–0.80/hr on Vast.ai. Full grid ~7h (8-way parallel) → **~$4–6 total**. The
LexSlot-only or restricted subsets finish in 1–2h → ~$1–2.
