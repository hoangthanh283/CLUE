# Rented-box bootstrap — LiLT + BROS slice (229 cells)

BERT slice is DONE locally (126/126). This box runs **LiLT + BROS only** (disjoint from
local BERT; LayoutLMv3 already complete). ~229 cells, ~3 days on a 24 GB card, ~$100.

## 0. Rent

Vast.ai / RunPod: **1× RTX 4090 (24 GB)** or better, ≥ 32 GB RAM, ≥ 60 GB disk, CUDA 12.x,
`pytorch` template. 2 GPUs halves wall-clock (`GPUS="0 1"`).

## 1. One-shot bootstrap (paste as a block after SSH)

```bash
set -e
cd ~ && git clone git@github.com:hoangthanh283/CLUE.git 2>/dev/null || (cd CLUE && git pull)
cd ~/CLUE && git checkout doccl && git pull    # MUST get the grid-script fixes (empty-override + CORE_BEFORE/AFTER)
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
# (optional) durable resume across box restarts — set your R2 creds, else skip:
# export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=doccl-results R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
```

Do NOT run `setup_remote.sh` — it prompts interactively and defaults to all 5 scenarios +
DocCL + ablation + BERT. Launch the grid directly with the scoped command below.

## 2. Sanity check (must print the model configs + a DRY_RUN plan of 229)

```bash
uv run python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
DRY_RUN=1 SCENARIOS="cil_cord dil mixed" BACKBONES="lilt bros" \
  BACKBONE_METHODS="naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora cl_lora er_cflat doccl lexslot" \
  RUN_SINGLETASK=0 RUN_BERT=0 RUN_DOCCL=0 RUN_LEXSLOT=0 RUN_ABLATION=0 \
  CORE_METHODS="" CORE_BEFORE_DOCCL="" CORE_AFTER_DOCCL="" PROMPT_METHODS="" CURRENCY_METHODS="" \
  bash scripts/run_grid_multigpu.sh | tail -3   # planned jobs; already-done cells skip
```

## 3. Launch (detached, resume-safe)

```bash
SCENARIOS="cil_cord dil mixed" BACKBONES="lilt bros" \
  BACKBONE_METHODS="naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora cl_lora er_cflat doccl lexslot" \
  RUN_SINGLETASK=0 RUN_BERT=0 RUN_DOCCL=0 RUN_LEXSLOT=0 RUN_ABLATION=0 \
  CORE_METHODS="" CORE_BEFORE_DOCCL="" CORE_AFTER_DOCCL="" PROMPT_METHODS="" CURRENCY_METHODS="" \
  GPUS="0" JOBS_PER_GPU=2 BATCH_SIZE=8 NUM_WORKERS=2 AMP=1 WANDB_MODE=offline \
  nohup setsid bash scripts/run_grid_multigpu.sh > grid.log 2>&1 & disown
tail -f grid.log      # watch; Ctrl-C detaches, grid keeps running
```

On 24 GB, all methods fit 2 slots (the local doccl-OOM was purely the 6 GB card — do NOT
carry over the 1-slot doccl workaround here). If you rented 2 GPUs: `GPUS="0 1"`.

## 4. When it finishes

```bash
# pull the results back to the local box (or push via R2 if creds were set), then locally:
uv run python scripts/analyze_results.py --source local   # regenerates table_backbone_*, all_runs.csv
uv run python scripts/ingest_to_thesis.py                 # copies tables/figures into thesis/
```

## Notes / traps (already handled at config level, listed for confidence)

- LiLT tokenizer: `configs/model/lilt_base.yaml` uses `nielsr/lilt-xlm-roberta-base` — no
  separate tokenizer arg needed.
- Scenario scope: `cil_cord dil mixed` only — never `dil_xlingual`/`cil_wildreceipt` (not
  part of the B+ headline; their datasets also trip the num_workers RAM trap).
- Resume: relaunch the exact command after any crash; `.done`-marked cells skip in µs.
- BERT is NOT in this command — it's done locally; keep the slices disjoint.
