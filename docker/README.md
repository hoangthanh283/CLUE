# Running the DocCL grid on a powerful machine (Docker)

This image runs the **heavy half** of the benchmark — Core (54 runs) + Prompt/LoRA
(36 runs) + the single-task FWT baselines (9) — that we offload from the 6 GB laptop.
DocCL (the proposed method) is run on the laptop and is **not** run here by default.

Training protocol matches the laptop: **val-F1 early stopping** (patience 2, epoch
cap 100, best-val weights restored) and **per-run FWT** (records the zero-shot
future-task term + seeds the metrics tracker with the single-task baselines `b_i`).

## Prerequisites
- A CUDA-13-capable NVIDIA driver.
- The **NVIDIA Container Toolkit** so `--gpus all` works. Verify with:
  ```bash
  docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi
  ```
  If this errors with `could not select device driver "" with capabilities: [[gpu]]`,
  install the toolkit (Ubuntu):
  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
    -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ```
- A Weights & Biases API key (or run offline).

> Build validated on 2026-06-15: image `doccl-grid` (14.8 GB) builds cleanly
> (`import doccl, torch` → torch 2.12.0+cu130; all 12 scenarios incl. the new
> `dil_xlingual`/`cil_wildreceipt` register inside the container). GPU-in-container
> training was NOT smoke-tested on the build machine because it lacked the NVIDIA
> Container Toolkit — verify `--gpus all` on the target box with the command above
> before launching the grid.

## 1. Build the image
From the **repo root** (build context = repo, see `.dockerignore`):

```bash
docker build -t doccl-grid -f docker/Dockerfile .
```

The build installs deps from `uv.lock` (reproducible) and bakes the source in. SROIE
data is **not** baked — it is regenerated from the HuggingFace mirror on first run.

## Multi-GPU (e.g. 2× RTX 4090) — fastest path to COMPLETE results

For a multi-GPU box, use the **parallel scheduler** `scripts/run_grid_multigpu.sh`
instead of the serial `run_grid_remote.sh`. It builds the full ~189-job grid
(single-task baselines → core → prompt/LoRA → DocCL + depth ablation), then keeps
every GPU slot saturated (one `train.py` per slot, pinned via `CUDA_VISIBLE_DEVICES`),
resume-safe via `results/<run>/.done`. The model is ~3 GB, so a 24 GB 4090 fits
several jobs at once with large batches and no gradient checkpointing.

```bash
# Inspect the plan first (no execution):
docker run --rm --gpus all --entrypoint bash doccl-grid -c \
  "DRY_RUN=1 bash scripts/run_grid_multigpu.sh"

# Run the COMPLETE grid on 2 GPUs, 2 jobs/GPU = 4 concurrent, bs=16:
docker run --rm --gpus all \
  -e WANDB_API_KEY=YOUR_KEY -e WANDB_PROJECT=CL4IE \
  -e GPUS="0 1" -e JOBS_PER_GPU=2 -e BATCH_SIZE=16 \
  -v "$PWD/results:/workspace/results" -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  --entrypoint bash doccl-grid -c "bash scripts/run_grid_multigpu.sh"
```

Tune for throughput: replay-free methods (naive/joint/ewc/lwf/prompt/LoRA) tolerate
`BATCH_SIZE=32` and `JOBS_PER_GPU=3`; ER/DER++ hold a replay buffer, so keep them at
`JOBS_PER_GPU=2`, `BATCH_SIZE=16` if VRAM gets tight. Then aggregate (incl. real FWT):
```bash
docker run --rm --gpus all -v "$PWD/results:/workspace/results" --entrypoint bash \
  doccl-grid -c "python scripts/analyze_results.py && python scripts/ingest_to_thesis.py"
```

## 2. Run the full remote grid, SINGLE GPU (baselines → core → prompt → aggregate)

```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY=YOUR_KEY \
  -e WANDB_PROJECT=CL4IE \
  -v "$PWD/results:/workspace/results" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  doccl-grid
```

- `-v .../results` persists `metrics.json` + `.done` markers so runs are **resume-safe**
  across container restarts (and so you can copy them back to the laptop).
- `-v .../.hf_cache` persists the HuggingFace dataset cache between runs (faster restarts).

## 3. Common variations

Run only the core baselines (skip prompt/LoRA), bigger batch on a big GPU:
```bash
docker run --rm --gpus all -e WANDB_API_KEY=KEY \
  -e PHASES="baselines core aggregate" -e BATCH_SIZE=16 -e GRAD_CKPT=false \
  -v "$PWD/results:/workspace/results" doccl-grid
```

Offline (no W&B):
```bash
docker run --rm --gpus all -e WANDB_MODE=offline \
  -v "$PWD/results:/workspace/results" doccl-grid
```

Tunable env vars (see `scripts/run_grid_remote.sh`): `BATCH_SIZE` (default 8),
`GRAD_CKPT` (default false), `EPOCHS_CAP` (100), `SEEDS` ("42 123 7"),
`SCENARIOS` ("cil_cord dil mixed"), `PHASES` ("baselines core prompt aggregate"),
`WANDB_MODE` (online).

## 4. Bringing results back to the laptop
The grid is keyed by `results/<run>/.done`. Copy the remote `results/` dir into the
laptop's `results/` (rsync/scp). The laptop's local driver already produced DocCL +
baselines; once both result sets coexist, run on the laptop:

```bash
python scripts/analyze_results.py     # aggregates ALL runs (incl. real FWT column)
python scripts/ingest_to_thesis.py    # refreshes thesis tables/figures
```

Result dirs use unique names (`<scenario>_<method>_seed<seed>`), so local and remote
sets merge without collision. The single-task baselines (`single_*`) may be produced
on both — identical by construction; whichever lands first wins via `.done`.

## Notes
- The image regenerates SROIE via `scripts/prepare_sroie.py --source hf` if
  `data/sroie/` is absent — no manual data prep needed.
- No 6 GB cgroup cap here (the laptop's `MEM_CAP`/watchdog are laptop-specific). On a
  big GPU you can raise `BATCH_SIZE` and turn `GRAD_CKPT=false` for speed.
