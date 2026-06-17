# Running the DocCL grid on a powerful machine (Docker)

## ⚡ One command (recommended)

```bash
git clone <repo> && cd CLUE && git checkout doccl
bash scripts/launch.sh
```
`scripts/launch.sh` does everything: checks Docker + GPU access, **prompts you for the
W&B + Cloudflare R2 credentials** (held in memory only — never written to disk on this
box), builds the image if needed, and launches the full multi-GPU grid with durable
resume + a live progress heartbeat. Re-run it after any interruption — completed runs are
skipped (pulled from R2). Press Enter at the R2 prompts to run without durable sync.

> The credentials you type are forwarded into the container as runtime env vars and exist
> only for the process lifetime. Nothing sensitive is saved to the (rented/shared) instance
> disk. Scope the R2 token to your bucket and delete it after the grid finishes.

The rest of this document explains the manual `docker build` / `docker run` paths and tuning
that `launch.sh` automates.

---

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

### Watching progress (`docker logs`)
`run_grid_multigpu.sh` prints a **heartbeat block** every `HEARTBEAT_SECS` (default 45s) plus
filtered live training lines (`=== Task`, `val_f1`, `STOP`, `Final: AA`), so `docker logs -f
<container>` shows both overall progress and what each slot is doing:
```
── HEARTBEAT ── 73/189 done | 0 failed | elapsed 2h11m | ETA ~3h28m
    running: dil_xlingual_ewc_seed42 | === Task 3/5: dil_xfund_fr ===
    running: cil_wildreceipt_der_pp_seed7 | val_f1=61.2 best=62.0 bad=1
```
A machine-readable `results/logs/progress.json` (`{done,total,failed,running:[...]}`) is also
written each heartbeat for external dashboards.

## On-demand / ephemeral instances (vast.ai, runpod, lambda, vilao.ai) — DURABLE RESUME

**The problem:** an on-demand instance's local disk dies with the instance. `-v $PWD/results`
only persists to *that ephemeral disk*, so a fresh instance has no `.done` markers and the
~189-run grid restarts from zero — re-billing GPU-hours you already paid for. The grid's resume
state is tiny (`metrics.json` + `matrix.npy` + `.done` ≈ 700 B/run, ~130 KB total), so the fix is
to keep it in durable storage. Two options:

### Option A — persistent/network volume (if your provider offers one)
Mount the provider's persistent disk at the results path and leave sync off:
```bash
docker run ... -v /mnt/persistent/results:/workspace/results ... doccl-grid ...
```
⚠️ A plain *local* instance disk is **not** durable — only a true network/persistent volume is.

### Option B — object-storage sync via rclone (RECOMMENDED, works on any provider)
The scheduler pulls prior resume state at startup, pushes it every `SYNC_SECS` (default 300s) and
on exit (incl. provider SIGTERM teardown). Off by default; enabled by setting `SYNC_REMOTE`.
`rclone` is baked into the image. On startup it runs a **fail-fast preflight** (write+list+delete
a `.synccheck` on the remote): if creds/bucket/endpoint are wrong it aborts in seconds rather than
running 189 un-persisted jobs (override with `SYNC_STRICT=0` to continue without sync).

#### Credentials — keep them OFF the rented disk

A rented/on-demand box is shared and its disk may be reused or snapshotted, so **do not leave a
`.env` (or any secret file) on it.** Pass R2 creds as **runtime `-e` flags** instead — the
scheduler configures rclone purely from environment variables (`RCLONE_CONFIG_OBJ_*`), so nothing
sensitive is ever written to disk inside the container:

```bash
# Creds live only in this command's process env (e.g. read from YOUR LAPTOP's .env, or a
# secret manager) — they never persist on the GPU instance.
set -a; source .env; set +a    # on your laptop, NOT on the rented box
docker run --rm --gpus all \
  -e WANDB_API_KEY -e WANDB_PROJECT=CL4IE \
  -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e R2_ENDPOINT -e R2_BUCKET=doccl-results \
  -e GPUS="0 1" -e JOBS_PER_GPU=2 -e BATCH_SIZE=16 \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  --entrypoint bash doccl-grid -c "bash scripts/run_grid_multigpu.sh"
```
(`-e VAR` with no `=value` forwards the value from the current shell — so the secret is never typed
on the command line or saved in a file on the instance.) On startup the scheduler configures the
rclone `obj` remote in-process, runs a **fail-fast preflight** (write+list+delete a `.synccheck`),
and aborts in seconds if creds/bucket are wrong. **Verified round-trip against R2.**

**Token hygiene (important):** create the R2 API token as **Object Read & Write scoped to ONLY the
`doccl-results` bucket**, and **delete/rotate it in the Cloudflare dashboard once the grid
finishes.** Then even a leak is bounded to tiny resume files and is dead after the run.

> Avoid `--env-file .env` if that means copying `.env` onto the rented box. `--env-file` is fine
> only when the file lives on a trusted machine you control (e.g. your laptop driving a remote
> Docker daemon). The `-e VAR` (forward-from-shell) form above is the safest default.

**Easiest backend: a private Hugging Face dataset repo** (you already have an HF token):
```bash
# 1) One-time: create a PRIVATE dataset repo to hold the resume state.
huggingface-cli repo create doccl-results --type dataset --private    # -> <user>/doccl-results

# 2) One-time: make an rclone remote for HF's S3-compatible endpoint, or use any S3/R2/B2.
#    Simplest portable path is Cloudflare R2 / B2 / S3 (rclone "s3" backend). Example rclone.conf:
#      [obj]
#      type = s3
#      provider = Cloudflare         # or Other / AWS / Backblaze
#      access_key_id = <KEY>
#      secret_access_key = <SECRET>
#      endpoint = <your-r2-endpoint>
#    Then base64 it for injection:  base64 -w0 ~/.config/rclone/rclone.conf

# 3) Run with sync ON. Kill the instance any time; a new instance with the SAME command
#    pulls the .done markers at startup and continues where it left off.
docker run --rm --gpus all \
  -e WANDB_API_KEY=KEY -e WANDB_PROJECT=CL4IE \
  -e GPUS="0 1" -e JOBS_PER_GPU=2 -e BATCH_SIZE=16 \
  -e SYNC_REMOTE="obj:doccl-results/results" \
  -e RCLONE_CONFIG_B64="$(base64 -w0 ~/.config/rclone/rclone.conf)" \
  -e SYNC_SECS=300 -e HEARTBEAT_SECS=45 \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  --entrypoint bash doccl-grid -c "bash scripts/run_grid_multigpu.sh"
```
> HF-native alternative to S3: rclone also has an experimental HF backend, or you can replace the
> sync step with `huggingface-cli upload <user>/doccl-results results/ --include "*/.done" ...`.
> The S3-compatible path (R2/B2) above is the most reliable with the baked-in rclone.

**Resume guarantee:** spot-killed mid-run → that job has no `.done` → it simply re-runs on the next
instance (correct). Completed jobs are skipped (`[skip]` in the log). When `SYNC_REMOTE` is unset,
all sync calls are pure no-ops.

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
