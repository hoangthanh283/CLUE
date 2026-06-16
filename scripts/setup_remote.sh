#!/usr/bin/env bash
# One-command setup + run for a GPU CONTAINER instance (e.g. ViLao RTX A5000/4090).
#
# Designed for pytorch/pytorch:*-cuda*-runtime images (torch + CUDA pre-installed).
# Use this when you are already INSIDE a container/VM with the GPUs visible (no
# docker-in-docker) — i.e. you picked a PyTorch/CUDA template and SSH'd in. It:
#   1) checks the GPUs are visible (nvidia-smi),
#   2) reuses the template's torch+CUDA (no reinstall); installs only missing project deps
#      via pip install -e . — skipped entirely if deps are already present (re-entry fast path),
#   3) installs rclone (for durable R2 resume),
#   4) PROMPTS for W&B + Cloudflare R2 creds — held in memory only, never written to disk,
#   5) runs the multi-GPU grid (scripts/run_grid_multigpu.sh) with durable resume + heartbeat.
#
#   git clone <repo> && cd CLUE && git checkout doccl && bash scripts/setup_remote.sh
#
# Re-run after any interruption: completed runs are pulled from R2 and skipped.
# Pre-set any prompted var in the environment to skip its prompt (non-interactive).

set -uo pipefail
cd "$(dirname "$0")/.."

c_bold=$'\e[1m'; c_dim=$'\e[2m'; c_grn=$'\e[32m'; c_red=$'\e[31m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
info(){ echo "${c_bold}==>${c_off} $*"; }
ok(){ echo "  ${c_grn}OK${c_off} $*"; }
warn(){ echo "  ${c_yel}!${c_off} $*"; }
die(){ echo "${c_red}ERROR:${c_off} $*" >&2; exit 1; }

# ── 0. Minimal-image tools (pytorch:*-runtime lacks curl/unzip/git) ──────────────
need_apt=""
for t in curl unzip git; do command -v "$t" >/dev/null 2>&1 || need_apt="$need_apt $t"; done
if [ -n "$need_apt" ]; then
  info "Installing missing tools:$need_apt"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq 2>/dev/null && apt-get install -y -qq $need_apt 2>/dev/null \
      && ok "tools installed" || warn "apt-get failed for$need_apt — install them manually if a later step needs them"
  else
    warn "no apt-get; install$need_apt manually if needed"
  fi
fi

# ── 1. GPUs ─────────────────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — are you on a GPU instance?"
info "GPUs visible:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/      /'
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU" -ge 1 ] || die "no GPUs detected"
ok "$NGPU GPU(s) detected"

# ── 2. Python deps (reuse the template's torch; install the rest) ────────────────
# Find the interpreter that actually HAS a working torch+CUDA. The pytorch/pytorch:*
# images install torch into a conda env (/opt/conda/bin/python), which is NOT always
# first on PATH — bare `python3` may exist with no torch, and `python` may not exist
# at all. So probe a list of candidates and pick the first whose torch sees CUDA;
# fall back to the first that merely imports torch; only then fall through to uv.
PY=""
TORCH_IMPORTABLE=""
for cand in "${PYTHON:-}" /opt/conda/bin/python python3 python; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
  ver=$("$cand" -c 'import torch; print(torch.__version__)' 2>/dev/null) || continue
  # Prefer an interpreter whose torch can see CUDA; take it immediately.
  if "$cand" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    PY="$cand"; TORCH_IMPORTABLE="$ver"; CUDA_OK=1; break
  fi
  # Otherwise remember the first torch-importable interpreter as a fallback.
  [ -z "$PY" ] && { PY="$cand"; TORCH_IMPORTABLE="$ver"; CUDA_OK=0; }
done

if [ -n "$TORCH_IMPORTABLE" ]; then
  info "Python: $($PY --version 2>&1) ($PY)"
  ok "template torch ${TORCH_IMPORTABLE} found — reusing it (no reinstall)"
  # CUDA availability gate. HARD STOP if torch can't see the GPU — otherwise training
  # silently falls back to CPU (~100x slower, ~27s/iter) and burns paid GPU-hours doing
  # nothing useful. This exact trap (driver too old for the image's torch -> CPU fallback)
  # wasted hours on a VastAI/rented box. Set ALLOW_CPU=1 to override (debugging only).
  if [ "${CUDA_OK:-0}" = "1" ]; then
    ok "CUDA available — torch can see the GPU"
  elif [ "${ALLOW_CPU:-0}" = "1" ]; then
    warn "torch CANNOT see the GPU, but ALLOW_CPU=1 set — continuing on CPU (very slow)."
  else
    echo
    die "torch cannot see the GPU (CUDA unavailable) — training would run on CPU (~100x slower).
       Driver/torch mismatch: this image's torch needs a newer CUDA driver than the host provides.
       FIX: pick a VastAI image whose CUDA matches the host driver, e.g.
         pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime  (match 12.x to the host's nvidia-smi CUDA)
       Quick check on any box:  python -c 'import torch; print(torch.cuda.is_available())'
       To force CPU anyway (NOT for the grid): ALLOW_CPU=1 bash scripts/setup_remote.sh"
  fi
  # Fast-path: skip pip ONLY if the project AND its real runtime deps all import.
  # `import doccl` alone is NOT sufficient — doccl is importable just from the source
  # tree on PATH, even when hydra/transformers/etc. were never installed (that exact
  # gap made every grid job die at `import hydra`). Probe the deps train.py actually needs.
  if "$PY" -c 'import doccl, hydra, transformers, seqeval, wandb, datasets' 2>/dev/null; then
    ok "project + runtime deps already installed — skipping pip install"
  else
    info "Installing project deps (pip install -e ., torch already satisfied) ..."
    "$PY" -m pip install -q -e . 2>&1 | tail -3 || die "pip install -e . failed"
    # Verify the deps that previously slipped through are now importable.
    "$PY" -c 'import doccl, hydra, transformers, seqeval, wandb, datasets' 2>/dev/null \
      || die "deps still missing after pip install -e . — check the error above"
  fi
else
  warn "torch not found in base image — installing full locked stack via uv"
  warn "(This will pull a torch wheel; ensure it matches the host CUDA driver)"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  uv sync --extra dev || die "uv sync failed"
  PY=".venv/bin/python"
  if ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    warn "CUDA not available after uv install — driver/torch mismatch likely."
    warn "Pick a pytorch image whose CUDA version matches the host driver and re-run."
    warn "Continuing anyway in case the GPU is accessible at runtime."
  fi
fi
# Tell run_grid_multigpu.sh which interpreter to use.
export PYTHON="$PY"
ok "deps ready (interpreter: $PY)"

# ── 3. rclone (durable resume) ───────────────────────────────────────────────────
if ! command -v rclone >/dev/null 2>&1; then
  info "Installing rclone (durable resume) ..."
  curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 \
    || { mkdir -p "$HOME/bin"; ( cd /tmp && curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o r.zip && unzip -oq r.zip && cp -f rclone-*/rclone "$HOME/bin/" ); export PATH="$HOME/bin:$PATH"; }
  command -v rclone >/dev/null 2>&1 && ok "rclone installed" || warn "rclone install failed — durable resume will be OFF"
else
  ok "rclone present"
fi

# ── 4. Credentials (memory only — never written to disk) ─────────────────────────
prompt_secret(){ local var="$1" label="$2"; [ -n "${!1:-}" ] && { ok "$label via env"; return; }; read -r -s -p "  ${label}: " v; echo; printf -v "$var" '%s' "$v"; }
prompt_plain(){ local var="$1" label="$2" def="${3:-}"; [ -n "${!1:-}" ] && { ok "$label = ${!1} (env)"; return; }; read -r -p "  ${label}${def:+ [$def]}: " v; printf -v "$var" '%s' "${v:-$def}"; }

info "Weights & Biases (Enter to log OFFLINE):"
prompt_secret WANDB_API_KEY "W&B API key"
prompt_plain  WANDB_PROJECT "W&B project" "CL4IE"
[ -n "${WANDB_API_KEY:-}" ] && export WANDB_MODE="${WANDB_MODE:-online}" || { export WANDB_MODE=offline; warn "no key -> offline"; }
export WANDB_API_KEY WANDB_PROJECT

info "Cloudflare R2 durable resume (Enter at all to DISABLE):"
prompt_secret R2_ACCESS_KEY_ID "R2 Access Key ID"
if [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
  prompt_secret R2_SECRET_ACCESS_KEY "R2 Secret Access Key"
  prompt_plain  R2_ENDPOINT "R2 Endpoint URL" ""
  prompt_plain  R2_BUCKET "R2 Bucket" "doccl-results"
  [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || die "R2 secret + endpoint required."
  export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ENDPOINT R2_BUCKET
  ok "durable resume ON -> R2 '${R2_BUCKET}' (in memory only)"
  warn "scope the R2 token to this bucket + DELETE it after the grid finishes."
else
  warn "no R2 creds -> durable resume OFF (a spot-kill would lose progress)."
fi

# ── 5. Grid sizing + launch ──────────────────────────────────────────────────────
# Safe defaults that fit a 24 GB card with headroom: per-job ~7-8 GB at bs=8 in bf16
# (our 6 GB laptop hit ~4.4 GB at bs=2 no-checkpointing, so bs=16 activations are heavy).
# 2 jobs/GPU x bs8 = effective batch 16/GPU with 2-way overlap. After launch, check the
# heartbeat + `nvidia-smi`: if VRAM has lots of free room, re-run with BATCH_SIZE=16 or
# JOBS_PER_GPU=3 (resume-safe — finished runs are skipped).
prompt_plain GPUS         "GPU ids"      "$(seq -s' ' 0 $((NGPU-1)))"
prompt_plain JOBS_PER_GPU "Jobs per GPU" "2"
prompt_plain BATCH_SIZE   "Batch size"   "8"
# DataLoader workers PER JOB. Total = NUM_WORKERS x JOBS_PER_GPU x NGPU, each forking the
# dataset working set. Heavy multilingual sets (XFUND, WildReceipt) can OOM HOST RAM at x4
# across all slots (SIGKILL -> rc=137). Default 2 here is the RAM-safe value for this grid.
prompt_plain NUM_WORKERS  "DataLoader workers/job" "2"
export GPUS JOBS_PER_GPU BATCH_SIZE NUM_WORKERS

# This is the POWERFUL-machine path: run absolutely everything, thoroughly.
#   - core (6) + prompt/LoRA (4) + DocCL + single-task baselines, all 5 scenarios x 3 seeds
#   - DocCL depth-ablation across ALL 5 scenarios (not just cil_cord) -> the fullest study
# All defaults below stay overridable from the environment (e.g. ABLATION_SCENARIOS=cil_cord
# for the lighter ablation, or RUN_ABLATION=0 to skip it).
export CORE_METHODS="${CORE_METHODS:-naive joint ewc lwf er der_pp}"
export PROMPT_METHODS="${PROMPT_METHODS:-l2p dualprompt coda_prompt o_lora}"
export RUN_DOCCL="${RUN_DOCCL:-1}"
export RUN_ABLATION="${RUN_ABLATION:-1}"
export ABLATION_SCENARIOS="${ABLATION_SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
export AMP="${AMP:-1}"   # bf16 on (RTX 4090); set AMP=0 for exact fp32

# Count the planned jobs so you see the full scope before it starts.
NJOBS=$(DRY_RUN=1 bash scripts/run_grid_multigpu.sh 2>/dev/null | grep -cE "  ->  " || echo "?")

echo
info "Launching FULL grid: ${NJOBS} runs (core + prompt/LoRA + DocCL + full ablation, all 5 scenarios x 3 seeds)"
info "  GPUS='${GPUS}' JOBS_PER_GPU=${JOBS_PER_GPU} BATCH_SIZE=${BATCH_SIZE} bf16=$([ "${AMP}" = "1" ] && echo ON || echo OFF) W&B=${WANDB_MODE} sync=$([ -n "${R2_ACCESS_KEY_ID:-}" ] && echo ON || echo OFF)"
echo "  ${c_dim}heartbeat + filtered training lines stream below; re-run this script to resume.${c_off}"
echo
exec bash scripts/run_grid_multigpu.sh
