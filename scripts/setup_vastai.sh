#!/usr/bin/env bash
# Vast.ai entry point for the DocCL grid — a thin wrapper over scripts/setup_remote.sh.
#
# Vast.ai gives you a container with the GPU visible (same shape as any rented box), so
# the real work — interpreter discovery, dep install, CUDA gate, R2 durable resume,
# heartbeat, grid sizing — all lives in setup_remote.sh (ONE source of truth). This
# wrapper only does the Vast.ai-specific pre-flight: verify the GPU is actually usable
# BEFORE handing off, because Vast.ai lets you pick the image and a CUDA/driver mismatch
# silently trains on CPU (~100x slower) and burns paid GPU-hours.
#
# IMAGE CHOICE (critical): pick a PyTorch CUDA-runtime image whose CUDA matches the
# host driver shown by `nvidia-smi`. A mismatch is the #1 cause of CPU fallback. Good:
#     pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime
#
# USAGE (after SSH-ing into the Vast.ai instance):
#     git clone <repo> && cd CLUE && git checkout doccl
#     bash scripts/setup_vastai.sh
#
# Sizing is via env (forwarded to setup_remote.sh). Tune to the card you rented:
#     24 GB card:  BATCH_SIZE=8  JOBS_PER_GPU=1  NUM_WORKERS=2   bash scripts/setup_vastai.sh
#     48 GB card:  BATCH_SIZE=16 JOBS_PER_GPU=2  NUM_WORKERS=2   bash scripts/setup_vastai.sh
#     tight VRAM:  GRAD_CKPT=true BATCH_SIZE=2 JOBS_PER_GPU=1     bash scripts/setup_vastai.sh
# Re-run after any interruption: completed runs are pulled from R2 and skipped.

set -uo pipefail
cd "$(dirname "$0")/.."

c_bold=$'\e[1m'; c_grn=$'\e[32m'; c_red=$'\e[31m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
info(){ echo "${c_bold}==>${c_off} $*"; }
ok(){ echo "  ${c_grn}OK${c_off} $*"; }
warn(){ echo "  ${c_yel}!${c_off} $*"; }
die(){ echo "${c_red}ERROR:${c_off} $*" >&2; exit 1; }

info "Vast.ai pre-flight (DocCL grid)"

# ── GPU usability pre-flight (the trap that cost hours: CUDA-driver mismatch -> CPU) ──
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — this is not a GPU instance."
HOST_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
info "GPU(s) on this Vast.ai box:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader | sed 's/^/      /'

# Free-VRAM sanity: a dirty/shared box may have a foreign process eating the card.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "${FREE_MIB:-}" ] && [ -n "${TOTAL_MIB:-}" ]; then
  ok "VRAM free: ${FREE_MIB} / ${TOTAL_MIB} MiB"
  # If less than ~40% is free, something else is on the card — warn loudly.
  if [ "$FREE_MIB" -lt $(( TOTAL_MIB * 4 / 10 )) ]; then
    warn "Most VRAM is already in use by another process (foreign/shared box?)."
    warn "Check: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
    warn "If it's not yours and unkillable, this instance is crippled — reprovision."
  fi
fi

# Hand off to the single source of truth. setup_remote.sh enforces the HARD CUDA gate
# (aborts unless ALLOW_CPU=1), does interpreter discovery + dep install + verification,
# configures R2 durable resume, and launches the multi-GPU grid with the heartbeat.
info "Handing off to scripts/setup_remote.sh (CUDA gate, deps, R2 resume, grid launch) ..."
exec bash scripts/setup_remote.sh
