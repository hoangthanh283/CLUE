#!/usr/bin/env bash
# One-command setup + run for a GPU CONTAINER instance (e.g. ViLao 2x RTX 4090).
#
# Use this when you are already INSIDE a container/VM with the GPUs visible (no
# docker-in-docker) — i.e. you picked a PyTorch/CUDA template and SSH'd in. It:
#   1) checks the GPUs are visible (nvidia-smi),
#   2) installs the Python deps on top of the template's existing torch (pip install -e .),
#      so the template's proven torch+CUDA+driver are reused (no CUDA-version mismatch),
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
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || die "$PY not found"
info "Python: $($PY --version 2>&1)"
if "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  ok "template torch present + CUDA OK ($("$PY" -c 'import torch;print(torch.__version__)')) — reusing it"
  info "Installing remaining deps (pip install -e ., torch already satisfied) ..."
  "$PY" -m pip install -q -e . 2>&1 | tail -3 || die "pip install -e . failed"
else
  warn "no working torch in the base image — installing the full locked stack via uv (may pull a CUDA build that needs a recent driver)"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  uv sync --extra dev || die "uv sync failed"
  PY=".venv/bin/python"
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available after install"' \
    || die "torch installed but CUDA not available — pick a template whose driver matches (see README)"
fi
# Tell run_grid_multigpu.sh which interpreter to use (uv .venv or the template's python3).
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
prompt_plain GPUS         "GPU ids"      "$(seq -s' ' 0 $((NGPU-1)))"
prompt_plain JOBS_PER_GPU "Jobs per GPU" "2"
prompt_plain BATCH_SIZE   "Batch size"   "16"
export GPUS JOBS_PER_GPU BATCH_SIZE

echo
info "Launching grid: GPUS='${GPUS}' JOBS_PER_GPU=${JOBS_PER_GPU} BATCH_SIZE=${BATCH_SIZE} W&B=${WANDB_MODE} sync=$([ -n "${R2_ACCESS_KEY_ID:-}" ] && echo ON || echo OFF)"
echo "  ${c_dim}heartbeat + filtered training lines stream below; re-run this script to resume.${c_off}"
echo
exec bash scripts/run_grid_multigpu.sh
