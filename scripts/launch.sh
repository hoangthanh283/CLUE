#!/usr/bin/env bash
# One-command launcher for the DocCL grid on a GPU box (e.g. 2x RTX 4090).
#
#   bash scripts/launch.sh
#
# It does EVERYTHING:
#   1) prompts for credentials (W&B + Cloudflare R2) — held in memory only, never
#      written to a persistent file on this (possibly rented/shared) machine;
#   2) verifies Docker + the NVIDIA Container Toolkit (--gpus all);
#   3) builds the doccl-grid image if it isn't already built;
#   4) runs the multi-GPU grid, forwarding creds as runtime -e env vars (in-process),
#      with durable resume (R2) + live progress heartbeat.
#
# Re-run it any time: completed runs are skipped (resume from R2), so a spot-killed
# instance just continues. Secrets are prompted fresh each run — nothing to leave behind.
#
# Non-interactive: export any of the prompted vars beforehand to skip its prompt
# (e.g. in CI). To run WITHOUT durable sync, leave the R2 prompts blank.

set -uo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-doccl-grid}"
DOCKERFILE="docker/Dockerfile"

c_bold=$'\e[1m'; c_dim=$'\e[2m'; c_grn=$'\e[32m'; c_red=$'\e[31m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
info(){ echo "${c_bold}==>${c_off} $*"; }
ok(){ echo "  ${c_grn}OK${c_off} $*"; }
warn(){ echo "  ${c_yel}!${c_off} $*"; }
die(){ echo "${c_red}ERROR:${c_off} $*" >&2; exit 1; }

# ── 0. Preconditions ────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker first."
info "Checking NVIDIA GPU access in Docker (--gpus all) ..."
if docker run --rm --gpus all nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi \
     --query-gpu=name,memory.total --format=csv,noheader >/tmp/_gpucheck 2>/tmp/_gpuerr; then
  ok "GPUs visible to Docker:"; sed 's/^/      /' /tmp/_gpucheck
  NGPU=$(wc -l < /tmp/_gpucheck)
else
  warn "Could not run a GPU container. The NVIDIA Container Toolkit may be missing."
  sed 's/^/      /' /tmp/_gpuerr | head -3
  die "Install it (see docker/README.md 'Prerequisites'), then re-run. (--gpus all must work.)"
fi
rm -f /tmp/_gpucheck /tmp/_gpuerr

# ── 1. Prompt for credentials (memory only) ─────────────────────────────────────
prompt_secret(){ # <var> <label>  -> sets the var if empty (silent input)
  local var="$1" label="$2" cur="${!1:-}"
  [ -n "$cur" ] && { ok "$label provided via env"; return 0; }
  read -r -s -p "  ${label}: " val; echo
  printf -v "$var" '%s' "$val"
}
prompt_plain(){ # <var> <label> <default>
  local var="$1" label="$2" def="${3:-}" cur="${!1:-}"
  [ -n "$cur" ] && { ok "$label = $cur (from env)"; return 0; }
  read -r -p "  ${label}${def:+ [$def]}: " val
  printf -v "$var" '%s' "${val:-$def}"
}

info "Weights & Biases (press Enter at the key prompt to log OFFLINE):"
prompt_secret WANDB_API_KEY "W&B API key"
prompt_plain  WANDB_PROJECT "W&B project" "CL4IE"
if [ -z "${WANDB_API_KEY:-}" ]; then WANDB_MODE=offline; warn "no W&B key -> WANDB_MODE=offline"; else WANDB_MODE="${WANDB_MODE:-online}"; fi

info "Cloudflare R2 durable resume (press Enter at all three to DISABLE sync):"
prompt_secret R2_ACCESS_KEY_ID     "R2 Access Key ID"
if [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
  prompt_secret R2_SECRET_ACCESS_KEY "R2 Secret Access Key"
  prompt_plain  R2_ENDPOINT          "R2 Endpoint URL" ""
  prompt_plain  R2_BUCKET            "R2 Bucket" "doccl-results"
  [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_ENDPOINT:-}" ] \
    || die "R2 secret + endpoint are required when an access key is given."
  ok "Durable resume ON -> R2 bucket '${R2_BUCKET}' (creds in memory, not written to disk)"
  warn "Reminder: scope this R2 token to the bucket + DELETE it after the grid finishes."
else
  warn "No R2 creds -> durable resume OFF. Results live on this instance's disk only;"
  warn "a spot-kill would lose progress. (Recommended: provide R2 creds.)"
fi

# ── 2. Grid sizing ──────────────────────────────────────────────────────────────
info "Grid parallelism:"
prompt_plain GPUS         "GPU ids"            "$(seq -s' ' 0 $((NGPU-1)))"
prompt_plain JOBS_PER_GPU "Jobs per GPU"       "2"
prompt_plain BATCH_SIZE   "Batch size"         "16"

# ── 3. Build image if missing ───────────────────────────────────────────────────
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "Image '$IMAGE' already built (use FORCE_BUILD=1 to rebuild)."
  [ "${FORCE_BUILD:-0}" = "1" ] && { info "FORCE_BUILD=1 -> rebuilding ..."; docker build -t "$IMAGE" -f "$DOCKERFILE" . || die "build failed"; }
else
  info "Building image '$IMAGE' (one-time, a few minutes) ..."
  docker build -t "$IMAGE" -f "$DOCKERFILE" . || die "docker build failed"
  ok "image built"
fi

# ── 4. Confirm + launch ─────────────────────────────────────────────────────────
echo
info "Launching grid:  GPUS='${GPUS}'  JOBS_PER_GPU=${JOBS_PER_GPU}  BATCH_SIZE=${BATCH_SIZE}  W&B=${WANDB_MODE}  sync=$([ -n "${R2_ACCESS_KEY_ID:-}" ] && echo ON || echo OFF)"
echo "  ${c_dim}Watch progress: this terminal streams a heartbeat + filtered training lines.${c_off}"
echo "  ${c_dim}Resume: re-run this script on a fresh instance; completed runs are skipped.${c_off}"
echo

# Forward creds via -e (value from this shell's env -> container process env only; never
# on the container/instance disk). -e VAR with no '=' forwards the current value.
docker run --rm --gpus all \
  -e WANDB_API_KEY -e WANDB_PROJECT="$WANDB_PROJECT" -e WANDB_MODE="$WANDB_MODE" \
  -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e R2_ENDPOINT -e R2_BUCKET="${R2_BUCKET:-doccl-results}" \
  -e GPUS="$GPUS" -e JOBS_PER_GPU="$JOBS_PER_GPU" -e BATCH_SIZE="$BATCH_SIZE" \
  -v "$PWD/.hf_cache:/workspace/.hf_cache" \
  --entrypoint bash "$IMAGE" -c "bash scripts/run_grid_multigpu.sh"

rc=$?
echo
if [ $rc -eq 0 ]; then
  info "${c_grn}Grid finished.${c_off} Aggregate results with:"
  echo "  docker run --rm --gpus all -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e R2_ENDPOINT \\"
  echo "    --entrypoint bash $IMAGE -c \"bash scripts/run_grid_multigpu.sh\"   # idempotent: pulls + verifies all done"
  echo "  # then inside the container or after syncing results back:"
  echo "  python scripts/analyze_results.py && python scripts/ingest_to_thesis.py"
else
  warn "Grid exited rc=$rc. Re-run this script to resume (R2 pull skips completed runs)."
fi
exit $rc
