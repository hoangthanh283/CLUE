#!/usr/bin/env bash
# Vast.ai L40 launcher — the SECOND of two independent per-machine grid scripts.
#
# This box owns a DISJOINT scenario partition so it never duplicates the RTX 6000 Ada's
# work (the Ada runs `SCENARIOS="cil_cord dil"` via setup_remote.sh; this box runs the
# remaining three). Because the two job lists never intersect, the grid's lock-free R2
# resume can't waste compute on a collision — partitioning IS the coordination.
#
# It is a THIN wrapper: it only sets this box's partition + L40 sizing, prints the vast.ai
# specifics, then `exec`s setup_remote.sh so ALL the real logic (GPU check, dep install,
# W&B/R2 cred prompts, multi-GPU launch, durable resume, heartbeat) has one source of truth.
#
# ── Vast.ai instance setup ───────────────────────────────────────────────────────────
#   IMAGE:   pytorch/pytorch:2.x-cuda12.x-cudnn9-runtime
#            Match cuda12.x to the host driver (nvidia-smi "CUDA Version"); setup_remote.sh
#            HARD-STOPS on a torch/driver CUDA mismatch and prints the exact fix.
#   ONSTART (or the first SSH command):
#            git clone <repo> && cd CLUE && git checkout doccl && bash scripts/setup_vastai.sh
#   SECRETS: pass W&B + Cloudflare R2 creds as instance ENV (-e) at creation, NOT a .env on
#            disk. setup_remote.sh reads them from the environment and never writes them out:
#              -e WANDB_API_KEY=...  -e WANDB_PROJECT=CL4IE
#              -e R2_ACCESS_KEY_ID=...  -e R2_SECRET_ACCESS_KEY=...  -e R2_ENDPOINT=...
#              -e R2_BUCKET=doccl-results
#            (Use the SAME R2 bucket as the Ada — disjoint SCENARIOS keep their work apart;
#             shared .done markers just let either box skip anything already finished.)
#
# Re-run after a spot-kill: completed runs are pulled from R2 and skipped. Override any
# default below from the environment (e.g. SCENARIOS=... to repartition).
set -uo pipefail
cd "$(dirname "$0")/.."

c_bold=$'\e[1m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
info(){ echo "${c_bold}==>${c_off} $*"; }
warn(){ echo "  ${c_yel}!${c_off} $*"; }

# ── This box's partition (disjoint from the Ada's "cil_cord dil") ────────────────────
export SCENARIOS="${SCENARIOS:-mixed dil_xlingual cil_wildreceipt}"
# The DocCL depth-ablation is pinned to cil_cord, which the Ada owns — so this box does
# NOT run it (would otherwise need cil_cord, breaking the partition).
export RUN_ABLATION="${RUN_ABLATION:-0}"
# Single-task baselines are scenario-INDEPENDENT — they run on exactly ONE box (this one,
# the L40). Without this the Ada AND the L40 would BOTH run all 15, wasting compute (R2
# resume can't dedup them: both start before the first .done syncs). The Ada launch sets
# RUN_SINGLETASK=0 to leave them here.
export RUN_SINGLETASK="${RUN_SINGLETASK:-1}"

# ── L40 (~44 GB usable) sizing — fill the card WITHOUT OOM; 5.8 GB watchdog cap is 2060-only ─
export BATCH_SIZE="${BATCH_SIZE:-16}"     # 32 also fine for non-replay methods
# MEASURED ~17 GB/job at bs16+AMP on LayoutLMv3 (the forward_with_prompts path peaks
# highest), NOT the ~8 GB first guessed -> 3 jobs (51 GB) OOMs a 44 GB L40. 2 jobs (~34 GB)
# fits with ~10 GB headroom. Bump back to 3 only after dropping BATCH_SIZE or adding GRAD_CKPT.
export JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
export AMP="${AMP:-1}"                     # bf16 (L40 is Ada-class); AMP=0 for exact fp32

info "Vast.ai L40 grid launcher"
info "  partition SCENARIOS='${SCENARIOS}'  RUN_ABLATION=${RUN_ABLATION}  (disjoint from the Ada)"
info "  L40 sizing: BATCH_SIZE=${BATCH_SIZE} JOBS_PER_GPU=${JOBS_PER_GPU} bf16=$([ "${AMP}" = "1" ] && echo ON || echo OFF)"
warn "rented box: setup_remote.sh warns loudly if <40% VRAM is free (a foreign process on a dirty instance)."
echo

# Hand off to the single source of truth (GPU check, deps, creds, launch, resume).
exec bash scripts/setup_remote.sh
