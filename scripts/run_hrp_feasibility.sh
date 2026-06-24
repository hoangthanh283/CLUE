#!/usr/bin/env bash
# One-command setup + run for the HRP (Hybrid-Routed Prompt) feasibility study.
#
# Runs the controlled routing ablation — hrp router={dense,sparse,hybrid} — plus two
# baselines (der_pp, doccl) on the DIL scenario (FUNSD->SROIE->CORD), then prints the
# per-variant routing hit-rate (the primary go/no-go metric) and the AA/BWT table.
#
# Designed for a fresh GPU container (e.g. an L40 on vast.ai) with a
# pytorch/pytorch:*-cuda*-runtime image. From the repo root:
#
#     git clone <repo> && cd CLUE && git checkout doccl && bash scripts/run_hrp_feasibility.sh
#
# Resume-safe: a variant whose results/<dir>/.done exists is skipped on re-run.
# Tunables (env): SEED (42), EPOCHS (3), BATCH_SIZE (8), ROUTERS, BASELINES, SCENARIO (dil).
set -uo pipefail
cd "$(dirname "$0")/.."

c_bold=$'\e[1m'; c_grn=$'\e[32m'; c_red=$'\e[31m'; c_yel=$'\e[33m'; c_off=$'\e[0m'
info(){ echo "${c_bold}==>${c_off} $*"; }
ok(){ echo "  ${c_grn}OK${c_off} $*"; }
warn(){ echo "  ${c_yel}!${c_off} $*"; }
die(){ echo "${c_red}ERROR:${c_off} $*" >&2; exit 1; }

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SCENARIO="${SCENARIO:-dil}"
ROUTERS="${ROUTERS:-dense sparse hybrid}"
BASELINES="${BASELINES:-der_pp doccl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# ── 0. Minimal-image tools ───────────────────────────────────────────────────────
need_apt=""
for t in git; do command -v "$t" >/dev/null 2>&1 || need_apt="$need_apt $t"; done
if [ -n "$need_apt" ] && command -v apt-get >/dev/null 2>&1; then
  info "Installing:$need_apt"; apt-get update -qq 2>/dev/null && apt-get install -y -qq $need_apt 2>/dev/null || true
fi

# ── 1. GPU check ─────────────────────────────────────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — are you on a GPU instance?"
info "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/      /'

# ── 2. Python deps (reuse the image's torch; install only what's missing) ─────────
PY=""; CUDA_OK=0
for cand in "${PYTHON:-}" /opt/conda/bin/python python3 python; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
  "$cand" -c 'import torch' 2>/dev/null || continue
  if "$cand" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    PY="$cand"; CUDA_OK=1; break
  fi
  [ -z "$PY" ] && PY="$cand"
done

if [ -n "$PY" ]; then
  info "Python: $($PY --version 2>&1) ($PY)"
  if [ "$CUDA_OK" = "1" ]; then ok "CUDA available"
  elif [ "${ALLOW_CPU:-0}" = "1" ]; then warn "no CUDA, ALLOW_CPU=1 — running on CPU (slow)"
  else die "torch cannot see the GPU. Pick an image whose CUDA matches the host driver, or ALLOW_CPU=1."; fi
  if "$PY" -c 'import doccl, hydra, transformers, seqeval, datasets' 2>/dev/null; then
    ok "deps already present — skipping pip install"
  else
    info "Installing project deps (pip install -e ., torch already satisfied) ..."
    "$PY" -m pip install -q -e . 2>&1 | tail -3 || die "pip install -e . failed"
    "$PY" -c 'import doccl, hydra, transformers, seqeval, datasets' 2>/dev/null || die "deps still missing"
  fi
else
  warn "torch not found — installing locked stack via uv"
  command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; }
  uv sync --extra dev || die "uv sync failed"; PY=".venv/bin/python"
fi
ok "interpreter: $PY"

run(){  # run <method> <router-or-none> <out_subdir>
  local method="$1" router="$2" sub="$3"
  local out="results/$sub"
  if [ -f "$out/.done" ]; then ok "skip $sub (.done exists)"; return; fi
  local extra="" rtag=""
  [ "$router" != "none" ] && { extra="method.router=$router"; rtag="router=$router "; }
  info "RUN $sub  (method=$method ${rtag}epochs=$EPOCHS bs=$BATCH_SIZE)"
  "$PY" scripts/train.py method="$method" scenario="$SCENARIO" seed="$SEED" \
    method.epochs="$EPOCHS" training.batch_size="$BATCH_SIZE" \
    wandb.mode="$WANDB_MODE" output_dir="$out" $extra \
    || { warn "$sub FAILED (continuing)"; return; }
  touch "$out/.done"
  ok "$sub done"
}

# ── 3. The runs: 3 routers + baselines ───────────────────────────────────────────
info "HRP feasibility on scenario=$SCENARIO seed=$SEED"
for R in $ROUTERS;  do run hrp "$R" "hrp_${R}"; done
for M in $BASELINES; do run "$M" none "${M}"; done

# ── 4. Results: routing hit-rate (primary) + AA/BWT (secondary) ───────────────────
echo; info "${c_bold}ROUTING HIT-RATE (primary go/no-go: hybrid >= dense)${c_off}"
for R in $ROUTERS; do
  f=$(find "results/hrp_${R}" -name routing.json 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    hr=$("$PY" -c "import json;d=json.load(open('$f'));print(f\"{d['overall_hit_rate']:.3f}\")" 2>/dev/null)
    echo "      router=${R}: overall hit-rate = ${hr}"
  else
    warn "router=${R}: no routing.json (run may have failed)"
  fi
done

echo; info "${c_bold}AA / BWT table${c_off}"
"$PY" scripts/analyze_results.py --source local 2>&1 | tail -25 || warn "analyze_results.py failed"
echo; ok "Done. Per-run details: results/{hrp_dense,hrp_sparse,hrp_hybrid,der_pp,doccl}/"
echo "      Routing JSON:    results/hrp_*/${SCENARIO}_hrp_seed${SEED}/routing.json"
