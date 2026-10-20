#!/bin/bash
# Pass 2b consolidated: single serial queue using flock to prevent concurrent instances.
# Probes 4 already-dumped CoLaR cells + 3 LexSlot cells (trains them first).
set -u
cd "$(dirname "$0")/../.."
LOCK=/tmp/claude-1000/pass2b.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another pass2b instance already running (lock $LOCK) — exit" >&2
  exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_single.log
mkdir -p results/logs_vision
echo "=== PASS 2B SINGLE $(date) pid=$$ ===" >>"$LOG"

probe() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then
    echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ ! -f "results/${name}/final_model.pt" ]; then
    echo "$(date +%H:%M) MISS ${name}" >>"$LOG"; return; fi
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
    >"results/logs_vision/${name}_probe.log" 2>&1
  if [ -f "results/${name}/probes.json" ]; then
    echo "$(date +%H:%M) DONE ${name}" >>"$LOG"
  else
    echo "$(date +%H:%M) FAIL ${name} (see ${name}_probe.log)" >>"$LOG"
  fi
}

train_then_probe_lexslot() {
  local name="$1"; local scen="$2"
  if [ -f "results/${name}/probes.json" ]; then echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ ! -f "results/${name}/final_model.pt" ]; then
    echo "$(date +%H:%M) TRAIN ${name}" >>"$LOG"
    uv run python scripts/train.py method=lexslot scenario="${scen}" model=layoutlmv3_base \
      seed=42 training.batch_size=2 training.gradient_checkpointing=true \
      training.num_workers=0 wandb.mode=offline +method.dump_final=true \
      hydra.run.dir="results/${name}" >"results/logs_vision/${name}_train.log" 2>&1
    if [ ! -f "results/${name}/final_model.pt" ]; then
      echo "$(date +%H:%M) TRAIN_FAIL ${name}" >>"$LOG"; return; fi
  fi
  probe "$name" "$scen"
}

# ---- CoLaR cells (already dumped) ----
probe dil_colar_pp_seed42                    dil
probe cil_cord_colar_pp_seed42               cil_cord
probe mixed_colar_pp_seed42                  mixed
probe cil_imagenet_r_colar_pp_seed42_vit     cil_imagenet_r

# ---- LexSlot cells (need training) ----
train_then_probe_lexslot dil_lexslot_pp_seed42       dil
train_then_probe_lexslot cil_cord_lexslot_pp_seed42  cil_cord
train_then_probe_lexslot mixed_lexslot_pp_seed42     mixed

echo "=== PASS 2B SINGLE COMPLETE $(date) ===" >>"$LOG"
