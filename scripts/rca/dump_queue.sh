#!/bin/bash
# Pass 2b: re-run 4 lead CoLaR cells with dump_final=true, then run the ridge probe.
# Resume-safe: skips any cell that already has final_model.pt + probes.json.
# Resource: VRAM/RAM constraints as per CLAUDE.md.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/pass2b_queue.log
mkdir -p results/logs_vision

cell_run() {
  local name="$1"; local scen="$2"; local extra="$3"
  if [ -f "results/${name}/probes.json" ]; then echo "$(date +%H:%M) SKIP ${name}" >>"$LOG"; return; fi
  if [ -f "results/${name}/final_model.pt" ]; then
    echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
    uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
      >>"results/logs_vision/${name}_probe.log" 2>&1
    [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) PROBE_OK ${name}" >>"$LOG" \
      || echo "$(date +%H:%M) PROBE_FAIL ${name}" >>"$LOG"
    return
  fi
  echo "$(date +%H:%M) TRAIN ${name}" >>"$LOG"
  uv run python scripts/train.py method=colar_pp scenario="${scen}" model=layoutlmv3_base \
    seed=42 method.epochs=5 training.batch_size=2 training.gradient_checkpointing=true \
    training.num_workers=0 wandb.mode=offline method.split_layer_k=4 method.rank_r=128 \
    method.docs_per_task=50 method.replay_batch_size=4 method.dump_final=true $extra \
    hydra.run.dir=results/${name}_pass2b >"results/logs_vision/${name}_train.log" 2>&1
  if [ -f "results/${name}_pass2b/final_model.pt" ]; then
    mkdir -p "results/${name}"
    cp -f results/${name}_pass2b/final_model.pt results/${name}/
    cp -f results/${name}_pass2b/final_store.pt results/${name}/
    uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario "${scen}" --k 4 \
      >>"results/logs_vision/${name}_probe.log" 2>&1
    [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
      || echo "$(date +%H:%M) PROBE_FAIL ${name}" >>"$LOG"
  else
    echo "$(date +%H:%M) TRAIN_FAIL ${name}" >>"$LOG"
  fi
}

echo "=== PASS 2B QUEUE $(date) ===" >>"$LOG"
cell_run cil_cord_colar_seed42_k4_d50_r128 cil_cord ""
cell_run mixed_colar_seed42_k4_d50_r128    mixed    ""
cell_run dil_colar_seed42_k4_d50_r128      dil      ""
# ViT ImageNet-R uses d=500, model=vit_b16
if [ ! -f results/cil_imagenet_r_colar500r128_seed42_vit_k4_d500_r128/probes.json ]; then
  name=cil_imagenet_r_colar500r128_seed42_vit_k4_d500_r128
  if [ ! -f results/${name}/final_model.pt ]; then
    echo "$(date +%H:%M) TRAIN ${name}" >>"$LOG"
    uv run python scripts/train.py method=colar_pp scenario=cil_imagenet_r model=vit_b16 \
      seed=42 method.epochs=20 training.batch_size=16 training.amp=true training.num_workers=2 \
      wandb.mode=offline method.split_layer_k=4 method.rank_r=128 method.docs_per_task=500 \
      method.replay_batch_size=16 method.dump_final=true \
      hydra.run.dir=results/${name}_pass2b >"results/logs_vision/${name}_train.log" 2>&1
    cp -f results/${name}_pass2b/final_model.pt results/${name}/ 2>/dev/null
    cp -f results/${name}_pass2b/final_store.pt results/${name}/ 2>/dev/null
  fi
  echo "$(date +%H:%M) PROBE ${name}" >>"$LOG"
  uv run python scripts/rca/probe_cell.py --run "results/${name}" --scenario cil_imagenet_r --k 4 \
    >>"results/logs_vision/${name}_probe.log" 2>&1
  [ -f "results/${name}/probes.json" ] && echo "$(date +%H:%M) DONE ${name}" >>"$LOG" \
    || echo "$(date +%H:%M) PROBE_FAIL ${name}" >>"$LOG"
fi
echo "=== PASS 2B COMPLETE $(date) ===" >>"$LOG"
