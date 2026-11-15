#!/usr/bin/env bash
# Clean-protocol queue v19 (2026-11-15, Dev): the BROS block of v17 with a recipe that fits 5 GB, then v13's list.
# Why v17's BROS smokes OOMed (11-15 09:50/09:53, both at 5.0 GiB allocated, first replay step of session 2):
#   BROS has no gradient checkpointing, CoLaR's latent replay runs without it anyway, and BROS's bbox
#   positional attention keeps a (B, L, L, 64) fp32 tensor per layer (256 MiB per layer at B=4, L=512).
#   Clean fp32 with batch 2 + replay batch 4 = six unchecked sequences through the plastic layers.
#   The earlier BROS CoLaR runs (results/*_colar_e10_*_bros_*) used batch 1 + replay batch 2 under AMP:
#   peak 2717 MB; fp32 roughly doubles the activations, about 4.1 GB expected.
# Recipe here: training.batch_size=1, method.replay_batch_size=2 (same 2:1 replay ratio as the LayoutLMv3
#   recipe), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; still fp32 (clean protocol). Batch size
#   changes, so every BROS run name carries _b1 (never the v17 names). Both smokes must write
#   full_label_f1 AND report peak_gpu_mem_mb < 5000 before any 3-seed run starts.
#   setsid nohup bash scripts/rca/clean_protocol_v19_2026-11.sh [pid-to-wait-for] [its-run-name] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; ORPHAN=${2:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60 9>&-; done; fi
if [ -n "$ORPHAN" ]; then
  if ls $P/${ORPHAN}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $ORPHAN" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $ORPHAN" >>$Q; fi
fi
# v17's T with training.batch_size=1 (the only change)
T="training.batch_size=1 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
smoke_ok() { local n; for n in "$@"; do grep -qs full_label_f1 $P/$n*/metrics.json || return 1; done; }
peak_ok() { local n; for n in "$@"; do uv run python -c "import json,sys,glob; f=glob.glob(sys.argv[1]+'*/metrics.json'); sys.exit(0 if f and json.load(open(f[0])).get('peak_gpu_mem_mb', 1e9) < 5000 else 1)" "$P/$n" 9>&- || return 1; done; }
S="42 7 123"
echo "$(date '+%m-%d %H:%M') v19 driver running (pid $$)" >>$Q
# B. BROS controls on cil_cord, recipe b1 (batch 1, replay batch 2, fp32): CoLaR-SBR, oracle-pre, constant
#    background, plain CoLaR; smokes first (VRAM-gated), oracle also gated on relabelled tokens > 0.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="scenario=cil_cord model=bros_base"
SM="cil_cord_smoke_colar_r_oracle_pre_bros_b1_e1_seed42 cil_cord_smoke_colar_sbr_const_o_bros_b1_e1_seed42"
run cil_cord_smoke_colar_r_oracle_pre_bros_b1_e1_seed42 method=colar_r_oracle_pre method.run_alias=smoke_colar_r_oracle_pre_bros_b1_e1 method.docs_per_task=50 $LR1 $B seed=42
run cil_cord_smoke_colar_sbr_const_o_bros_b1_e1_seed42 method=colar_sbr_const_o method.run_alias=smoke_colar_sbr_const_o_bros_b1_e1 method.docs_per_task=50 $LR1 $B seed=42
if smoke_ok $SM && peak_ok $SM; then
  for s in $S; do run cil_cord_colar_sbr_bros_b1_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros_b1 method.docs_per_task=50 $LR $B seed=$s; done
  if grep -Eqs 'colar_r\[oracle\]: refreshed .* [1-9][0-9]* tokens relabelled' $P/cil_cord_smoke_colar_r_oracle_pre_bros_b1_e1_seed42.log; then
    for s in $S; do run cil_cord_colar_r_oracle_pre_bros_b1_seed$s method=colar_r_oracle_pre method.run_alias=colar_r_oracle_pre_bros_b1 method.docs_per_task=50 $LR $B seed=$s; done
  else echo "$(date '+%m-%d %H:%M') SKIP  BROS b1 oracle block (smoke relabelled 0 tokens)" >>$Q; fi
  for s in $S; do run cil_cord_colar_sbr_const_o_bros_b1_seed$s method=colar_sbr_const_o method.run_alias=colar_sbr_const_o_bros_b1 method.docs_per_task=50 $LR $B seed=$s; done
  for s in $S; do run cil_cord_colar_bros_b1_seed$s method=colar +method.run_alias=colar_bros_b1 method.docs_per_task=50 $LR $B seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  BROS b1 block (a smoke has no full_label_f1 or peaked >= 5000 MB)" >>$Q; fi
unset PYTORCH_CUDA_ALLOC_CONF
echo "$(date '+%m-%d %H:%M') v19 BROS block done; handing over to v13 list" >>$Q
exec bash scripts/rca/clean_protocol_v13_2026-11.sh
