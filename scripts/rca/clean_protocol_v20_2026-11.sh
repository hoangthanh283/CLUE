#!/usr/bin/env bash
# Clean-protocol queue v20 (2026-11-15): the two document datasets the paper does not cover yet.
# The document setup is five datasets (FUNSD, CORD, SROIE, WildReceipt, XFUND); FUNSD/CORD/SROIE have
# clean cells, WildReceipt and XFUND only have June-2026 legacy baselines and no CoLaR / CoLaR-SBR run.
#   W. cil_wildreceipt (class-incremental, 4 sessions): CoLaR-SBR, CoLaR, joint, naive, ER
#   X. dil_xlingual   (XFUND, fixed labels, one language per session): the same five
# Order: smokes (must write metrics and peak < 5000 MB), then seed 42 of every cell on both streams,
# then seeds 7 and 123, so a one-seed column exists as early as possible.
# Recipe = the LayoutLMv3 clean recipe of v14 (batch 2, replay batch 4, fp32, k4, r128, 50 docs/session).
#   setsid nohup bash scripts/rca/clean_protocol_v20_2026-11.sh [pid-to-wait-for] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
exec 9>$P/.driver.lock
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; fi
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
L="model=layoutlmv3_base"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
# a smoke passes when it wrote metrics.json and stayed under 5000 MB of VRAM
smoke_ok() { local n; for n in "$@"; do uv run python -c "import json,sys,glob; f=glob.glob(sys.argv[1]+'*/metrics.json'); sys.exit(0 if f and json.load(open(f[0])).get('peak_gpu_mem_mb', 1e9) < 5000 else 1)" "$P/$n" 9>&- || return 1; done; }
cells() {  # cells <scenario> <seed>
  local SC=$1 s=$2
  run ${SC}_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=$SC seed=$s $L
  run ${SC}_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=$SC seed=$s $L
  run ${SC}_joint_seed$s method=joint method.epochs=100 scenario=$SC seed=$s $L
  run ${SC}_naive_seed$s method=naive method.epochs=10 scenario=$SC seed=$s $L
  run ${SC}_er_seed$s method=er method.epochs=10 scenario=$SC seed=$s $L
}
echo "$(date '+%m-%d %H:%M') v20 driver running (pid $$)" >>$Q
OK=""
for SC in cil_wildreceipt dil_xlingual; do
  SM="${SC}_smoke_colar_sbr_e1_seed42 ${SC}_smoke_joint_e1_seed42"
  run ${SC}_smoke_colar_sbr_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_e1 method.docs_per_task=50 $LR1 scenario=$SC seed=42 $L
  run ${SC}_smoke_joint_e1_seed42 method=joint +method.run_alias=smoke_joint_e1 method.epochs=1 scenario=$SC seed=42 $L
  if smoke_ok $SM; then OK="$OK $SC"; cells $SC 42
  else echo "$(date '+%m-%d %H:%M') SKIP  $SC block (a smoke wrote no metrics or peaked >= 5000 MB)" >>$Q; fi
done
for s in 7 123; do for SC in $OK; do cells $SC $s; done; done
echo "$(date '+%m-%d %H:%M') v20 done" >>$Q
