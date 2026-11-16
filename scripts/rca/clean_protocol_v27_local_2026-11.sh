#!/usr/bin/env bash
# v27 local (2026-11-16): light Mixed re-runs on the RTX 2060 while the rented box does the rest.
# One run at a time. Starts only when no train.py is alive and the driver lock is free, so it
# never overlaps another local queue. The same runs are marked as taken in the box's claims dir.
#   STOP_AT=<epoch> setsid nohup bash scripts/rca/clean_protocol_v27_local_2026-11.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
STOP_AT=${STOP_AT:-$(( $(date +%s) + 19 * 3600 ))}
exec 9>$P/.driver.lock
while :; do
  while pgrep -f "scripts/train.py" >/dev/null; do sleep 30 9>&-; done
  flock -n 9 && ! pgrep -f "scripts/train.py" >/dev/null && break
  flock -u 9; sleep 30 9>&-
done
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
L="model=layoutlmv3_base"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <hours reserved> <name> <args...>
  local H=$1 NAME=$2; shift 2
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  [ $(( $(date +%s) + H * 3600 )) -gt $STOP_AT ] && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
echo "$(date '+%m-%d %H:%M') v27 local driver running (pid $$): Mixed LwF / CoLaR / IS3, stop $(date -d @$STOP_AT '+%m-%d %H:%M')" >>$Q
for s in 7 123; do run 2 mixed_lwf_seed$s method=lwf method.epochs=10 scenario=mixed seed=$s $L; done
for s in 42 7 123; do run 3 mixed_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in 42 7 123; do run 3 mixed_is3_seed$s method=is3 method.epochs=10 scenario=mixed seed=$s $L; done
echo "$(date '+%m-%d %H:%M') v27 done" >>$Q
