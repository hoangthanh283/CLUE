#!/usr/bin/env bash
# v24 local (2026-11-16, plan rippling-pondering-comet): the free card's share of the budgeted
# queue. One run at a time, CIL-CORD baselines under the clean protocol (the cells that are
# legacy-protocol in tab:main), seed 42 of each method first. The rented box (v25) works the
# same list from the other end. Waits for the driver lock (the image audit holds it) and stops
# starting runs at STOP_AT (epoch seconds; default: 24 h after launch).
#   setsid nohup bash scripts/rca/clean_protocol_v24_local_2026-11.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
STOP_AT=${STOP_AT:-$(( $(date +%s) + 24 * 3600 ))}
exec 9>$P/.driver.lock
flock 9   # blocks until the other local driver is done: one dataset-building job at a time
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
L="model=layoutlmv3_base"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <name> <args...>   (v22's, plus the stop time and a skip file the remote side can set)
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  [ -e $P/.skip_local/$NAME ] && return 0          # the rented box already took this run
  [ $(( $(date +%s) + 9000 )) -gt $STOP_AT ] && return 0   # a CIL-CORD run takes about 2 h (2.5 h reserved)
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
echo "$(date '+%m-%d %H:%M') v24 local driver running (pid $$): CIL-CORD ER-250 / DER++-250 / LwF / CL-LoRA x3, stop $(date -d @$STOP_AT '+%m-%d %H:%M')" >>$Q
for s in 42 7 123; do
  run cil_cord_er250_seed$s method=er method.buffer_size=250 +method.run_alias=er250 method.epochs=10 scenario=cil_cord seed=$s $L
  run cil_cord_der_pp_seed$s method=der_pp method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L
  run cil_cord_lwf_seed$s method=lwf method.epochs=10 scenario=cil_cord seed=$s $L
  run cil_cord_cl_lora_seed$s method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L
done
echo "$(date '+%m-%d %H:%M') v24 done" >>$Q
