#!/usr/bin/env bash
# v28 (2026-11-16 night, plan rippling-pondering-comet): the main-body runs still missing, on the
# rented box, finished by T_STOP at the lowest cost. Measured on this box: a card shared by n runs
# gives each about 1/n of its speed, so cost and time are (sum of GPU work) / (number of cards).
# The scheduler therefore keeps every card full and never starts what cannot finish:
#   each card runs at most JOBS_PER_GPU runs; a free slot on card g scans the list in order and
#   starts the first unstarted run j with
#     (a) memory: known peaks of the runs on g + peak(j) <= CAP          (v26's peak-aware start)
#     (b) time:   now + SLACK * (GPU work left on g + work(j)) <= T_STOP - MARGIN
#   it skips a run that does not fit (no head-of-line blocking) and waits when nothing fits; it
#   exits only when the list is done or when its card is empty and nothing can still finish.
# List order: long runs first (one lands on each card), then the cells by value.
# Recipe (T, LR), run() and claim directories are v25's. Work per class (GPU-hours with a card to
# itself) is in $P/.work_v28 ("class hours", re-read on every scan; measured from the v25 queue log).
#   T_STOP_AT="2026-11-17 00:00 UTC" setsid nohup bash scripts/rca/clean_protocol_v28_remote_2026-11.sh &
#   DRY_RUN=1 bash scripts/rca/clean_protocol_v28_remote_2026-11.sh     # ordered list, work, peak
#   DRY_RUN=2 bash scripts/rca/clean_protocol_v28_remote_2026-11.sh     # raw list (pre-flight check)
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
L="model=layoutlmv3_base"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOT="${SLOT:-main}"
run() {  # run <name> <args...>   (v25's)
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "[v28 $SLOT] $(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&- 8>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "[v28 $SLOT] $(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "[v28 $SLOT] $(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
# ---------------------------------------------------------------- the ordered job list
job() { echo "$1|$2|$3"; }   # class|name|args
jobs() {
  local s m
  # long runs first: with one slot per card starting in turn, each lands on its own card
  job wr_der cil_wildreceipt_der_pp250_seed123 "method=der_pp method.buffer_size=250 scenario=cil_wildreceipt +method.run_alias=der_pp250 method.epochs=10 seed=123 $L"
  for s in 42 7 123; do job cord_der cil_cord_der_pp_seed$s "method=der_pp method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L"; done
  # the two empty cells of the three-repairs table (need the labels_full data change c345507)
  for s in 42 7 123; do job wr_ctrl cil_wildreceipt_colar_r_oracle_pre_seed$s "method=colar_r_oracle_pre method.docs_per_task=50 $LR scenario=cil_wildreceipt seed=$s $L"; done
  for s in 42 7 123; do job wr_ctrl cil_wildreceipt_colar_mib_seed$s "method=colar_mib method.docs_per_task=50 $LR scenario=cil_wildreceipt seed=$s $L"; done
  # the headline column of the comparison table
  for m in coda_prompt is3; do for s in 42 7 123; do job cord_$m cil_cord_${m}_seed$s "method=$m method.epochs=10 scenario=cil_cord seed=$s $L"; done; done
  for s in 42 7; do job cord_lca cil_cord_lca_seed$s "method=lca method.epochs=10 scenario=cil_cord seed=$s $L"; done
  # the Mixed column
  for s in 7 123; do job mixed_er mixed_er_seed$s "method=er method.epochs=10 scenario=mixed seed=$s $L"; done
  for s in 42 7 123; do job mixed_cl_lora mixed_cl_lora_seed$s "method=cl_lora method.epochs=10 scenario=mixed seed=$s $L"; done
  for s in 42 7 123; do job mixed_is3 mixed_is3_seed$s "method=is3 method.epochs=10 scenario=mixed seed=$s $L"; done
  for s in 42 7 123; do job mixed_colar mixed_colar_seed$s "method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L"; done
}
WORK_DEFAULT="wr_der 3.6
cord_der 1.7
wr_ctrl 1.1
cord_coda_prompt 0.5
cord_is3 0.5
cord_lca 0.35
mixed_er 1.3
mixed_cl_lora 0.8
mixed_is3 0.6
mixed_colar 0.6"
peak() {  # peak <method>: MiB a run of this method reaches on this box (nvidia-smi view, rounded up)
  case $1 in
    der_pp) echo 4700 ;; cpfd) echo 5000 ;; er) echo 4400 ;; lwf) echo 3400 ;; is3) echo 3100 ;;
    lca) echo 2600 ;; cl_lora) echo 1900 ;; coda_prompt) echo 1300 ;; *) echo 3000 ;;
  esac
}
method_of() { echo "$1" | sed -E 's/(^|.* )method=([a-z0-9_]+).*/\2/'; }
[ "${DRY_RUN:-0}" = 2 ] && { jobs | awk -F'|' '{print $1 "|-|" $2 "|" $3}'; echo "T|$T"; exit 0; }
if [ "${DRY_RUN:-0}" = 1 ]; then
  jobs | while IFS='|' read -r class name args; do
    echo "$class $(echo "$WORK_DEFAULT" | awk -v c=$class '$1 == c {print $2}') $(peak "$(method_of "$args")") $name"
  done | awk '{printf "%3d  %-18s %4s GPU-h  %5s MiB  %s\n", NR, $1, $2, $3, $4; w += $2} END {printf "total %d runs, %.1f GPU-hours\n", NR, w}'
  jobs | cut -d'|' -f2 | sort | uniq -d | sed 's/^/DUPLICATE /'
  exit 0
fi
# ---------------------------------------------------------------- slots
exec 9>$P/.driver_v28.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another v28 driver holds $P/.driver_v28.lock (pid $$)" >>$Q; exit 1; }
C=${CLAIMS:-$P/.claims_v28_main}; mkdir -p $C; [ -f $C/jobs.txt ] || jobs >$C/jobs.txt
W=$P/.work_v28; [ -f $W ] || echo "$WORK_DEFAULT" >$W
T_STOP=$(date -d "${T_STOP_AT:?T_STOP_AT, e.g. 2026-11-17 00:00 UTC}" +%s); echo $T_STOP >$C/t_stop
CAP=${CAP:-11900}; SLACK=${SLACK:-1.15}; MARGIN=${MARGIN:-600}
work_of() { awk -v c=$1 '$1 == c {h = $2} END {print (h == "" ? 1.0 : h)}' $W; }
card_load() {  # summed known peaks of every train.py on the card, never below what is in use (v26's)
  local pid s=0 used
  for pid in $(pgrep -f '/python[0-9.]* scripts/train.py'); do
    [ "$(tr '\0' '\n' </proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')" = "$1" ] || continue
    s=$(( s + $(peak "$(method_of "$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null)")") ))
  done
  used=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${used:-0}" -gt $s ] && s=$used
  echo $s
}
card_runs() { local f; for f in $C/*/meta; do [ -f "$f" ] && [ ! -f "${f%meta}finished" ] && [ "$(cut -d' ' -f1 "$f")" = "$1" ] && echo "$f"; done; }
work_left() {  # GPU-hours the runs on card <gpu> still need: w - elapsed / (runs on the card), never below 10% of w
  local f n now; now=$(date +%s); n=$(card_runs $1 | wc -l)
  for f in $(card_runs $1); do
    awk -v w="$(work_of "$(cut -d' ' -f2 "$f")")" -v t0="$(cut -d' ' -f3 "$f")" -v now=$now -v n=$n \
      'BEGIN {r = w - (now - t0) / 3600 / n; if (r < 0.1 * w) r = 0.1 * w; print r}'
  done | awk '{s += $1} END {printf "%.3f\n", s + 0}'
}
pick() {  # pick <gpu>: first unstarted run that fits card <gpu> in memory and in time: "class|name|args"
  local class name args load left; load=$(card_load $1); left=$(work_left $1)
  while IFS='|' read -r class name args; do
    [ -d $C/$name ] && continue
    if ls $P/${name}*/metrics.json >/dev/null 2>&1; then mkdir $C/$name 2>/dev/null && touch $C/$name/finished; continue; fi
    [ $(( load + $(peak "$(method_of "$args")") )) -gt $CAP ] && continue
    awk -v now=$(date +%s) -v stop=$T_STOP -v m=$MARGIN -v k=$SLACK -v l=$left -v w="$(work_of $class)" \
      'BEGIN {exit !(now + k * (l + w) * 3600 <= stop - m)}' || continue
    echo "$class|$name|$args"; return
  done <$C/jobs.txt
}
unstarted() { local class name args; while IFS='|' read -r class name args; do [ -d $C/$name ] || { echo x; return; }; done <$C/jobs.txt; }
worker() {  # worker <slot id> <gpu index>
  SLOT=$1; export CUDA_VISIBLE_DEVICES=$2
  local h class name args pid
  exec 8>/tmp/v28_gpu$2.lock   # one start at a time per card
  while [ $(date +%s) -lt $T_STOP ]; do
    [ -z "$(unstarted)" ] && break                       # every run is started or done
    flock 8
    h=$(pick $2)
    if [ -z "$h" ]; then
      flock -u 8
      [ -z "$(card_runs $2)" ] && [ ! -e $C/open ] && break   # empty card and still nothing fits: it never will
      sleep ${SCAN:-30}; continue
    fi
    IFS='|' read -r class name args <<<"$h"
    mkdir $C/$name 2>/dev/null || { flock -u 8; continue; }   # atomic claim: another slot got it first
    echo "$2 $class $(date +%s)" >$C/$name/meta
    run $name $args & pid=$!
    sleep ${HOLD:-25}; flock -u 8; wait $pid   # the python process is visible to card_load before the next start
    touch $C/$name/finished
  done
  echo "[v28 $SLOT] $(date '+%m-%d %H:%M') slot exit" >>$Q
}
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
echo "$(date '+%m-%d %H:%M') v28 remote driver running (pid $$): GPUS=[$GPUS] JOBS_PER_GPU=$JOBS_PER_GPU, $(wc -l <$C/jobs.txt) jobs, stop $(date -d @$T_STOP '+%m-%d %H:%M'), claims in $C" >>$Q
for k in $(seq 0 $((JOBS_PER_GPU - 1))); do
  for g in $GPUS; do worker g${g}s$k $g & sleep 2; done
done
wait
echo "$(date '+%m-%d %H:%M') v28 ALL SLOTS EXITED" >>$Q
