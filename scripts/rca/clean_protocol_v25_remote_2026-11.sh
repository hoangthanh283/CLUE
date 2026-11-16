#!/usr/bin/env bash
# v25 (2026-11-16, plan rippling-pondering-comet): budgeted queue for the rented box. Finish as
# many tab:main runs as possible before T_HOURS of paid time are over (P | d_j = d | sum U_j).
# Online SPT list scheduling: one shared list ordered short -> heavy -> medium (shortest first);
# every slot claims the first unclaimed run that passes
#   (a) its port test (the method's CIL-FUNSD seed-42 run wrote metrics), and
#   (b) the admission test  now + 1.25 x estimate(class) <= T_stop - 15 min.
# Heavy runs sit right after the short ones (least laxity first): started late, a slowdown would
# kill them after hours of paid time. Estimates per class live in $P/.est_v25 ("class hours",
# re-read on every scan: edit it when real run times drift).
# Recipe (T, LR) and run()/claims are v21's / v23's, unchanged. Names are the ones
# scripts/rca/make_clean_tables.py reads.
#   T_HOURS=11.5 GPUS="0 1 2 3 4 5" JOBS_PER_GPU=4 setsid nohup bash scripts/rca/clean_protocol_v25_remote_2026-11.sh &
#   DRY_RUN=1 bash scripts/rca/clean_protocol_v25_remote_2026-11.sh      # ordered list, class, estimate
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
L="model=layoutlmv3_base"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
PORTS="coda_prompt cpfd is3 lca"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOT="${SLOT:-main}"
run() {  # run <name> <args...>   (v21's)
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "[$SLOT] $(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&- 8>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "[$SLOT] $(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "[$SLOT] $(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
# ---------------------------------------------------------------- the ordered job list
# one line per job: class|gate|name|args   (gate "-" = none, else the port whose CIL-FUNSD test gates it)
job() { echo "$1|$2|$3|$4"; }
base() {  # base <class> <scenario> <seed> <method key>: one baseline run, name <scenario>_<key>_seed<s>
  local c=$1 SC=$2 s=$3 m=$4 g=-
  case " $PORTS " in *" $m "*) g=$m ;; esac
  job $c $g ${SC}_${m}_seed$s "method=$m method.epochs=10 scenario=$SC seed=$s $L"
}
jobs() {
  local s m SC
  # ---- short: CIL-FUNSD column. Seed 42 of the four ports first: they are the port tests.
  for m in $PORTS; do job funsd - cil_funsd_${m}_seed42 "method=$m method.epochs=10 scenario=cil_funsd seed=42 $L"; done
  for s in 42 7 123; do
    job funsd - cil_funsd_lwf_seed$s "method=lwf method.epochs=10 scenario=cil_funsd seed=$s $L"
    job funsd - cil_funsd_cl_lora_seed$s "method=cl_lora method.epochs=10 scenario=cil_funsd seed=$s $L"
    # 20 documents = the memory of the CIL-FUNSD CoLaR cells (200 would hold all 149 forms)
    job funsd - cil_funsd_er_b20_seed$s "method=er method.buffer_size=20 +method.run_alias=er_b20 method.epochs=10 scenario=cil_funsd seed=$s $L"
    job funsd - cil_funsd_der_pp_b20_seed$s "method=der_pp method.buffer_size=20 +method.run_alias=der_pp_b20 method.epochs=10 scenario=cil_funsd seed=$s $L"
    job funsd - cil_funsd_colar_sbr_recall_r64_seed$s "$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR scenario=cil_funsd seed=$s $L"
    [ $s = 42 ] || for m in $PORTS; do base funsd cil_funsd $s $m; done
  done
  # ---- heavy: CIL-WildReceipt (empty cells; v21 names and args for ER / DER++-250)
  job wr - cil_wildreceipt_er_seed7 "method=er method.epochs=10 scenario=cil_wildreceipt seed=7 $L"
  for s in 42 7 123; do job wr - cil_wildreceipt_der_pp250_seed$s "method=der_pp method.buffer_size=250 scenario=cil_wildreceipt +method.run_alias=der_pp250 method.epochs=10 seed=$s $L"; done
  for s in 42 7 123; do for m in $PORTS; do base wr cil_wildreceipt $s $m; done; done
  # ---- medium, shortest class first
  for SC in dil mixed; do  # legacy-protocol cells re-run clean: same args, clean flags
    for s in 42 7 123; do
      job $SC - ${SC}_naive_seed$s "method=naive method.epochs=10 scenario=$SC seed=$s $L"
      for m in lwf er der_pp cl_lora $PORTS; do base $SC $SC $s $m; done
      job $SC - ${SC}_colar_seed$s "method=colar method.docs_per_task=50 $LR scenario=$SC seed=$s $L"
      [ $SC = dil ] && job dil - dil_colar_sbr_seed$s "method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L"
      [ $SC = dil ] && job dil - dil_joint_seed$s "method=joint method.epochs=100 scenario=dil seed=$s $L"
    done
  done
  for s in 42 7 123; do for m in $PORTS; do base xfund dil_xlingual $s $m; done; done
  # CIL-CORD: the four ports, then the list the local card works from the top (v24), reversed here
  for s in 42 7 123; do for m in $PORTS; do base cord cil_cord $s $m; done; done
  for s in 123 7 42; do
    job cord - cil_cord_cl_lora_seed$s "method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L"
    job cord - cil_cord_lwf_seed$s "method=lwf method.epochs=10 scenario=cil_cord seed=$s $L"
    job cord - cil_cord_der_pp_seed$s "method=der_pp method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L"
    job cord - cil_cord_er250_seed$s "method=er method.buffer_size=250 +method.run_alias=er250 method.epochs=10 scenario=cil_cord seed=$s $L"
  done
}
EST_DEFAULT="funsd 0.17
wr 4.5
dil 0.8
mixed 1.3
xfund 1.5
cord 2.0"
[ "${DRY_RUN:-0}" = 2 ] && { jobs; echo "T|$T"; exit 0; }   # raw list (the pre-flight check reads it)
if [ "${DRY_RUN:-0}" = 1 ]; then
  jobs | awk -F'|' -v e="$EST_DEFAULT" 'BEGIN {n=split(e, a, /[ \n]/); for (i=1; i<n; i+=2) est[a[i]]=a[i+1]}
       {printf "%3d  %-6s %4sh  gate %-12s %s\n", NR, $1, est[$1], $2, $3; c[$1]++; h+=est[$1]}
       END {for (k in c) printf "%s: %d runs\n", k, c[k]; printf "total %d runs, %.0f slot-hours\n", NR, h}'
  jobs | cut -d'|' -f3 | sort | uniq -d | sed 's/^/DUPLICATE /'
  exit 0
fi
# ---------------------------------------------------------------- slots
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
C=$P/.claims_v25_$(date +%Y%m%d%H%M%S); mkdir -p $C; jobs >$C/jobs.txt
E=$P/.est_v25; [ -f $E ] || echo "$EST_DEFAULT" >$E
T_STOP=$(( $(date +%s) + $(awk -v h="${T_HOURS:-11.5}" 'BEGIN {printf "%d", h * 3600}') )); echo $T_STOP >$C/t_stop
admit() {  # admit <class>: does a run of this class still fit before T_stop - 15 min?
  awk -v c=$1 -v now=$(date +%s) -v stop=$T_STOP -v m=${MARGIN:-900} '$1 == c {h = $2} END {exit !(now + 1.25 * h * 3600 <= stop - m)}' $E
}
port_state() {  # ok | fail | wait: the port's CIL-FUNSD seed-42 run is its test
  [ "$1" = - ] && { echo ok; return; }
  local n=cil_funsd_${1}_seed42
  if ls $P/${n}*/metrics.json >/dev/null 2>&1; then echo ok
  elif [ -f $C/$n/finished ]; then echo fail
  else echo wait; fi
}
worker() {  # worker <slot id> <gpu index>
  SLOT=$1; export CUDA_VISIBLE_DEVICES=$2
  local class gate name args waiting st free pid
  exec 8>/tmp/v25_gpu$2.lock
  while :; do
    waiting=0
    while IFS='|' read -r -u 3 class gate name args; do  # fd 3: children (uv) never eat the list
      [ -d $C/$name ] && continue
      admit $class || continue
      st=$(port_state $gate)
      case $st in
        wait) waiting=1; continue ;;
        fail) mkdir $C/$name 2>/dev/null && { touch $C/$name/finished; echo "[$SLOT] $(date '+%m-%d %H:%M') SKIP  $name (port test failed)" >>$Q; }; continue ;;
      esac
      # one start at a time per card, and only with room: 4 slots share 12 GB
      flock 8
      free=$(nvidia-smi -i $2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
      if [ "${free:-0}" -lt ${MEMFREE:-3600} ]; then flock -u 8; waiting=1; sleep ${MEMWAIT:-60}; continue 2; fi
      mkdir $C/$name 2>/dev/null || { flock -u 8; continue; }  # atomic claim: another slot got it first
      run $name $args & pid=$!
      sleep ${HOLD:-150}; flock -u 8; wait $pid   # let the run reach its memory before the next start on this card
      touch $C/$name/finished
      continue 2  # rescan from the top after every job
    done 3<$C/jobs.txt
    [ $waiting = 0 ] && break          # nothing admissible left
    [ $(date +%s) -ge $T_STOP ] && break
    sleep ${SCAN:-30}
  done
  echo "[$SLOT] $(date '+%m-%d %H:%M') slot exit: nothing admissible left" >>$Q
}
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
echo "$(date '+%m-%d %H:%M') v25 remote driver running (pid $$): GPUS=[$GPUS] JOBS_PER_GPU=$JOBS_PER_GPU, $(wc -l <$C/jobs.txt) jobs, stop $(date -d @$T_STOP '+%m-%d %H:%M'), claims in $C" >>$Q
for k in $(seq 0 $((JOBS_PER_GPU - 1))); do
  for g in $GPUS; do worker g${g}s$k $g & sleep 5; done
done
wait
echo "$(date '+%m-%d %H:%M') v25 ALL SLOTS EXITED" >>$Q
