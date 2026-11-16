#!/usr/bin/env bash
# v26 (2026-11-16, plan ok-plan-the-improvement-abstract-mist): the rented box's queue for the last
# window before the CVPR deadline. v25's recipe, run() and claim directories, with three changes:
#   (1) strict priority: a slot starts the FIRST unclaimed job that still fits before T_stop, or
#       waits for room; it never skips ahead to a smaller job, so the long jobs are not starved;
#   (2) peak-aware start: the known peak memory of every train.py already on the card (v25's or
#       v26's, read from /proc) plus the candidate's must stay under CAP. v25's free-memory gate
#       let runs start that later grew and died of CUDA OOM (09:14 / 09:19 ICT, Nov 16);
#   (3) the list is $C/jobs.txt, re-read on every scan: append "class|name|args" lines to add
#       work. Slots stay alive on an empty list while $C/open exists.
# Estimates per class (hours on this box under sharing) live in $P/.est_v26, re-read on every scan.
#   T_HOURS=11.3 setsid nohup bash scripts/rca/clean_protocol_v26_remote_2026-11.sh >/dev/null 2>&1 &
#   DRY_RUN=1 bash scripts/rca/clean_protocol_v26_remote_2026-11.sh      # ordered list, class, estimate
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
LRB2="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
L="model=layoutlmv3_base"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
PORTS="coda_prompt cpfd is3 lca"
P10="++method.early_stop_patience=10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOT="${SLOT:-main}"
run() {  # run <name> <args...>   (v21's)
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "[v26 $SLOT] $(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&- 8>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "[v26 $SLOT] $(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "[v26 $SLOT] $(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
# ---------------------------------------------------------------- the ordered job list
job() { echo "$1|$2|$3"; }   # class|name|args
trio() {  # trio <class> <scenario> <seed> [name infix] [extra args]: CoLaR, CoLaR-SBR, SBR-Recall
  local c=$1 SC=$2 s=$3
  job $c ${SC}_colar_seed$s "method=colar method.docs_per_task=50 $LR scenario=$SC seed=$s $L"
  job $c ${SC}_colar_sbr_seed$s "method=colar_sbr method.docs_per_task=50 $LR scenario=$SC seed=$s $L"
  job $c ${SC}_colar_sbr_recall_r64_seed$s "$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR scenario=$SC seed=$s $L"
}
jobs() {
  local s m SC
  # T0a patience probe on CIL-FUNSD (10 minutes each): are the below-naive ports an early-stopping artefact?
  for m in $PORTS; do job funsd cil_funsd_${m}_p10_seed42 "method=$m method.epochs=10 $P10 ++method.run_alias=${m}_p10 scenario=cil_funsd seed=42 $L"; done
  for s in 42 7 123; do job funsd cil_funsd_joint_p10_seed$s "method=joint method.epochs=100 $P10 ++method.run_alias=joint_p10 scenario=cil_funsd seed=$s $L"; done
  # the one run that only fits if it starts now (DER++ with 250 documents: about 8 h here)
  job cord_der_pp cil_cord_der_pp_seed42 "method=der_pp method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=42 $L"
  # T0b the same probe on CIL-CORD, and ICE-O without the 10-epoch cap
  for m in $PORTS; do job cordp cil_cord_${m}_p10_seed42 "method=$m method.epochs=10 $P10 ++method.run_alias=${m}_p10 scenario=cil_cord seed=42 $L"; done
  job cordp cil_cord_ice_o_e30_seed42 "method=ice_o method.epochs=30 ++method.early_stop_patience=5 ++method.run_alias=ice_o_e30 scenario=cil_cord seed=42 $L"
  # T1 CIL-CORD baselines under the clean protocol (v25's names: make_clean_tables.py reads them)
  for s in 42 7 123; do
    job cord_er250 cil_cord_er250_seed$s "method=er method.buffer_size=250 +method.run_alias=er250 method.epochs=10 scenario=cil_cord seed=$s $L"
    job cord_lwf cil_cord_lwf_seed$s "method=lwf method.epochs=10 scenario=cil_cord seed=$s $L"
    job cord_cl_lora cil_cord_cl_lora_seed$s "method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L"
  done
  # T3 the label-side repairs on the third class-incremental stream (the oracle rows are appended
  #    once build_cil_wildreceipt carries labels_full)
  for s in 42 7 123; do job wr cil_wildreceipt_colar_mib_seed$s "method=colar_mib method.docs_per_task=50 $LR scenario=cil_wildreceipt seed=$s $L"; done
  # T2 joint reference with patience 10
  for s in 42 7 123; do job jcord cil_cord_joint_p10_seed$s "method=joint method.epochs=100 $P10 ++method.run_alias=joint_p10 scenario=cil_cord seed=$s $L"; done
  for s in 42 7 123; do job jwr cil_wildreceipt_joint_p10_seed$s "method=joint method.epochs=100 $P10 ++method.run_alias=joint_p10 scenario=cil_wildreceipt seed=$s $L"; done
  for s in 42 7 123; do job jmixed mixed_joint_p10_seed$s "method=joint method.epochs=100 $P10 ++method.run_alias=joint_p10 scenario=mixed seed=$s $L"; done
  # T5 separated-softmax controls with the same memory: both losses sliced (SS-IL-style, no
  #    distillation), current loss sliced only (ER-ACE-style)
  for s in 42 7 123; do
    job colar cil_cord_colar_r_sliced2_seed$s "method=colar_r_sliced2 method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L"
    job colar cil_cord_colar_cur_sliced_seed$s "method=colar +method.current_sliced=true method.docs_per_task=50 +method.run_alias=colar_cur_sliced $LR scenario=cil_cord seed=$s $L"
  done
  # T6a two more class orders, seed 42
  for SC in cil_cord_o2 cil_cord_o3; do trio colar $SC 42; done
  # T1 the four ports on CIL-CORD, patience 2 as every other cell
  for s in 42 7 123; do for m in $PORTS; do job cord_$m cil_cord_${m}_seed$s "method=$m method.epochs=10 scenario=cil_cord seed=$s $L"; done; done
  # runs that died once under v25 (out of memory): one more try under the peak-aware start
  job wr cil_wildreceipt_cpfd_seed42 "method=cpfd method.epochs=10 scenario=cil_wildreceipt seed=42 $L"
  job wr cil_wildreceipt_lca_seed123 "method=lca method.epochs=10 scenario=cil_wildreceipt seed=123 $L"
  job dil_heavy dil_der_pp_seed42 "method=der_pp method.epochs=10 scenario=dil seed=42 $L"
  job dil_heavy dil_er_seed42 "method=er method.epochs=10 scenario=dil seed=42 $L"
  # T10 ten sessions of three classes
  job long cil_cord_long_naive_seed42 "method=naive method.epochs=10 scenario=cil_cord_long seed=42 $L"
  trio long cil_cord_long 42
  # T9 memory size (10 documents per session) and replay ratio (2 stored per current batch of 2)
  for s in 42 7 123; do
    job colar cil_cord_colar_d10_seed$s "method=colar method.docs_per_task=10 +method.run_alias=colar_d10 $LR scenario=cil_cord seed=$s $L"
    job colar cil_cord_colar_sbr_d10_seed$s "method=colar_sbr method.docs_per_task=10 +method.run_alias=colar_sbr_d10 $LR scenario=cil_cord seed=$s $L"
    job colar cil_cord_colar_rb2_seed$s "method=colar method.docs_per_task=50 +method.run_alias=colar_rb2 $LRB2 scenario=cil_cord seed=$s $L"
    job colar cil_cord_colar_sbr_rb2_seed$s "method=colar_sbr method.docs_per_task=50 +method.run_alias=colar_sbr_rb2 $LRB2 scenario=cil_cord seed=$s $L"
  done
  for s in 7 123; do job cord_der_pp cil_cord_der_pp_seed$s "method=der_pp method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L"; done
}
EST_DEFAULT="funsd 0.3
cordp 4.5
cord_der_pp 8.5
cord_er250 3.6
cord_lwf 2.8
cord_cl_lora 3.5
cord_cpfd 1.4
cord_lca 2.2
cord_is3 2.4
cord_coda_prompt 3.1
colar 2.5
wr 4.0
jcord 1.5
jwr 4.0
jmixed 2.5
dil_heavy 3.0
long 4.5"
if [ "${DRY_RUN:-0}" = 1 ]; then
  jobs | awk -F'|' -v e="$EST_DEFAULT" 'BEGIN {n=split(e, a, /[ \n]/); for (i=1; i<n; i+=2) est[a[i]]=a[i+1]}
       {printf "%3d  %-16s %4sh  %s\n", NR, $1, est[$1], $2; c[$1]++; h+=est[$1]}
       END {for (k in c) printf "%s: %d runs\n", k, c[k]; printf "total %d runs, %.0f slot-hours\n", NR, h}'
  jobs | cut -d'|' -f2 | sort | uniq -d | sed 's/^/DUPLICATE /'
  jobs | awk -F'|' 'NF != 3 {print "BAD LINE " NR}'
  exit 0
fi
# ---------------------------------------------------------------- slots
exec 9>$P/.driver_v26.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another v26 driver holds $P/.driver_v26.lock (pid $$)" >>$Q; exit 1; }
C=${CLAIMS:-$P/.claims_v26_$(date +%Y%m%d%H%M%S)}; mkdir -p $C; [ -f $C/jobs.txt ] || jobs >$C/jobs.txt; touch $C/open
E=$P/.est_v26; [ -f $E ] || echo "$EST_DEFAULT" >$E
T_STOP=$(( $(date +%s) + $(awk -v h="${T_HOURS:-11.3}" 'BEGIN {printf "%d", h * 3600}') )); echo $T_STOP >$C/t_stop
CAP=${CAP:-11900}   # MiB of a 12282 MiB card the known peaks may add up to
admit() {  # admit <class>: does a run of this class still fit before T_stop - 15 min? (unknown class: yes)
  awk -v c=$1 -v now=$(date +%s) -v stop=$T_STOP -v m=${MARGIN:-900} '$1 == c {h = $2} END {exit !(now + 1.25 * h * 3600 <= stop - m)}' $E
}
peak() {  # peak <method>: MiB a run of this method reaches (measured on this box by the v25 session)
  case $1 in
    der_pp) echo 4700 ;; cpfd) echo 5000 ;; er) echo 4400 ;; lwf) echo 3400 ;; is3) echo 3100 ;;
    lca) echo 2600 ;; cl_lora) echo 1900 ;; coda_prompt) echo 1300 ;; *) echo 3000 ;;
  esac
}
method_of() { echo "$1" | sed -E 's/(^|.* )method=([a-z0-9_]+).*/\2/'; }
card_load() {  # card_load <gpu>: summed known peaks of every train.py on the card, never below what is in use
  local pid s=0 used
  for pid in $(pgrep -f '/python[0-9.]* scripts/train.py'); do
    [ "$(tr '\0' '\n' </proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')" = "$1" ] || continue
    s=$(( s + $(peak "$(method_of "$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null)")") ))
  done
  used=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${used:-0}" -gt $s ] && s=$used
  echo $s
}
head_job() {  # first unclaimed, unfinished job that still fits in time: "class|name|args" or nothing
  local class name args
  while IFS='|' read -r class name args; do
    [ -d $C/$name ] && continue
    if ls $P/${name}*/metrics.json >/dev/null 2>&1; then mkdir $C/$name 2>/dev/null && touch $C/$name/finished; continue; fi
    admit $class || continue
    echo "$class|$name|$args"; return
  done <$C/jobs.txt
}
worker() {  # worker <slot id> <gpu index>
  SLOT=$1; export CUDA_VISIBLE_DEVICES=$2
  local h class name args pid
  exec 8>/tmp/v26_gpu$2.lock   # one start at a time per card (v25 is drained before launch: it starts nothing more)
  while [ $(date +%s) -lt $T_STOP ]; do
    h=$(head_job)
    if [ -z "$h" ]; then [ -e $C/open ] || break; sleep ${SCAN:-30}; continue; fi
    IFS='|' read -r class name args <<<"$h"
    flock 8
    if [ $(( $(card_load $2) + $(peak "$(method_of "$args")") )) -gt $CAP ]; then flock -u 8; sleep ${SCAN:-30}; continue; fi
    mkdir $C/$name 2>/dev/null || { flock -u 8; continue; }   # atomic claim: another slot got it first
    run $name $args & pid=$!
    sleep ${HOLD:-25}; flock -u 8; wait $pid   # the python process is visible to card_load before the next start
    touch $C/$name/finished
  done
  echo "[v26 $SLOT] $(date '+%m-%d %H:%M') slot exit" >>$Q
}
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"
echo "$(date '+%m-%d %H:%M') v26 remote driver running (pid $$): GPUS=[$GPUS] JOBS_PER_GPU=$JOBS_PER_GPU, $(wc -l <$C/jobs.txt) jobs, stop $(date -d @$T_STOP '+%m-%d %H:%M'), claims in $C. DO NOT STOP THE INSTANCE BEFORE 'v26 ALL SLOTS EXITED'." >>$Q
for k in $(seq 0 $((JOBS_PER_GPU - 1))); do
  for g in $GPUS; do worker g${g}s$k $g & sleep 2; done
done
wait
echo "$(date '+%m-%d %H:%M') v26 ALL SLOTS EXITED" >>$Q
