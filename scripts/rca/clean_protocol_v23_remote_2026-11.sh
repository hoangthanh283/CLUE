#!/usr/bin/env bash
# v23 (2026-11-16): fills the tab:main cells v21 left empty. Same clean recipe (T) and slot loop as
# scripts/rca/clean_protocol_v21_remote_2026-11.sh; only the job list differs:
#   Tier 1: one-epoch smokes of CODA-Prompt / CPFD / IS3 / LCA on CIL-FUNSD (each gates its method),
#           the CIL-WildReceipt runs cut off on 2026-11-15 (ER seed 7, DER++-250 x3), and the CIL-FUNSD
#           column: LwF, CL-LoRA, ER / DER++ with 20 documents, SBR-Recall, the four ports, 3 seeds.
#   Tier 2: the four ports on DIL-XFUND and CIL-WildReceipt, 3 seeds.
# Clean-protocol queue v21 (2026-11-15, Dev): slot-based driver for the RENTED multi-GPU box (plan
# rippling-pondering-comet, five document datasets). One worker per slot (GPUS x JOBS_PER_GPU), each with
# CUDA_VISIBLE_DEVICES set; workers pull jobs from one ordered list and claim each with an atomic mkdir, so
# no job runs twice. Resume-safe: run() skips any run whose metrics.json exists (push copies the local
# metrics.json files, so runs finished on the local box are skipped here too).
# Recipes, run names and results/clean_2026-11/ layout are v20's (LayoutLMv3 clean recipe) and v19's (BROS
# b1 recipe), unchanged: batch sizes are part of the protocol and are NOT raised for the bigger cards.
# queue.log lines are v20's format prefixed with the slot id, e.g. "[g0s1] 11-16 01:02 START <name>".
#  Tier 1: smokes (cil_wildreceipt, dil_xlingual; BROS b1 smokes, skipped when their metrics exist); seed 42
#          of the five v20 cells on both streams (released only if the stream's smokes pass smoke_ok);
#          BROS const_o x3 and plain CoLaR x3 (v19, gated on the BROS smokes); rank-64 seeds 7/123 and
#          Mixed joint seeds 7/123 (v13's args); seeds 7 and 123 of the v20 cells.
#  Tier 2 (eligible only once every Tier-1 job is claimed): the tab:main baselines that have clean-protocol
#          args (LwF, CL-LoRA, DER++-250; SBR-Recall on the class-incremental stream) on both new streams,
#          3 seeds, seed 42 first. CODA-Prompt, CPFD, IS3, LCA have no clean-protocol args: not queued.
#   GPUS="0 1 2 3" JOBS_PER_GPU=2 setsid nohup bash scripts/rca/clean_protocol_v21_remote_2026-11.sh &
#   DRY_RUN=1 bash scripts/rca/clean_protocol_v21_remote_2026-11.sh      # print the ordered job list
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
LR64="method.epochs=10 method.split_layer_k=4 method.rank_r=64 method.replay_batch_size=4"  # v13
L="model=layoutlmv3_base"
# v19 BROS b1 recipe: v19's T and LR/LR1 (batch 1, replay batch 2, fp32)
TB="training.batch_size=1 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LRB="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
LRB1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=2"
B="scenario=cil_cord model=bros_base"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"  # v11
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STREAMS="cil_wildreceipt dil_xlingual"
SLOT="${SLOT:-main}"
run() {  # run <name> <args...>   (v20's, with the slot id on each queue line)
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "[$SLOT] $(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "[$SLOT] $(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "[$SLOT] $(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
# a smoke passes when it wrote metrics.json and stayed under 5000 MB of VRAM (v20's)
smoke_ok() { local n; for n in "$@"; do uv run python -c "import json,sys,glob; f=glob.glob(sys.argv[1]+'*/metrics.json'); sys.exit(0 if f and json.load(open(f[0])).get('peak_gpu_mem_mb', 1e9) < 5000 else 1)" "$P/$n" 9>&- || return 1; done; }
smokes_of() { echo "cil_funsd_smoke_${1}_e1_seed42"; }  # one smoke per ported baseline, on the smallest stream
# ---------------------------------------------------------------- the ordered job list
# one line per job: tier|gate|recipe|name|args   (gate "-" = none, else the method whose smoke gates it)
job() { echo "$1|$2|$3|$4|$5"; }
PORTS="coda_prompt cpfd is3 lca"   # config defaults (LCA: its own SGD recipe), 10 epochs, as on CIL-CORD
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
jobs() {
  local SC s m
  # Tier 1: one-epoch smokes of the four ports under the clean protocol
  for m in $PORTS; do job 1 - std cil_funsd_smoke_${m}_e1_seed42 "method=$m +method.run_alias=smoke_${m}_e1 method.epochs=1 scenario=cil_funsd seed=42 $L"; done
  # Tier 1: what the v21 night left unfinished on CIL-WildReceipt (same names and args)
  job 1 - std cil_wildreceipt_er_seed7 "method=er method.epochs=10 scenario=cil_wildreceipt seed=7 $L"
  for s in 42 7 123; do job 1 - std cil_wildreceipt_der_pp250_seed$s "method=der_pp method.buffer_size=250 scenario=cil_wildreceipt +method.run_alias=der_pp250 method.epochs=10 seed=$s $L"; done
  # Tier 1: CIL-FUNSD baselines. ER / DER++ keep 20 documents, the memory of the CIL-FUNSD CoLaR
  # cells (10 per session, 2 sessions): the default 200 would hold all 149 training forms.
  for s in 42 7 123; do
    job 1 - std cil_funsd_lwf_seed$s "method=lwf method.epochs=10 scenario=cil_funsd seed=$s $L"
    job 1 - std cil_funsd_cl_lora_seed$s "method=cl_lora method.epochs=10 scenario=cil_funsd seed=$s $L"
    job 1 - std cil_funsd_er_b20_seed$s "method=er method.buffer_size=20 +method.run_alias=er_b20 method.epochs=10 scenario=cil_funsd seed=$s $L"
    job 1 - std cil_funsd_der_pp_b20_seed$s "method=der_pp method.buffer_size=20 +method.run_alias=der_pp_b20 method.epochs=10 scenario=cil_funsd seed=$s $L"
    job 1 - std cil_funsd_colar_sbr_recall_r64_seed$s "$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR scenario=cil_funsd seed=$s $L"
    for m in $PORTS; do job 1 $m std cil_funsd_${m}_seed$s "method=$m method.epochs=10 scenario=cil_funsd seed=$s $L"; done
  done
  # Tier 2: the four ports on DIL-XFUND, then CIL-WildReceipt (the long one last)
  for SC in dil_xlingual cil_wildreceipt; do for s in 42 7 123; do for m in $PORTS; do
    job 2 $m std ${SC}_${m}_seed$s "method=$m method.epochs=10 scenario=$SC seed=$s $L"
  done; done; done
}
if [ "${DRY_RUN:-0}" = 1 ]; then
  jobs | awk -F'|' '{printf "%3d  tier %s  gate %-16s %-5s %s\n", NR, $1, $2, $3, $4; n[$1]++}
                   END {for (t in n) printf "tier %s: %d jobs\n", t, n[t]}'
  exit 0
fi
# ---------------------------------------------------------------- slots
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
C=$P/.claims_v23_$(date +%Y%m%d%H%M%S)   # one claim dir per launch; a relaunch re-scans (run() skips done)
mkdir -p $C; jobs >$C/jobs.txt
gate_state() {  # ok | fail | wait (cached once decided)
  local G=$1 f=$C/gate_$1 n all=1
  [ "$G" = - ] && { echo ok; return; }
  [ -f $f ] && { cat $f; return; }
  if smoke_ok $(smokes_of $G); then echo ok >$f; echo ok; return; fi
  for n in $(smokes_of $G); do [ -f $C/$n/finished ] || all=0; done
  if [ $all = 1 ]; then echo fail >$f; echo "[$SLOT] $(date '+%m-%d %H:%M') SKIP  $G jobs (a smoke wrote no metrics or peaked >= 5000 MB)" >>$Q; echo fail
  else echo wait; fi
}
worker() {  # worker <slot id> <gpu index>
  SLOT=$1; export CUDA_VISIBLE_DEVICES=$2
  local tier gate recipe name args left t1left st
  while :; do
    left=0; t1left=0; local -A GS=()  # gate states, decided once per scan (smoke_ok is slow)
    while IFS='|' read -r -u 3 tier gate recipe name args; do  # fd 3: children (uv) never eat the list
      [ -d $C/$name ] && continue
      left=1; [ $tier = 1 ] && t1left=1
      [ $tier = 2 ] && [ $t1left = 1 ] && continue  # Tier 2 waits until Tier 1 is fully claimed
      st=${GS[$gate]:-}; [ -n "$st" ] || { st=$(gate_state $gate); GS[$gate]=$st; }
      case $st in
        wait) continue ;;
        fail) mkdir $C/$name 2>/dev/null && touch $C/$name/finished; continue ;;
      esac
      mkdir $C/$name 2>/dev/null || continue  # atomic claim: another slot got it first
      if [ $recipe = bros ]; then T="$TB" run $name $args; else run $name $args; fi
      touch $C/$name/finished
      continue 2  # rescan from the top after every job
    done 3<$C/jobs.txt
    [ $left = 0 ] && break
    sleep 30  # everything left is gated on a smoke still running
  done
  echo "[$SLOT] $(date '+%m-%d %H:%M') slot idle: no job left" >>$Q
}
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
echo "$(date '+%m-%d %H:%M') v23 remote driver running (pid $$): GPUS=[$GPUS] JOBS_PER_GPU=$JOBS_PER_GPU, $(wc -l <$C/jobs.txt) jobs, claims in $C" >>$Q
for g in $GPUS; do
  for k in $(seq 0 $((JOBS_PER_GPU - 1))); do worker g${g}s$k $g & sleep 20; done  # stagger dataset builds
done
wait
echo "$(date '+%m-%d %H:%M') v23 ALL DONE" >>$Q
