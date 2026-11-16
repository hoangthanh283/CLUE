#!/usr/bin/env bash
# External admission control for the running v25 queue (2026-11-16). v25 starts a run when the
# card shows >= 3.6 GB free, but runs grow after they start (CPFD, DER++, ER reach 4.4-5 GB), so
# four of them on one 12 GB card run out of memory. This guard does not touch the driver script:
# every 10 s it sums the known peak of the runs on each card and, when the next queued run would
# not fit at peak, SIGSTOPs that card's slot shells (their training children keep running);
# it SIGCONTs them as soon as there is room. On exit it resumes every slot.
#   setsid nohup bash scripts/rca/v25_memguard_2026-11.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
P=results/clean_2026-11; LOG=$P/memguard.log; CAP=${CAP:-12282}; MARGIN=${MARGIN:-400}
peak() {  # nvidia-smi view of a run at its peak, MiB (measured on this box, rounded up)
  case $1 in der_pp) echo 4700 ;; cpfd) echo 5000 ;; er) echo 4400 ;; lwf) echo 3400 ;; is3) echo 3100 ;;
    naive|joint|colar|colar_sbr|colar_sbr_recall) echo 3000 ;; lca) echo 2600 ;; cl_lora) echo 1900 ;;
    coda_prompt) echo 1300 ;; *) echo 3500 ;; esac
}
slots() { pgrep -f clean_protocol_v25_remote; }
trap 'for p in $(slots); do kill -CONT $p 2>/dev/null; done; echo "$(date "+%m-%d %H:%M:%S") guard exit: all slots resumed" >>$LOG' EXIT
declare -A STATE=()
while slots >/dev/null; do
  C=$(ls -d $P/.claims_v25_* | tail -1)
  # the run a free slot would take next: first unclaimed line of the job list
  nm=$(while IFS='|' read -r class gate name args; do [ -d $C/$name ] || { echo "$args" | grep -o 'method=[a-z_0-9]*' | head -1 | cut -d= -f2; break; }; done <$C/jobs.txt)
  need=$(( $(peak "${nm:-x}") + MARGIN ))
  declare -A PROJ=(); for g in 0 1 2 3 4 5; do PROJ[$g]=0; done
  for p in $(pgrep -f "bin/python scripts/train.py"); do
    a=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null) || continue
    case "$a" in *"uv run"*|"") continue ;; esac
    g=$(tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$g" ] || continue
    m=$(echo "$a" | grep -o 'method=[a-z_0-9]*' | head -1 | cut -d= -f2)
    PROJ[$g]=$(( ${PROJ[$g]} + $(peak "$m") ))
  done
  for g in 0 1 2 3 4 5; do
    if [ $(( ${PROJ[$g]} + need )) -gt $CAP ]; then want=STOP; else want=CONT; fi
    for w in $(slots); do
      [ "$(readlink /proc/$w/fd/8 2>/dev/null)" = "/tmp/v25_gpu$g.lock" ] && kill -$want $w 2>/dev/null
    done
    if [ "${STATE[$g]:-}" != $want ]; then
      echo "$(date '+%m-%d %H:%M:%S') gpu$g $want (peak of running ${PROJ[$g]} MiB, next ${nm:-?} needs $need)" >>$LOG
      STATE[$g]=$want
    fi
  done
  sleep 10
done
