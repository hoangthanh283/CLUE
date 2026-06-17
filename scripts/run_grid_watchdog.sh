#!/usr/bin/env bash
# Resource guardian for the autonomous grid. Samples every 20s and:
#   - HARD-ABORT if RAM > RAM_LIMIT_MB or VRAM > VRAM_LIMIT_MB (kills driver + training).
#   - Detects stalls (no .done progress AND GPU idle for STALL_SAMPLES) and restarts the
#     driver (resume-safe via .done markers) up to MAX_RESTARTS times.
#   - Logs a heartbeat with RAM/VRAM/GPU/CPU + .done progress to results/logs/watchdog.log.
#
# This process is the safety net the operator emphasised: it must keep the box alive.

set -uo pipefail
cd "$(dirname "$0")/.."

RAM_LIMIT_MB=${RAM_LIMIT_MB:-14000}     # user's hard ceiling: total system RAM must stay <= 14 GB.
                                        # The per-run cgroup MEM_CAP (set in the driver) is kept well
                                        # below this so a single run is OOM-killed before the SYSTEM
                                        # hits 14 GB — the machine itself never OOMs.
RAM_SAFE_MB=${RAM_SAFE_MB:-6000}        # only (re)start the driver once RAM has drained below this
VRAM_LIMIT_MB=${VRAM_LIMIT_MB:-5800}    # 6 GB RTX 2060; raised from 5000 so DocCL (Fisher + GPU
                                        # teacher + replay buffer stacked, peaks ~5.5 GB) fits.
STALL_SAMPLES=${STALL_SAMPLES:-30}     # ~10 min of no progress + idle GPU => stall
MAX_RESTARTS=${MAX_RESTARTS:-50}
DRIVER="scripts/run_autonomous_grid.sh"

WLOG=results/logs/watchdog.log
mkdir -p results/logs
wsay() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$WLOG"; }

restarts=0
stall=0
prev_done=-1

driver_alive() { pgrep -f "$DRIVER" >/dev/null 2>&1; }
kill_all() {
  pkill -9 -f "$DRIVER" 2>/dev/null || true
  pkill -9 -f 'scripts/run_grid.sh' 2>/dev/null || true
  pkill -9 -f 'scripts/train.py' 2>/dev/null || true
  # tear down any leftover systemd run scopes from capped train.py launches
  systemctl --user stop 'run-*.scope' 2>/dev/null || true
}
# Block until system RAM drains below RAM_SAFE_MB (memory + swap reclaim after a kill),
# so we never restart the driver on top of un-reclaimed memory — the cause of the
# earlier restart->rebuild->OOM->restart thrash loop. Times out after ~3 min.
wait_for_ram() {
  for _ in $(seq 1 36); do
    local u; u=$(free -m | awk '/Mem:/{print $3}')
    [ "${u:-99999}" -lt "$RAM_SAFE_MB" ] && return 0
    sleep 5
  done
  wsay "WARN: RAM did not drain below ${RAM_SAFE_MB}MB after 3 min (now $(free -m|awk '/Mem:/{print $3}')MB)"
  return 0
}
start_driver() {
  wait_for_ram
  nohup bash "$DRIVER" >> results/logs/autonomous.log 2>&1 &
  wsay "driver started (pid $!) at RAM=$(free -m|awk '/Mem:/{print $3}')MB"
}

wsay "=== WATCHDOG START (RAM<${RAM_LIMIT_MB}MB VRAM<${VRAM_LIMIT_MB}MB) ==="
driver_alive || start_driver

while true; do
  sleep 20
  used=$(free -m | awk '/Mem:/{print $3}')
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  gutil=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  load=$(cut -d' ' -f1 /proc/loadavg)
  done=$(ls results/*/.done 2>/dev/null | wc -l)

  # HARD safety abort
  if [ "${used:-0}" -gt "$RAM_LIMIT_MB" ]; then
    wsay "!!! RAM ${used}MB > ${RAM_LIMIT_MB}MB — HARD ABORT"; kill_all
    wsay "killed all; sleeping 30s to let RAM settle, then restart (resume-safe)"; sleep 30
    if [ "$restarts" -lt "$MAX_RESTARTS" ]; then restarts=$((restarts+1)); start_driver; stall=0; else wsay "max restarts hit; exiting"; exit 1; fi
    continue
  fi
  if [ "${vram:-0}" -gt "$VRAM_LIMIT_MB" ]; then
    wsay "!!! VRAM ${vram}MB > ${VRAM_LIMIT_MB}MB — HARD ABORT"; kill_all; sleep 20
    if [ "$restarts" -lt "$MAX_RESTARTS" ]; then restarts=$((restarts+1)); start_driver; stall=0; else wsay "max restarts hit; exiting"; exit 1; fi
    continue
  fi

  # progress / stall tracking
  if [ "$done" != "$prev_done" ]; then
    wsay "done=${done} RAM=${used}MB VRAM=${vram}MB GPU=${gutil}% load=${load}"
    prev_done=$done; stall=0
  else
    # only count as stall if GPU is idle AND no driver progress
    if [ "${gutil:-0}" -lt 5 ]; then stall=$((stall+1)); else stall=0; fi
  fi

  # driver finished?
  if ! driver_alive; then
    if grep -qa 'AUTONOMOUS GRID COMPLETE' results/logs/autonomous.log 2>/dev/null; then
      wsay "=== driver reported COMPLETE (done=${done}) — watchdog exiting ==="; exit 0
    fi
    # driver died without completing -> restart (resume-safe)
    if [ "$restarts" -lt "$MAX_RESTARTS" ]; then
      restarts=$((restarts+1)); wsay "driver not running and not complete — restart #${restarts}"; start_driver; stall=0
    else
      wsay "driver gone, max restarts hit; exiting"; exit 1
    fi
    continue
  fi

  # stall recovery
  if [ "$stall" -ge "$STALL_SAMPLES" ]; then
    wsay "STALL: GPU idle + no progress for ${STALL_SAMPLES} samples — restarting driver"
    kill_all; sleep 15
    if [ "$restarts" -lt "$MAX_RESTARTS" ]; then restarts=$((restarts+1)); start_driver; stall=0; else wsay "max restarts; exiting"; exit 1; fi
  fi

  # periodic heartbeat even without progress (every ~5 min)
  if [ $((SECONDS % 300)) -lt 20 ]; then
    wsay "heartbeat done=${done} RAM=${used}MB VRAM=${vram}MB GPU=${gutil}% load=${load} stall=${stall}"
  fi
done
