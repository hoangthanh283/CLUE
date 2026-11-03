#!/usr/bin/env bash
# Wait for a running train PID to exit, then (re)start the resume-safe gate chain.
set -u
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
P=results/gates/cil_protocol_2026-11
if ls $P/cil_cord_colar_sbr_noreplay_seed42*/metrics.json >/dev/null 2>&1; then echo "$(date "+%m-%d %H:%M") DONE  cil_cord_colar_sbr_noreplay_seed42" >>$P/queue.log; fi
bash scripts/rca/audit_gates_2026-11.sh
