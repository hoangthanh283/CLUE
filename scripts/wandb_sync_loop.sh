#!/usr/bin/env bash
# Upload finished offline W&B runs (the queues run with wandb.mode=offline) to
# thanh-workspace/doccl-aaai2027, now and every 30 min. A run counts as finished when its
# .wandb file has not been written for 20 min; `wandb sync` drops a .synced marker so each
# run uploads once (crashed runs with no exit record get no marker from wandb, so the loop
# touches it after a clean sync). A still-running run is picked up on a later pass.
#   setsid nohup bash scripts/wandb_sync_loop.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG=results/wandb_sync.log
exec 9>wandb/.sync.lock; flock -n 9 || exit 0
while :; do
  ok=0; bad=0
  for d in wandb/offline-run-*; do
    f=$(ls $d/*.wandb 2>/dev/null | head -1); [ -n "$f" ] || continue
    [ -e "$f.synced" ] && continue
    [ -n "$(find "$f" -mmin -20)" ] && continue            # still being written
    if timeout 900 uv run wandb sync "$d" >>$LOG.detail 2>&1 9>&-; then touch "$f.synced"; ok=$((ok+1)); else bad=$((bad+1)); echo "$(date '+%m-%d %H:%M') FAIL $d" >>$LOG; fi
  done
  echo "$(date '+%m-%d %H:%M') pass: synced $ok, failed $bad" >>$LOG
  sleep 1800
done
