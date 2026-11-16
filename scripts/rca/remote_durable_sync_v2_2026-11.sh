#!/usr/bin/env bash
# v2 (2026-11-16) of remote_durable_sync_2026-11.sh. Runs ON the rented box. Every 10 min: upload finished
# offline W&B runs and copy results + wandb dirs to the S3 bucket, so a dead instance loses at most
# the runs in flight. Reads WANDB_* and R2_* from ./.env (the user copies it; never printed).
# Each half is a no-op when its credentials or tool are missing.
#   setsid nohup bash scripts/rca/remote_durable_sync_2026-11.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
[ -f .env ] || { echo "no .env: nothing to sync with" >&2; exit 1; }
set -a; . ./.env; set +a
P=results/clean_2026-11; LOG=results/durable_sync.log; mkdir -p wandb
exec 9>wandb/.durable.lock; flock -n 9 || exit 0
DEST="obj:${R2_BUCKET:-doccl-results}/vast_2026-11"
if [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_ENDPOINT:-}" ]; then
  # same in-memory rclone remote as maybe_build_r2_remote in scripts/run_grid_multigpu.sh
  export RCLONE_CONFIG_OBJ_TYPE=s3 RCLONE_CONFIG_OBJ_PROVIDER=Cloudflare RCLONE_CONFIG_OBJ_REGION=auto
  export RCLONE_CONFIG_OBJ_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID RCLONE_CONFIG_OBJ_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY RCLONE_CONFIG_OBJ_ENDPOINT=$R2_ENDPOINT
  S3=1
else S3=0; fi
while :; do
  ok=0; bad=0
  if [ -n "${WANDB_API_KEY:-}" ]; then
    for d in wandb/offline-run-*; do
      f=$(ls $d/*.wandb 2>/dev/null | head -1); [ -n "$f" ] || continue
      [ -e "$f.synced" ] && continue
      [ -n "$(find "$f" -mmin -20)" ] && continue            # still being written
      if timeout 900 .venv/bin/wandb sync "$d" >>$LOG.detail 2>&1 9>&-; then touch "$f.synced"; ok=$((ok+1)); else bad=$((bad+1)); fi
    done
  fi
  s3=skip
  if [ $S3 = 1 ] && command -v rclone >/dev/null 2>&1; then
    # ponytail: results are small (no checkpoints are written); copy everything, rclone skips unchanged files
    # v2: LCA stashes one 500 MB backbone per session under <run>/lca_backbones/ while it runs and deletes
    # them at the end: never upload those. Logs of running jobs change during the copy, so the
    # post-transfer hash check is off (a later pass re-copies the grown file).
    if rclone copy $P "$DEST/$P" --transfers 8 --exclude '**/lca_backbones/**' --ignore-checksum >>$LOG.detail 2>&1 9>&- && rclone copy wandb "$DEST/wandb" --transfers 8 --exclude '.*.lock' --ignore-checksum >>$LOG.detail 2>&1 9>&-; then s3=ok; else s3=FAIL; fi
  fi
  echo "$(date '+%m-%d %H:%M') wandb synced $ok failed $bad; s3 $s3" >>$LOG
  sleep 600
done
