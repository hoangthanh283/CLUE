#!/usr/bin/env bash
# Prime the R2 durable-resume bucket from THIS box's local results/ — run ONCE before
# launching the remote grids so they skip everything already finished.
#
# WHY: the multi-GPU grid resumes by pulling */.done markers from R2 (sync_pull) BEFORE
# building its job list. But already-completed runs only exist in R2 if they were pushed
# there. Runs produced locally (e.g. the 25 done on this box) are NOT in R2 unless seeded,
# so a fresh remote box would re-run them and waste compute. This uploads the local resume
# markers (the SAME tiny set the grid syncs) so every machine's sync_pull sees them.
#
# It can also PULL R2 -> local (--pull) to refresh this box with what the remotes finished,
# and LIST what R2 currently has (--list) so you can confirm coverage before launching.
#
# Just run it — it PROMPTS for the R2 creds (held in memory, never written to disk). Any
# cred already in the environment is reused (no prompt), so it also works non-interactively.
#   bash scripts/sync_results_to_r2.sh            # push local .done/metrics -> R2 (default)
#   bash scripts/sync_results_to_r2.sh --pull     # pull R2 -> local (refresh after remotes run)
#   bash scripts/sync_results_to_r2.sh --list     # list run-names present in R2
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-push}"   # push | --push | pull | --pull | list | --list
MODE="${MODE#--}"

# ── Credentials (prompt if not already in env; memory only, never written to disk) ──
prompt_secret(){ local var="$1" label="$2"; [ -n "${!1:-}" ] && { echo "  $label (from env)"; return; }; read -r -s -p "  ${label}: " v; echo; printf -v "$var" '%s' "$v"; }
prompt_plain(){  local var="$1" label="$2" def="${3:-}"; [ -n "${!1:-}" ] && { echo "  $label = ${!1} (env)"; return; }; read -r -p "  ${label}${def:+ [$def]}: " v; printf -v "$var" '%s' "${v:-$def}"; }

command -v rclone >/dev/null 2>&1 || { echo "ERROR: rclone not installed."; exit 1; }

echo "Cloudflare R2 credentials (Enter reuses any env value):"
prompt_secret R2_ACCESS_KEY_ID     "R2 Access Key ID"
prompt_secret R2_SECRET_ACCESS_KEY "R2 Secret Access Key"
prompt_plain  R2_ENDPOINT          "R2 Endpoint URL"
prompt_plain  R2_BUCKET            "R2 Bucket" "doccl-results"
[ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_ENDPOINT:-}" ] \
  || { echo "ERROR: R2 Access Key ID, Secret, and Endpoint are all required."; exit 1; }

SYNC_REMOTE="obj:${R2_BUCKET}/results"
# Only the tiny resume markers travel (mirrors run_grid_multigpu.sh SYNC_INCLUDES).
INCLUDES=(--include "*/.done" --include "*/metrics.json" --include "*/matrix.npy"
          --include "table_single_task_baselines.csv")

# In-process rclone 'obj' remote (no config file on disk) — identical to the grid's.
export RCLONE_CONFIG_OBJ_TYPE=s3
export RCLONE_CONFIG_OBJ_PROVIDER=Cloudflare
export RCLONE_CONFIG_OBJ_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export RCLONE_CONFIG_OBJ_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export RCLONE_CONFIG_OBJ_ENDPOINT="${R2_ENDPOINT}"
export RCLONE_CONFIG_OBJ_REGION=auto

case "$MODE" in
  list)
    echo "==> Completed runs already in R2 (${SYNC_REMOTE}) — SKIPPED by every box:"
    # List .done files recursively; their parent dir is the run-name.
    done_list=$(rclone lsf "$SYNC_REMOTE" -R --include "*/.done" 2>/dev/null | sed 's#/\.done$##' | sort)
    [ -n "$done_list" ] && echo "$done_list"
    n=$(printf '%s\n' "$done_list" | grep -c . || true)
    echo "==> ${n} completed (.done) run(s) in R2."
    ;;
  pull)
    echo "==> Pulling R2 -> local results/ (refresh this box with what the remotes finished) ..."
    rclone copy "$SYNC_REMOTE" results/ "${INCLUDES[@]}" -P
    # Partial-write safety: drop metrics/matrix in any dir lacking .done.
    for d in results/*/; do [ -f "${d}.done" ] || rm -f "${d}metrics.json" "${d}matrix.npy" 2>/dev/null; done
    echo "==> Local now has $(ls results/*/.done 2>/dev/null | wc -l) completed run(s)."
    ;;
  push)
    LOCAL_DONE=$(ls results/*/.done 2>/dev/null | wc -l)
    echo "==> Priming R2 (${SYNC_REMOTE}) from ${LOCAL_DONE} local completed run(s) ..."
    [ "$LOCAL_DONE" -gt 0 ] || { echo "    (no local .done markers — nothing to seed)"; exit 0; }
    rclone copy results/ "$SYNC_REMOTE" "${INCLUDES[@]}" -P
    echo "==> Done. Remote boxes' sync_pull will now skip these ${LOCAL_DONE} run(s)."
    echo "    Verify: bash scripts/sync_results_to_r2.sh --list"
    ;;
  *)
    echo "ERROR: unknown mode '$MODE' (use push | pull | list)"; exit 2 ;;
esac
