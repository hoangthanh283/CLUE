#!/usr/bin/env bash
# Copy the experiment to / from the rented GPU box (plan rippling-pondering-comet, 2026-11-15). Key-based
# ssh only (BatchMode=yes: never prompts, never stores a password or key). Run from anywhere.
#   bash scripts/rca/remote_sync_2026-11.sh push user@host [port]   # repo + data/ + HF caches + metrics
#   bash scripts/rca/remote_sync_2026-11.sh pull user@host [port]   # finished runs, logs, queue, wandb back
# REMOTE_DIR (default CLUE) is the repo path on the remote, relative to its home directory.
#  push: the repo without results/, wandb/, .venv/ (data/ included), plus the HF dataset cache
#        (~/.cache/huggingface/datasets), the hub datasets and the two backbones the queue uses
#        (LayoutLMv3-base, BROS-base), plus every results/clean_2026-11/*/metrics.json so the remote
#        driver's run() skips runs that already finished here (metrics only; no checkpoint, no log).
#  pull: only remote run dirs that have a metrics.json and do not exist here (never overwrites a local
#        file), the remote run logs (newer copy wins), the remote queue.log as queue_remote.log (the
#        local queue.log is never touched) and the offline wandb run dirs (the local sync loop uploads them).
set -euo pipefail
MODE=${1:?push|pull}; TARGET=${2:?user@host}; PORT=${3:-22}
cd "$(dirname "$0")/../.."
R=${REMOTE_DIR:-CLUE}; P=results/clean_2026-11
SSH="ssh -p $PORT -o BatchMode=yes"  # host key must already be in known_hosts (nothing is written here)
RS=(rsync -az --partial -e "$SSH")
HF=$HOME/.cache/huggingface
case $MODE in
  push)
    $SSH "$TARGET" "mkdir -p $R/$P .cache/huggingface/datasets .cache/huggingface/hub"
    # never .env (credentials stay local; the remote logs W&B offline); archives / caches are not needed
    "${RS[@]}" --exclude=/results/ --exclude=/wandb/ --exclude=/.venv/ --exclude='__pycache__/' --exclude=/.env \
      --exclude='/results_*/' --exclude=/outputs/ --exclude=/htmlcov/ --exclude=/tmp --exclude=/.coverage \
      --exclude=/.pytest_cache/ --exclude=/.ruff_cache/ --exclude=/memory_storage/ ./ "$TARGET:$R/"
    "${RS[@]}" "$HF/datasets/" "$TARGET:.cache/huggingface/datasets/"
    "${RS[@]}" --include='/datasets--*/***' --include='/models--microsoft--layoutlmv3-base/***' \
      --include='/models--jinho8345--bros-base-uncased/***' --exclude='*' "$HF/hub/" "$TARGET:.cache/huggingface/hub/"
    "${RS[@]}" --include='*/' --include='/*/metrics.json' --exclude='*' --prune-empty-dirs "$P/" "$TARGET:$R/$P/"
    echo "push done: $TARGET:$R (then on the remote: uv sync; DRY_RUN=1 bash scripts/rca/clean_protocol_v21_remote_2026-11.sh)"
    ;;
  pull)
    LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
    $SSH "$TARGET" "cd $R/$P && ls -d */metrics.json 2>/dev/null | sed 's#/metrics.json##'" | while read -r d; do
      [ -e "$P/$d" ] || echo "$d"; done >"$LIST"
    if [ -s "$LIST" ]; then "${RS[@]}" -r --ignore-existing --files-from="$LIST" "$TARGET:$R/$P/" "$P/"; fi
    "${RS[@]}" --update --exclude='/queue.log' --include='/*.log' --exclude='*' "$TARGET:$R/$P/" "$P/"
    "${RS[@]}" "$TARGET:$R/$P/queue.log" "$P/queue_remote.log"
    mkdir -p wandb
    "${RS[@]}" --ignore-existing --include='/offline-run-*/***' --exclude='*' "$TARGET:$R/wandb/" wandb/
    echo "pull done: $(wc -l <"$LIST") new run dirs from $TARGET"
    ;;
  *) echo "usage: $0 push|pull user@host [port]" >&2; exit 2 ;;
esac
