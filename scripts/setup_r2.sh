#!/usr/bin/env bash
# Interactive helper: configure a Cloudflare R2 (or any S3-compatible) rclone remote
# for the grid's durable-resume sync, then print the exact `docker run` command.
#
# Run on your laptop (or the GPU box). It writes ~/.config/rclone/rclone.conf with an
# "obj" remote, does a tiny round-trip test against your bucket, and emits the docker
# command with SYNC_REMOTE + RCLONE_CONFIG_B64 pre-filled.
#
# Usage:  bash scripts/setup_r2.sh
#         BUCKET=doccl-results PROVIDER=Cloudflare bash scripts/setup_r2.sh   # non-interactive-ish

set -uo pipefail

BUCKET="${BUCKET:-doccl-results}"
PROVIDER="${PROVIDER:-Cloudflare}"   # Cloudflare | AWS | Backblaze | Other
REMOTE_NAME="obj"
CONF_DIR="$HOME/.config/rclone"
CONF="$CONF_DIR/rclone.conf"

echo "=== R2 / S3-compatible setup for grid durable-resume sync ==="
echo "Provider: $PROVIDER   Bucket: $BUCKET   (override via PROVIDER=/BUCKET= env)"
echo

if ! command -v rclone >/dev/null 2>&1; then
  echo "NOTE: rclone is not installed locally. It's only needed HERE for the round-trip"
  echo "test; the Docker image already bundles it. Install: curl https://rclone.org/install.sh | sudo bash"
  echo
fi

read -r -p "Access Key ID:        " AKID
read -r -s -p "Secret Access Key:    " SAK; echo
read -r -p "Endpoint URL (e.g. https://<acct>.r2.cloudflarestorage.com): " ENDPOINT

if [ -z "$AKID" ] || [ -z "$SAK" ] || [ -z "$ENDPOINT" ]; then
  echo "ERROR: all three values are required. Aborting (nothing written)."; exit 1
fi

mkdir -p "$CONF_DIR"
# Append (don't clobber an existing conf); replace a prior [obj] block if present.
if [ -f "$CONF" ] && grep -q "^\[$REMOTE_NAME\]" "$CONF"; then
  echo "An [$REMOTE_NAME] remote already exists in $CONF — backing up to $CONF.bak and replacing it."
  cp -f "$CONF" "$CONF.bak"
  # strip the existing [obj] block (from its header to the next [section] or EOF)
  awk -v sec="[$REMOTE_NAME]" '
    $0==sec {skip=1; next}
    skip && /^\[/ {skip=0}
    !skip {print}
  ' "$CONF.bak" > "$CONF"
fi
cat >> "$CONF" <<EOF

[$REMOTE_NAME]
type = s3
provider = $PROVIDER
access_key_id = $AKID
secret_access_key = $SAK
endpoint = $ENDPOINT
region = auto
EOF
chmod 600 "$CONF"
echo "Wrote remote [$REMOTE_NAME] to $CONF"

SYNC_REMOTE="${REMOTE_NAME}:${BUCKET}/results"

# Round-trip test (only if rclone present locally).
if command -v rclone >/dev/null 2>&1; then
  echo; echo "=== Round-trip test against ${SYNC_REMOTE} ==="
  TMP=$(mktemp -d); echo "ok $(date +%s)" > "$TMP/.synccheck"
  if rclone copy "$TMP" "$SYNC_REMOTE" --include ".synccheck" 2>/dev/null \
     && rclone lsf "$SYNC_REMOTE" 2>/dev/null | grep -q ".synccheck"; then
    echo "PASS: wrote + listed .synccheck on the remote. Durable sync is ready."
    rclone delete "$SYNC_REMOTE/.synccheck" 2>/dev/null || true
  else
    echo "FAIL: could not write/list on ${SYNC_REMOTE}."
    echo "  Check: bucket '$BUCKET' exists, token has Object Read&Write, endpoint is correct."
    rm -rf "$TMP"; exit 1
  fi
  rm -rf "$TMP"
fi

echo
echo "=== DONE. Use this in your docker run on the GPU box: ==="
echo
cat <<EOF
docker run --rm --gpus all \\
  -e WANDB_API_KEY=YOUR_WANDB_KEY -e WANDB_PROJECT=CL4IE \\
  -e GPUS="0 1" -e JOBS_PER_GPU=2 -e BATCH_SIZE=16 \\
  -e SYNC_REMOTE="$SYNC_REMOTE" \\
  -e RCLONE_CONFIG_B64="\$(base64 -w0 $CONF)" \\
  -v "\$PWD/.hf_cache:/workspace/.hf_cache" \\
  --entrypoint bash doccl-grid -c "bash scripts/run_grid_multigpu.sh"
EOF
echo
echo "Resume guarantee: kill the instance any time; a fresh instance with the SAME"
echo "command pulls the .done markers and continues. (base64 -w0 may be 'base64 -b0' on macOS.)"
