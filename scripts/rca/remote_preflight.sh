#!/usr/bin/env bash
# Pre-flight a rented box BEFORE cloning or spending metered time (plan rev. 9, Tier B).
#   usage: bash scripts/rca/remote_preflight.sh <user@host> [ssh-port]
# Checks: GPU + driver CUDA version, host RAM (>= 32 GB to absorb the Mixed LCA cells, >= 8 GB
# for the image cells alone), free disk (>= 40 GB), CPU cores (>= 4), download bandwidth to the
# ImageNet-R host, git/curl presence. Read-only on the remote; prints PASS/WARN per item.
set -u
T="${1:?user@host}"; P="${2:-22}"
S="ssh -p $P -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new $T"
echo "== $T:$P =="
$S 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' 2>/dev/null \
  || { echo "FAIL: no nvidia-smi / ssh refused"; exit 1; }
$S 'nvidia-smi | grep -oE "CUDA Version: [0-9.]+"'
$S 'awk "/MemTotal/{printf \"RAM %.1f GB  \", \$2/1048576}" /proc/meminfo; nproc | xargs printf "cores %s  "; df -BG --output=avail / | tail -1 | xargs printf "disk free %sB\n"'
$S 'command -v git >/dev/null && echo "git ok" || echo "WARN: git missing"; command -v curl >/dev/null && echo "curl ok" || echo "WARN: curl missing"'
echo "-- bandwidth to the ImageNet-R host (10 s sample) --"
$S 'curl -sL --max-time 10 -o /dev/null -w "%{speed_download}" https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar 2>/dev/null' \
  | awk '{printf "download %.1f MB/s (%s)\n", $1/1048576, ($1>12e6?"ok: 2 GB in < 3 min":"WARN: slow host")}'
$S 'awk "/MemTotal/{exit !(\$2>=32*1048576)}" /proc/meminfo' && echo "RAM >= 32 GB: can also take the Mixed LCA cells" || echo "RAM < 32 GB: image cells only (Mixed LCA stays local)"
