#!/bin/bash
# Chain docs retry AFTER lexslot probe completes.
set -u
cd "$(dirname "$0")/../.."
until grep -q 'LEXSLOT PROBE COMPLETE' results/logs_vision/pass2b_lexslot_probe.log 2>/dev/null; do sleep 60; done
exec bash scripts/rca/probe_docs_retry.sh
