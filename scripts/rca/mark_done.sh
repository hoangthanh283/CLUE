#!/bin/bash
# Phase 3 pre-step (plan rev. 8): scripts/analyze_results.py ingests a run only if
# results/<run>/.done exists next to metrics.json. The grid drivers write that marker; the
# Phase-1 / Tier-A queues and the manual CoLaR cells do not. Mark every cell whose
# metrics.json carries a final AA (partial files from killed runs have none). Idempotent.
set -u
cd "$(dirname "$0")/../.."
n=0
for mp in results/*/metrics.json; do
  d=$(dirname "$mp")
  [ -f "$d/.done" ] && continue
  python3 -c "import json,sys; d=json.load(open('$mp')); sys.exit(0 if d.get('AA') is not None else 1)" 2>/dev/null \
    && { touch "$d/.done"; n=$((n+1)); }
done
echo "marked $n run dirs"
