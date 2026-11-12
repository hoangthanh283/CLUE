#!/usr/bin/env bash
# Clean-protocol queue v12 (2026-11-12, PM): DER++ on CIL-CORD with full-label F1 (Reviewer's list
# for a referee 4), same 250-token-document budget and overrides as v11's ER-250. Runs AFTER v11:
# waits for its pid, then takes the driver lock. The name matches after_clean's busy() pattern, so
# that script keeps waiting while this one waits or runs.
#   setsid nohup bash scripts/rca/clean_protocol_v12_2026-11.sh <v11-pid> &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:?pid of the running v11 driver}; P=results/clean_2026-11; Q=$P/queue.log
while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
L="model=layoutlmv3_base"
D="method=der_pp method.buffer_size=250 scenario=cil_cord"
echo "$(date '+%m-%d %H:%M') v12 driver running (pid $$)" >>$Q
run cil_cord_smoke_der_pp_e1_seed42 $D +method.run_alias=smoke_der_pp_e1 method.epochs=1 seed=42 $L
if grep -qs full_label_f1 $P/cil_cord_smoke_der_pp_e1_seed42*/metrics.json; then
  for s in 42 7 123; do run cil_cord_der_pp250_seed$s $D +method.run_alias=der_pp250 method.epochs=10 seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  der_pp block (smoke has no full_label_f1)" >>$Q; fi
echo "$(date '+%m-%d %H:%M') ALL DONE v12" >>$Q
