#!/usr/bin/env bash
# Clean-protocol queue driver v18 (2026-11-15). One new block, then hands over to v17 (which skips
# every finished run and continues with BROS controls -> v13 list).
#  S-d. cil_funsd current-side-only repair (user go-ahead 2026-11-15; R22 L3): plain CoLaR, 10 forms,
#       current-session CE sliced (O tokens over {O} + new classes), replay loss untouched
#       (method=colar +method.current_sliced=true). Smoke first, gated on full_label_f1.
#   setsid nohup bash scripts/rca/clean_protocol_v18_2026-11.sh [pid-to-wait-for] [its-run-name] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; ORPHAN=${2:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60 9>&-; done; fi
if [ -n "$ORPHAN" ]; then
  if ls $P/${ORPHAN}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $ORPHAN" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $ORPHAN" >>$Q; fi
fi
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
smoke_ok() { local n; for n in "$@"; do grep -qs full_label_f1 $P/$n*/metrics.json || return 1; done; }
L="model=layoutlmv3_base"; S="42 7 123"
echo "$(date '+%m-%d %H:%M') v18 driver running (pid $$)" >>$Q
C="method=colar +method.current_sliced=true method.docs_per_task=10 scenario=cil_funsd $L"
run cil_funsd_smoke_colar_cur_sliced_e1_seed42 $C +method.run_alias=smoke_colar_cur_sliced_e1 $LR1 seed=42
if smoke_ok cil_funsd_smoke_colar_cur_sliced_e1_seed42; then
  for s in $S; do run cil_funsd_colar_cur_sliced_seed$s $C +method.run_alias=colar_cur_sliced $LR seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  S-d cil_funsd colar_cur_sliced block (smoke has no full_label_f1)" >>$Q; fi
echo "$(date '+%m-%d %H:%M') v18 block done; handing over to v17" >>$Q
exec bash scripts/rca/clean_protocol_v17_2026-11.sh
