#!/usr/bin/env bash
# Clean-protocol queue v16 (2026-11-14, Dev): v15 plus block E2b (mask-only oracle, R18 E2b) right before block B.
# v15 (2026-11-14, PM): v14 (R18 E1/E2) plus block B, BROS controls: the two acceptance experiments
# first, then v13's whole job list (exec; its run() skips finished dirs, its r64 check still applies).
#  E2a. cil_cord colar_r_sliced x3 (replay CE over capture-time classes only), smoke first.
#  E1.  cil_funsd: comparison cells (CoLaR-SBR d10, CoLaR d10, joint, naive; v13's names/overrides)
#       plus colar_r_oracle_pre and colar_sbr_const_o x3, smokes first. The oracle block runs only if
#       its smoke log shows relabelled tokens > 0 (else it is silently plain CoLaR).
#  E2b. cil_cord colar_r_oracle_mask_pre x3 (every false O of the memory masked, no label added), smoke first;
#       runs only if the smoke wrote full_label_f1 AND its log shows masked tokens > 0.
#   setsid nohup bash scripts/rca/clean_protocol_v16_2026-11.sh [pid-to-wait-for] [its-run-name] &
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
echo "$(date '+%m-%d %H:%M') v16 driver running (pid $$)" >>$Q
# E2a
run cil_cord_smoke_colar_r_sliced_e1_seed42 method=colar_r_sliced method.run_alias=smoke_colar_r_sliced_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_r_sliced_e1_seed42; then
  for s in $S; do run cil_cord_colar_r_sliced_seed$s method=colar_r_sliced method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  colar_r_sliced block (smoke has no full_label_f1)" >>$Q; fi
# E1
FS="scenario=cil_funsd seed=42 $L"; F="scenario=cil_funsd $L"
run cil_funsd_smoke_joint_e1_seed42 method=joint +method.run_alias=smoke_joint_e1 method.epochs=1 $FS
run cil_funsd_smoke_colar_sbr_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_e1 method.docs_per_task=10 $LR1 $FS
run cil_funsd_smoke_colar_r_oracle_pre_e1_seed42 method=colar_r_oracle_pre method.run_alias=smoke_colar_r_oracle_pre_e1 method.docs_per_task=10 $LR1 $FS
run cil_funsd_smoke_colar_sbr_const_o_e1_seed42 method=colar_sbr_const_o method.run_alias=smoke_colar_sbr_const_o_e1 method.docs_per_task=10 $LR1 $FS
if smoke_ok cil_funsd_smoke_{joint,colar_sbr,colar_r_oracle_pre,colar_sbr_const_o}_e1_seed42; then
  if grep -Eqs 'colar_r\[oracle\]: refreshed .* [1-9][0-9]* tokens relabelled' $P/cil_funsd_smoke_colar_r_oracle_pre_e1_seed42.log; then
    for s in $S; do run cil_funsd_colar_r_oracle_pre_seed$s method=colar_r_oracle_pre method.docs_per_task=10 $LR $F seed=$s; done
  else echo "$(date '+%m-%d %H:%M') SKIP  cil_funsd oracle block (smoke relabelled 0 tokens)" >>$Q; fi
  for s in $S; do run cil_funsd_colar_sbr_const_o_seed$s method=colar_sbr_const_o method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_colar_sbr_d10_seed$s method=colar_sbr +method.run_alias=colar_sbr_d10 method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_colar_seed${s}_k4_d10 method=colar method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_joint_seed$s method=joint method.epochs=100 $F seed=$s; done
  for s in $S; do run cil_funsd_naive_seed$s method=naive method.epochs=10 $F seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  E1 cil_funsd block (a smoke has no full_label_f1)" >>$Q; fi
# E2b. mask-only oracle on cil_cord (R18 E2b), smoke first, gated on masked tokens > 0
run cil_cord_smoke_colar_r_oracle_mask_pre_e1_seed42 method=colar_r_oracle_mask_pre method.run_alias=smoke_colar_r_oracle_mask_pre_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_r_oracle_mask_pre_e1_seed42 \
   && grep -Eqs 'colar_r\[oracle_mask\]: refreshed [0-9]+ docs, [1-9][0-9]* tokens masked' $P/cil_cord_smoke_colar_r_oracle_mask_pre_e1_seed42.log; then
  for s in $S; do run cil_cord_colar_r_oracle_mask_pre_seed$s method=colar_r_oracle_mask_pre method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  E2b oracle_mask block (smoke has no full_label_f1 or masked 0 tokens)" >>$Q; fi
# B. BROS controls on cil_cord (user go-ahead 2026-11-14): CoLaR-SBR (v13's name/overrides), oracle-pre,
#    constant background, plain CoLaR; smokes first, oracle gated on relabelled tokens > 0.
B="scenario=cil_cord model=bros_base"
run cil_cord_smoke_colar_r_oracle_pre_bros_e1_seed42 method=colar_r_oracle_pre method.run_alias=smoke_colar_r_oracle_pre_bros_e1 method.docs_per_task=50 $LR1 $B seed=42
run cil_cord_smoke_colar_sbr_const_o_bros_e1_seed42 method=colar_sbr_const_o method.run_alias=smoke_colar_sbr_const_o_bros_e1 method.docs_per_task=50 $LR1 $B seed=42
if smoke_ok cil_cord_smoke_{colar_r_oracle_pre,colar_sbr_const_o}_bros_e1_seed42; then
  for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR $B seed=$s; done
  if grep -Eqs 'colar_r\[oracle\]: refreshed .* [1-9][0-9]* tokens relabelled' $P/cil_cord_smoke_colar_r_oracle_pre_bros_e1_seed42.log; then
    for s in $S; do run cil_cord_colar_r_oracle_pre_bros_seed$s method=colar_r_oracle_pre method.run_alias=colar_r_oracle_pre_bros method.docs_per_task=50 $LR $B seed=$s; done
  else echo "$(date '+%m-%d %H:%M') SKIP  BROS oracle block (smoke relabelled 0 tokens)" >>$Q; fi
  for s in $S; do run cil_cord_colar_sbr_const_o_bros_seed$s method=colar_sbr_const_o method.run_alias=colar_sbr_const_o_bros method.docs_per_task=50 $LR $B seed=$s; done
  for s in $S; do run cil_cord_colar_bros_seed$s method=colar +method.run_alias=colar_bros method.docs_per_task=50 $LR $B seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  BROS controls block (a smoke has no full_label_f1)" >>$Q; fi
echo "$(date '+%m-%d %H:%M') v16 blocks done; handing over to v13 list" >>$Q
exec bash scripts/rca/clean_protocol_v13_2026-11.sh
