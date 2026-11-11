#!/usr/bin/env bash
# Clean-protocol queue v6 (2026-11-11, Dev D13; PM decisions on D12): v5 up to and including
# mixed_joint; then cil_funsd (secondary replication) with a 10-doc/session memory for CoLaR
# and CoLaR-SBR (~7% of a session, CORD's fraction), smokes first, block skipped if a smoke
# lacks full_label_f1; then BROS; then E1 = er_sbr (raw whole-document memory, nothing frozen,
# blocked readout), smoke first, same skip rule; then the rest of v4 unchanged.
# New vs v5: arg 2 names the orphan job of arg 1, logged DONE/FAIL once it exits; the wait
# loop's sleep closes fd 9 so a killed driver frees the lock at once.
#   setsid nohup bash scripts/rca/clean_protocol_v6_2026-11.sh [pid-to-wait-for] [its-run-name] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; ORPHAN=${2:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
# ponytail: lock held by the driver only (children close fd 9), so killing the driver frees it.
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60 9>&-; done; fi
if [ -n "$ORPHAN" ]; then
  if ls $P/${ORPHAN}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $ORPHAN" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $ORPHAN" >>$Q; fi
fi
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
REC="$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR"
RECC="$RECB +method.run_alias=colar_sbr_recall_r64c method.docs_per_task=50 $LR"  # with the 50-doc memory
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
smoke_ok() {  # smoke_ok <run-name>...: every smoke wrote full_label_f1
  local n; for n in "$@"; do grep -qs full_label_f1 $P/$n/metrics.json || return 1; done
}
L="model=layoutlmv3_base"; S="42 7 123"
ER250="method=er +method.run_alias=er250 method.buffer_size=250 method.epochs=10 scenario=cil_cord"
echo "$(date '+%m-%d %H:%M') v6 driver running (pid $$)" >>$Q
# a. headline rows
for s in $S; do run cil_cord_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_recall_r64_seed$s $REC scenario=cil_cord seed=$s $L; done
# b. the other two cells of the 2x2
for s in $S; do run cil_cord_colar_sbr_noreplay_seed$s method=colar_sbr +method.run_alias=colar_sbr_noreplay method.docs_per_task=0 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# c. references
for s in $S; do run cil_cord_joint_seed$s method=joint method.epochs=100 scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_naive_seed$s method=naive method.epochs=10 scenario=cil_cord seed=$s $L; done
# d. per-session isolation with a shared softmax (2x2 cell), then matched raw replay seed 42
for s in $S; do run cil_cord_cl_lora_seed$s method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L; done
run cil_cord_er250_seed42 $ER250 seed=42 $L
# e. Recall session-order check (base is frozen after session 0)
for o in o2 o3; do run cil_cord_${o}_colar_sbr_recall_r64_seed42 $REC scenario=cil_cord_$o seed=42 $L; done
# f. Mixed headline (the paper's Mixed claim rests on these)
for s in $S; do run mixed_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_joint_seed$s method=joint method.epochs=100 scenario=mixed seed=$s $L; done
# f2. E2 smokes: joint pool, SBR (10-doc memory) and Recall on cil_funsd, 1 epoch, seed 42
FS="scenario=cil_funsd seed=42 $L"; LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
run cil_funsd_smoke_joint_e1_seed42 method=joint +method.run_alias=smoke_joint_e1 method.epochs=1 $FS
run cil_funsd_smoke_colar_sbr_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_e1 method.docs_per_task=10 $LR1 $FS
run cil_funsd_smoke_recall_e1_seed42 $RECB +method.run_alias=smoke_recall_e1 method.docs_per_task=0 $LR1 $FS
# f3. E2: cil_funsd, 10 docs/session for CoLaR and CoLaR-SBR (colar dirs get _k4_d10_r128)
if smoke_ok cil_funsd_smoke_{joint,colar_sbr,recall}_e1_seed42; then
  F="scenario=cil_funsd $L"
  for s in $S; do run cil_funsd_colar_sbr_d10_seed$s method=colar_sbr +method.run_alias=colar_sbr_d10 method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_colar_sbr_recall_r64_seed$s $REC $F seed=$s; done
  for s in $S; do run cil_funsd_colar_seed${s}_k4_d10 method=colar method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_joint_seed$s method=joint method.epochs=100 $F seed=$s; done
  for s in $S; do run cil_funsd_naive_seed$s method=naive method.epochs=10 $F seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  E2 cil_funsd block (a smoke has no full_label_f1)" >>$Q; fi
# g. second backbone for the method claim (run dir gets a _bros suffix; NAME stays a prefix)
for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR scenario=cil_cord seed=$s model=bros_base; done
# g2. E1: raw-document memory x blocked readout (er_sbr), smoke first
run cil_cord_smoke_er_sbr_e1_seed42 method=er_sbr +method.run_alias=smoke_er_sbr_e1 method.epochs=1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_er_sbr_e1_seed42; then
  for s in $S; do run cil_cord_er_sbr_seed$s method=er_sbr method.docs_per_task=50 method.epochs=10 method.replay_batch_size=4 scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  E1 er_sbr block (smoke has no full_label_f1)" >>$Q; fi
# h. remaining ER-250 seeds, then LwF in fp32 (fp16 collapses it to 0)
for s in 7 123; do run cil_cord_er250_seed$s $ER250 seed=$s $L; done
for s in $S; do run cil_cord_lwf_seed$s method=lwf method.epochs=10 scenario=cil_cord seed=$s $L; done
# i. rest of Mixed (no full-label set there; the hook is a no-op)
for s in $S; do run mixed_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_naive_seed$s method=naive method.epochs=10 scenario=mixed seed=$s $L; done
# j. DIL under the same protocol
for s in $S; do run dil_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_joint_seed$s method=joint method.epochs=100 scenario=dil seed=$s $L; done
for s in $S; do run dil_naive_seed$s method=naive method.epochs=10 scenario=dil seed=$s $L; done
for s in $S; do run dil_er_seed$s method=er method.epochs=10 scenario=dil seed=$s $L; done
# k. last, only if reached: SBR-Recall on Mixed WITH the 50-document memory
for s in 7 123; do run mixed_colar_sbr_recall_r64c_seed$s $RECC scenario=mixed seed=$s $L; done
echo "$(date '+%m-%d %H:%M') ALL DONE" >>$Q
