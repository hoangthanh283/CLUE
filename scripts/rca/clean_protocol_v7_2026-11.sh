#!/usr/bin/env bash
# Clean-protocol queue v7 (2026-11-11, Dev D14; PM order for the rewritten paper): CIL-CORD
# joint/naive; the true-label (oracle) memory refresh x3 (same recipe as the seed-42 gate run);
# Mixed SBR/joint; cil_funsd (10-doc memory); BROS; er_sbr; colar_mib (MiB-style unbiased CE,
# shared softmax); DIL joint/ER moved up; then the rest. Smokes gate their blocks (SKIP if no
# full_label_f1). Same run(), lock, orphan DONE/FAIL line, output dir and queue log as v6.
#   setsid nohup bash scripts/rca/clean_protocol_v7_2026-11.sh [pid-to-wait-for] [its-run-name] &
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
echo "$(date '+%m-%d %H:%M') v7 driver running (pid $$)" >>$Q
# 0. finished cells (skipped; re-run only if a metrics.json is missing)
for s in $S; do run cil_cord_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_recall_r64_seed$s $REC scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_noreplay_seed$s method=colar_sbr +method.run_alias=colar_sbr_noreplay method.docs_per_task=0 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# 1. CIL-CORD references
for s in $S; do run cil_cord_joint_seed$s method=joint method.epochs=100 scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_naive_seed$s method=naive method.epochs=10 scenario=cil_cord seed=$s $L; done
# 2. true-label memory refresh (oracle), recipe of results/gates/cil_protocol_2026-11/cil_cord_colar_r_oracle_seed42_k4_d50_r128
for s in $S; do run cil_cord_colar_r_oracle_seed$s method=colar_r_oracle method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# 3. Mixed headline
for s in $S; do run mixed_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_joint_seed$s method=joint method.epochs=100 scenario=mixed seed=$s $L; done
# 4. E2 smokes: joint pool, SBR (10-doc memory) and Recall on cil_funsd, 1 epoch, seed 42
FS="scenario=cil_funsd seed=42 $L"; LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
run cil_funsd_smoke_joint_e1_seed42 method=joint +method.run_alias=smoke_joint_e1 method.epochs=1 $FS
run cil_funsd_smoke_colar_sbr_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_e1 method.docs_per_task=10 $LR1 $FS
run cil_funsd_smoke_recall_e1_seed42 $RECB +method.run_alias=smoke_recall_e1 method.docs_per_task=0 $LR1 $FS
# 4b. E2: cil_funsd, 10 docs/session for CoLaR and CoLaR-SBR (colar dirs get _k4_d10_r128)
if smoke_ok cil_funsd_smoke_{joint,colar_sbr,recall}_e1_seed42; then
  F="scenario=cil_funsd $L"
  for s in $S; do run cil_funsd_colar_sbr_d10_seed$s method=colar_sbr +method.run_alias=colar_sbr_d10 method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_colar_sbr_recall_r64_seed$s $REC $F seed=$s; done
  for s in $S; do run cil_funsd_colar_seed${s}_k4_d10 method=colar method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_joint_seed$s method=joint method.epochs=100 $F seed=$s; done
  for s in $S; do run cil_funsd_naive_seed$s method=naive method.epochs=10 $F seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  E2 cil_funsd block (a smoke has no full_label_f1)" >>$Q; fi
# 5. second backbone for the method claim (run dir gets a _bros suffix; NAME stays a prefix)
for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR scenario=cil_cord seed=$s model=bros_base; done
# 6. E1: raw-document memory x blocked readout (er_sbr), smoke first
run cil_cord_smoke_er_sbr_e1_seed42 method=er_sbr +method.run_alias=smoke_er_sbr_e1 method.epochs=1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_er_sbr_e1_seed42; then
  for s in $S; do run cil_cord_er_sbr_seed$s method=er_sbr method.docs_per_task=50 method.epochs=10 method.replay_batch_size=4 scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  E1 er_sbr block (smoke has no full_label_f1)" >>$Q; fi
# 7. MiB-style unbiased CE on the CoLaR memory, shared softmax (no MiB distillation/init)
run cil_cord_smoke_colar_mib_e1_seed42 method=colar_mib +method.run_alias=smoke_colar_mib_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_mib_e1_seed42; then
  for s in $S; do run cil_cord_colar_mib_seed$s method=colar_mib method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  colar_mib block (smoke has no full_label_f1)" >>$Q; fi
# 8. DIL contrast (ER vs joint) under the clean protocol
for s in $S; do run dil_joint_seed$s method=joint method.epochs=100 scenario=dil seed=$s $L; done
for s in $S; do run dil_er_seed$s method=er method.epochs=10 scenario=dil seed=$s $L; done
# 9. CL-LoRA (one shared adapter, shared softmax)
for s in $S; do run cil_cord_cl_lora_seed$s method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L; done
# 10. Recall session-order check
for o in o2 o3; do run cil_cord_${o}_colar_sbr_recall_r64_seed42 $REC scenario=cil_cord_$o seed=42 $L; done
# 11. ER-250
for s in $S; do run cil_cord_er250_seed$s $ER250 seed=$s $L; done
# 12. LwF in fp32 (fp16 collapses it to 0)
for s in $S; do run cil_cord_lwf_seed$s method=lwf method.epochs=10 scenario=cil_cord seed=$s $L; done
# 13. rest of DIL
for s in $S; do run dil_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_naive_seed$s method=naive method.epochs=10 scenario=dil seed=$s $L; done
# 14. rest of Mixed (no full-label set there; the hook is a no-op)
for s in $S; do run mixed_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_naive_seed$s method=naive method.epochs=10 scenario=mixed seed=$s $L; done
# 15. last, only if reached: SBR-Recall on Mixed WITH the 50-document memory
for s in 7 123; do run mixed_colar_sbr_recall_r64c_seed$s $RECC scenario=mixed seed=$s $L; done
echo "$(date '+%m-%d %H:%M') ALL DONE" >>$Q
