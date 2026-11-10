#!/usr/bin/env bash
# Clean-protocol priority set for the CVPR rewrite (plan 2026-11-11, review items C4/C5/I2).
# ONE protocol for every row: fp32, early stopping on a 10% held-out split of the session's
# TRAIN data (never the test split), <=10 epochs per session (joint: 100), session-restricted
# scoring, full-label final F1 on the unmasked CORD test set, seeds 42/7/123.
# Resume-safe (skips a run whose metrics.json exists). One job at a time (6 GB box).
#   setsid nohup bash scripts/rca/clean_protocol_2026-11.sh [pid-to-wait-for] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; fi
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
REC="method=colar_sbr_recall +method.run_alias=colar_sbr_recall_r64 method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense] method.docs_per_task=0 $LR"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
L="model=layoutlmv3_base"; S="42 7 123"
# 0. smoke: does the full-label hook write its number? (naive, 1 epoch)
run cil_cord_smoke_naive_e1_seed42 method=naive +method.run_alias=smoke_naive_e1 method.epochs=1 scenario=cil_cord seed=42 $L
# A. headline rows
for s in $S; do run cil_cord_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_recall_r64_seed$s $REC scenario=cil_cord seed=$s $L; done
# B. the other two cells of the 2x2
for s in $S; do run cil_cord_colar_sbr_noreplay_seed$s method=colar_sbr +method.run_alias=colar_sbr_noreplay method.docs_per_task=0 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# C. references
for s in $S; do run cil_cord_joint_seed$s method=joint method.epochs=100 scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_naive_seed$s method=naive method.epochs=10 scenario=cil_cord seed=$s $L; done
# D. Recall session-order check (base is frozen after session 0)
for o in o2 o3; do run cil_cord_${o}_colar_sbr_recall_r64_seed42 $REC scenario=cil_cord_$o seed=42 $L; done
# E. second backbone for the method claim
for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR scenario=cil_cord seed=$s model=bros_base; done
# F. raw replay with matched memory (250 documents = 50 per session)
for s in $S; do run cil_cord_er250_seed$s method=er +method.run_alias=er250 method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L; done
# G. Mixed under the same protocol (no full-label set there; the hook is a no-op)
for s in $S; do run mixed_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_joint_seed$s method=joint method.epochs=100 scenario=mixed seed=$s $L; done
for s in $S; do run mixed_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_naive_seed$s method=naive method.epochs=10 scenario=mixed seed=$s $L; done
# H. DIL under the same protocol
for s in $S; do run dil_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_joint_seed$s method=joint method.epochs=100 scenario=dil seed=$s $L; done
for s in $S; do run dil_naive_seed$s method=naive method.epochs=10 scenario=dil seed=$s $L; done
for s in $S; do run dil_er_seed$s method=er method.epochs=10 scenario=dil seed=$s $L; done
echo "$(date '+%m-%d %H:%M') ALL DONE" >>$Q
