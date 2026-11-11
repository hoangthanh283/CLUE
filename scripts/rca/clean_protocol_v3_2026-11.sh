#!/usr/bin/env bash
# Clean-protocol queue v3 (2026-11-11): v2 plus a capacity-matched CL-LoRA (rank 64, Q/K/V +
# dense, layers 4..11 = SBR-Recall's 48 sites) after plain CL-LoRA. v2 was: v1 plus CL-LoRA and LwF
# under the clean protocol, with the 2x2-table cells first and the Mixed SBR-Recall memory
# runs last. Same protocol, run(), output dir and queue log as v1, so finished v1 runs are
# skipped (resume-safe on metrics.json). One job at a time (6 GB box).
#   setsid nohup bash scripts/rca/clean_protocol_v3_2026-11.sh [pid-to-wait-for] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60; done; fi
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
REC="$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR"
RECC="$RECB +method.run_alias=colar_sbr_recall_r64c method.docs_per_task=50 $LR"  # with the 50-doc memory
CLR64="method=cl_lora +method.run_alias=cl_lora_r64 method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense] +method.layers_to_transform=[4,5,6,7,8,9,10,11] method.epochs=10"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
L="model=layoutlmv3_base"; S="42 7 123"
echo "$(date '+%m-%d %H:%M') v3 driver running (pid $$)" >>$Q
# 1-2. headline rows
for s in $S; do run cil_cord_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_recall_r64_seed$s $REC scenario=cil_cord seed=$s $L; done
# 3. the other two cells of the 2x2
for s in $S; do run cil_cord_colar_sbr_noreplay_seed$s method=colar_sbr +method.run_alias=colar_sbr_noreplay method.docs_per_task=0 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# 4. references
for s in $S; do run cil_cord_joint_seed$s method=joint method.epochs=100 scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_naive_seed$s method=naive method.epochs=10 scenario=cil_cord seed=$s $L; done
# 5. per-session isolation with a shared softmax (2x2 cell): CL-LoRA, config defaults
for s in $S; do run cil_cord_cl_lora_seed$s method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L; done
# 5b. capacity-matched CL-LoRA: same 48 LoRA sites, rank and alpha as SBR-Recall (7.08M LoRA params)
for s in $S; do run cil_cord_cl_lora_r64_seed$s $CLR64 scenario=cil_cord seed=$s $L; done
# 6. raw replay with matched memory (250 documents = 50 per session), seed 42 first
run cil_cord_er250_seed42 method=er +method.run_alias=er250 method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=42 $L
# 7. Recall session-order check (base is frozen after session 0)
for o in o2 o3; do run cil_cord_${o}_colar_sbr_recall_r64_seed42 $REC scenario=cil_cord_$o seed=42 $L; done
# 8. second backbone for the method claim (run dir gets a _bros suffix; NAME stays a prefix)
for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR scenario=cil_cord seed=$s model=bros_base; done
# 9. remaining ER-250 seeds
for s in 7 123; do run cil_cord_er250_seed$s method=er +method.run_alias=er250 method.buffer_size=250 method.epochs=10 scenario=cil_cord seed=$s $L; done
# 10. LwF in fp32 (fp16 collapses it to 0)
for s in $S; do run cil_cord_lwf_seed$s method=lwf method.epochs=10 scenario=cil_cord seed=$s $L; done
# 11a. Mixed under the same protocol (no full-label set there; the hook is a no-op)
for s in $S; do run mixed_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_joint_seed$s method=joint method.epochs=100 scenario=mixed seed=$s $L; done
for s in $S; do run mixed_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_naive_seed$s method=naive method.epochs=10 scenario=mixed seed=$s $L; done
# 11b. DIL under the same protocol
for s in $S; do run dil_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_joint_seed$s method=joint method.epochs=100 scenario=dil seed=$s $L; done
for s in $S; do run dil_naive_seed$s method=naive method.epochs=10 scenario=dil seed=$s $L; done
for s in $S; do run dil_er_seed$s method=er method.epochs=10 scenario=dil seed=$s $L; done
# 12. last, only if reached: SBR-Recall on Mixed WITH the 50-document memory
for s in 7 123; do run mixed_colar_sbr_recall_r64c_seed$s $RECC scenario=mixed seed=$s $L; done
echo "$(date '+%m-%d %H:%M') ALL DONE" >>$Q
