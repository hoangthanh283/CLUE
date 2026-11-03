#!/usr/bin/env bash
# Re-run the 12 reference rows (+ joint) on CIL-CORD and Mixed under the corrected token-CIL
# scoring (`training.cil_restrict_eval=true`, see train.py:restrict_preds). Legacy scoring
# charged correct cross-session predictions as false positives (Joint CIL-CORD 33 -> 93).
# Rented-GPU scheduler: LANES parallel jobs round-robin over GPUS. Resume-safe (SKIP on
# metrics.json). Runs land in results/rescore/<run>/ so legacy cells are never overwritten.
#   GPUS="0 1" LANES=4 BATCH_SIZE=16 bash scripts/rescore_cil_2026-11.sh
#   DRY_RUN=1 bash scripts/rescore_cil_2026-11.sh   # print the job plan
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS="${GPUS:-0}"; LANES="${LANES:-1}"; BATCH_SIZE="${BATCH_SIZE:-16}"; GRAD_CKPT="${GRAD_CKPT:-false}"
AMP="${AMP:-true}"; NUM_WORKERS="${NUM_WORKERS:-2}"; SEEDS="${SEEDS:-42 7 123}"; SCENARIOS="${SCENARIOS:-cil_cord mixed}"
OUT="${OUT:-results/rescore}"; LOG=$OUT/queue.log; mkdir -p $OUT
COMMON="model=layoutlmv3_base training.batch_size=$BATCH_SIZE training.gradient_checkpointing=$GRAD_CKPT training.amp=$AMP training.num_workers=$NUM_WORKERS wandb.mode=offline output_dir=$OUT +training.cil_restrict_eval=true"
AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
# name | method args  (one row per reference-table row; legacy run-dir naming preserved)
jobs() { for sd in $SEEDS; do for sc in $SCENARIOS; do
  echo "${sc}_naive_seed${sd}|method=naive method.epochs=100"
  echo "${sc}_lwf_seed${sd}|method=lwf method.epochs=100"
  echo "${sc}_er_seed${sd}|method=er method.epochs=100"
  echo "${sc}_der_pp_seed${sd}|method=der_pp method.epochs=100"
  echo "${sc}_coda_prompt_seed${sd}|method=coda_prompt method.epochs=30"
  echo "${sc}_cl_lora_seed${sd}|method=cl_lora method.epochs=30"
  echo "${sc}_cpfd_seed${sd}|method=cpfd method.epochs=100"
  echo "${sc}_is3_seed${sd}|method=is3 method.epochs=100"
  echo "${sc}_lca_seed${sd}|method=lca"
  echo "${sc}_colar_seed${sd}_k4_d50_r128|method=colar method.epochs=10 $AX"
  echo "${sc}_colar_r_sliced_seed${sd}_k4_d50_r128|method=colar_r_sliced method.epochs=10 $AX"
  echo "${sc}_colar_sbr_seed${sd}|method=colar_sbr method.epochs=10 $AX"
  echo "${sc}_lexslot_seed${sd}_off|method=lexslot method.slot_sharing=off method.epochs=100"
  echo "${sc}_joint_seed${sd}|method=joint method.epochs=100"
done; done; }
if [ "${DRY_RUN:-0}" = 1 ]; then jobs; exit 0; fi
run_one() { local spec=$1 lane=$2; local name=${spec%%|*} args=${spec#*|}
  local gpus=($GPUS); local gpu=${gpus[$((lane % ${#gpus[@]}))]}
  local sc=${name%%_*}; sc=$(echo "$name" | sed -E 's/^(cil_cord|mixed)_.*/\1/'); local sd=$(echo "$name" | grep -oE 'seed[0-9]+' | tr -d seed)
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name gpu$gpu" >>$LOG
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/train.py $args scenario=$sc seed=$sd $COMMON >"$OUT/$name.log" 2>&1
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
export -f run_one; export GPUS OUT LOG COMMON
echo "=== RESCORE START $(date) lanes=$LANES gpus=$GPUS ===" >>$LOG
i=0; jobs | while IFS= read -r spec; do echo "$spec|$((i % LANES))"; i=$((i+1)); done \
  | xargs -P "$LANES" -I{} bash -c 'spec="{}"; run_one "${spec%|*}" "${spec##*|}"'
echo "=== RESCORE COMPLETE $(date) ===" >>$LOG
