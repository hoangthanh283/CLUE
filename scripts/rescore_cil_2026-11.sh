#!/usr/bin/env bash
# Re-run the reference rows on CIL-CORD and Mixed under the corrected token-CIL scoring
# (`training.cil_restrict_eval=true`, see train.py:restrict_preds). Legacy scoring charged correct
# cross-session predictions as false positives (Joint CIL-CORD 33 -> 93).
#
# Each job writes to its own dir  $OUT/<job>/<run>/  and is resume-safe (skipped when its
# metrics.json exists), so a local box and a rented box can share the same plan: run it here,
# rsync results/rescore/ to the rented box, run the same command there, rsync back.
# Out-of-memory: a job is retried once at batch 1 (CoLaR rows also halve their replay batch);
# if it still fails it is appended to $OUT/needs_rent.txt (OOM) or $OUT/failures.txt (other error).
#
#   ROWS=main MODELS=layoutlmv3_base bash scripts/rescore_cil_2026-11.sh      # main table
#   ROWS=grid MODELS="bert_base bros_base lilt_base" bash scripts/...        # backbone grid
#   DRY_RUN=1 ...                                                            # print the plan
#   GPUS="0 1" LANES=4 BATCH_SIZE=16 ...                                     # rented box
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS="${GPUS:-0}"; LANES="${LANES:-1}"; BATCH_SIZE="${BATCH_SIZE:-16}"; GRAD_CKPT="${GRAD_CKPT:-false}"
AMP="${AMP:-true}"; NUM_WORKERS="${NUM_WORKERS:-2}"; SEEDS="${SEEDS:-42 7 123}"
SCENARIOS="${SCENARIOS:-cil_cord mixed}"; MODELS="${MODELS:-layoutlmv3_base}"; ROWS="${ROWS:-main}"
OUT="${OUT:-results/rescore}"; LOG=$OUT/queue.log; mkdir -p $OUT
AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"

row_args() {  # row -> method args
  case $1 in
    naive) echo "method=naive method.epochs=100" ;;
    joint) echo "method=joint method.epochs=100 +training.val_frac=0.1" ;;  # pooled CIL-masked eval is a broken early-stop signal (flat 36 on CIL-CORD); hold out 10% of the full-label pool
    ewc) echo "method=ewc method.epochs=100" ;;
    lwf) echo "method=lwf method.epochs=100" ;;
    er) echo "method=er method.epochs=100" ;;
    der_pp) echo "method=der_pp method.epochs=100" ;;
    coda_prompt) echo "method=coda_prompt method.epochs=30" ;;
    cl_lora) echo "method=cl_lora method.epochs=30" ;;
    cpfd) echo "method=cpfd method.epochs=100" ;;
    is3) echo "method=is3 method.epochs=100" ;;
    lca) echo "method=lca" ;;
    colar) echo "method=colar method.epochs=10 $AX" ;;
    colar_r_sliced) echo "method=colar_r_sliced method.epochs=10 $AX" ;;
    colar_sbr) echo "method=colar_sbr method.epochs=10 $AX" ;;
    lexslot_off) echo "method=lexslot method.slot_sharing=off method.epochs=100" ;;
  esac
}
case $ROWS in
  main) ROW_LIST="joint naive lwf er der_pp coda_prompt cl_lora cpfd is3 lca colar colar_r_sliced colar_sbr lexslot_off" ;;  # joint first: the ceiling gates every claim
  grid) ROW_LIST="naive joint ewc lwf er der_pp colar colar_sbr" ;;
  *) ROW_LIST="$ROWS" ;;
esac
fam() { case $1 in layoutlmv3_base) echo "" ;; bert_base) echo "_bert" ;; bros_base) echo "_bros" ;; lilt_base) echo "_lilt" ;; *) echo "_$1" ;; esac; }
start_bs() { case $1 in bros_base|lilt_base) echo 1 ;; *) echo "$BATCH_SIZE" ;; esac; }

# job spec: key|model|scenario|seed|row
jobs() { for sd in $SEEDS; do for model in $MODELS; do for sc in $SCENARIOS; do for row in $ROW_LIST; do
  echo "${sc}_${row}_seed${sd}$(fam $model)|$model|$sc|$sd|$row"
done; done; done; done; }
if [ "${DRY_RUN:-0}" = 1 ]; then jobs; exit 0; fi

run_one() { local spec=$1 lane=$2
  IFS='|' read -r key model sc sd row <<<"$spec"
  local gpus=($GPUS); local gpu=${gpus[$((lane % ${#gpus[@]}))]}
  local dir=$OUT/$key args; args=$(row_args "$row")
  if ls $dir/*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $key" >>$LOG; return; fi
  local bs; bs=$(start_bs "$model")
  for attempt in 1 2; do
    rm -rf "$dir"; mkdir -p "$dir"
    echo "$(date '+%m-%d %H:%M') START $key gpu$gpu bs$bs" >>$LOG
    local extra=""; [ "$bs" = 1 ] && [[ "$row" == colar* ]] && extra="method.replay_batch_size=2"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/train.py $args $extra scenario=$sc seed=$sd \
      model=$model training.batch_size=$bs training.gradient_checkpointing=$GRAD_CKPT training.amp=$AMP \
      training.num_workers=$NUM_WORKERS wandb.mode=offline output_dir=$dir +training.cil_restrict_eval=true \
      >"$dir.log" 2>&1
    if ls $dir/*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $key" >>$LOG; return; fi
    if grep -aq "out of memory" "$dir.log" && [ "$bs" != 1 ]; then bs=1; continue; fi
    break
  done
  if grep -aq "out of memory" "$dir.log"; then
    echo "$key|$model|$sc|$sd|$row  # OOM at batch 1 on the local 6 GB card" >>$OUT/needs_rent.txt
    echo "$(date '+%m-%d %H:%M') OOM   $key -> needs_rent.txt" >>$LOG
  else
    echo "$key  # $(tr '\r' '\n' <"$dir.log" | grep -a "Error" | tail -1 | cut -c1-200)" >>$OUT/failures.txt
    echo "$(date '+%m-%d %H:%M') FAIL  $key -> failures.txt" >>$LOG
  fi
}
export -f run_one row_args start_bs; export GPUS OUT LOG AX BATCH_SIZE GRAD_CKPT AMP NUM_WORKERS
echo "=== RESCORE START $(date) rows=$ROWS models=$MODELS lanes=$LANES ===" >>$LOG
i=0; jobs | while IFS= read -r spec; do echo "$spec#$((i % LANES))"; i=$((i+1)); done \
  | xargs -P "$LANES" -I{} bash -c 'spec="{}"; run_one "${spec%#*}" "${spec##*#}"'
echo "=== RESCORE COMPLETE $(date) rows=$ROWS models=$MODELS ===" >>$LOG
