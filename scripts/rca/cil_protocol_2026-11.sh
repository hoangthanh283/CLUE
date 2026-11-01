#!/usr/bin/env bash
# CIL-CORD evaluation-protocol batch (2026-11-01): re-score the seed-42 reference cells with
# `training.cil_restrict_eval=true` (predictions outside session j's entity types -> O when
# scoring R[t, j]). Legacy scoring charged Joint 70/32/25/26/14 for correct cross-session
# predictions. Afterwards chains the audit gates (resume-safe).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/gates/cil_protocol_2026-11; LOG=$OUT/queue.log; mkdir -p $OUT
COMMON="scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$OUT +training.cil_restrict_eval=true"
AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
run() { local name=$1; shift
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  "$@" >"$OUT/$name.log" 2>&1
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
echo "=== CIL PROTOCOL BATCH START $(date) ===" >>$LOG
run cil_cord_joint_seed42  uv run python scripts/train.py method=joint method.epochs=100 $COMMON
run cil_cord_naive_seed42  uv run python scripts/train.py method=naive method.epochs=100 $COMMON
run cil_cord_er_seed42     uv run python scripts/train.py method=er    method.epochs=100 $COMMON
run cil_cord_colar_seed42  uv run python scripts/train.py method=colar method.epochs=10 $AX $COMMON
run cil_cord_colar_r_sliced_seed42 uv run python scripts/train.py method=colar_r_sliced method.epochs=10 $AX $COMMON
run cil_cord_colar_r_oracle_seed42 uv run python scripts/train.py method=colar_r_oracle method.epochs=10 $AX $COMMON
echo "=== CIL PROTOCOL BATCH COMPLETE $(date) ===" >>$LOG
bash scripts/rca/audit_gates_2026-11.sh
