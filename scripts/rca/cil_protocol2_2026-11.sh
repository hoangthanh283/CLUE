#!/usr/bin/env bash
# G0 follow-ups (2026-11-02): double-sliced CoLaR-R under the corrected scorer.
#   seeds 7/123 on CIL-CORD; Mixed seed 42; sliced2+oracle ceiling seed 42; then audit gates.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/gates/cil_protocol_2026-11; LOG=$OUT/queue.log
AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
run() { local name=$1 sc=$2 sd=$3 me=$4
  local COMMON="scenario=$sc model=layoutlmv3_base seed=$sd training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$OUT +training.cil_restrict_eval=true"
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  uv run python scripts/train.py method=$me method.epochs=10 $AX $COMMON >"$OUT/$name.log" 2>&1
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
echo "=== G0 FOLLOW-UPS START $(date) ===" >>$LOG
run cil_cord_colar_r_sliced2_seed7    cil_cord 7   colar_r_sliced2
run cil_cord_colar_r_sliced2_seed123  cil_cord 123 colar_r_sliced2
run mixed_colar_r_sliced2_seed42      mixed    42  colar_r_sliced2
run cil_cord_colar_r_sliced2_oracle_seed42 cil_cord 42 colar_r_sliced2_oracle
run mixed_joint_seed42                mixed    42  joint
echo "=== G0 FOLLOW-UPS COMPLETE $(date) ===" >>$LOG
bash scripts/rca/audit_gates_2026-11.sh
