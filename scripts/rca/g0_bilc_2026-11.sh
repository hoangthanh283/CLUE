#!/usr/bin/env bash
# BiLC orchestrator (2026-11-02). Waits for the detached Mixed-joint train (PID $WAIT_PID) to
# exit, then: G0c replay-ratio probe, G0d ceiling, E1 forward completion, E2 BiLC — all
# CIL-CORD seed 42 under the corrected scorer — then the image audit gates.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=results/gates/cil_protocol_2026-11; LOG=$P/queue.log
WAIT_PID="${WAIT_PID:-0}"
while [ "$WAIT_PID" != 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
if ls $P/mixed_joint_seed42/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  mixed_joint_seed42" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  mixed_joint_seed42" >>$LOG; fi
AX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128"
run() { local name=$1 me=$2 sd=$3; shift 3
  local COMMON="scenario=cil_cord model=layoutlmv3_base seed=$sd training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true"
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  uv run python scripts/train.py method=$me method.epochs=10 $AX "$@" $COMMON >"$P/$name.log" 2>&1
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
echo "=== BiLC BATCH START $(date) ===" >>$LOG
run cil_cord_colar_r_sliced2_pseudo_seed42 colar_r_sliced2_pseudo 42 method.replay_batch_size=4
run cil_cord_colar_r_bilc_seed42          colar_r_bilc           42 method.replay_batch_size=4
run cil_cord_colar_r_sliced2_oracle_seed42 colar_r_sliced2_oracle 42 method.replay_batch_size=4
run cil_cord_colar_r_sliced2_rb2_seed123  colar_r_sliced2        123 method.run_alias=colar_r_sliced2_rb2 method.replay_batch_size=2
echo "=== BiLC BATCH COMPLETE $(date) ===" >>$LOG
bash scripts/rca/audit_gates_2026-11.sh
