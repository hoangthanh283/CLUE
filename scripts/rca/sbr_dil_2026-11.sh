#!/usr/bin/env bash
# DIL SBR seed 42 (2026-11-03): waits for the last document gate (G0d ceiling) to finish, preempts the
# image gates, runs DIL SBR under the DIL CoLaR recipe, then resumes the gate script (resume-safe).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=results/gates/cil_protocol_2026-11; LOG=$P/queue.log
until grep -q "DONE  cil_cord_colar_r_sliced2_oracle_seed42\|FAIL  cil_cord_colar_r_sliced2_oracle_seed42" $LOG; do sleep 120; done
sleep 30
for pid in $(pgrep -f "audit_gates_2026-11.sh"); do kill $pid 2>/dev/null; done; sleep 2
for pid in $(pgrep -f "scenario=cil_cifar100\|scenario=cil_imagenet_r"); do kill $pid 2>/dev/null; done; sleep 5
echo "$(date '+%m-%d %H:%M') PREEMPT: image gates paused for DIL SBR" >>$LOG
run() { local name=$1 me=$2; shift 2
  local COMMON="scenario=dil model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  uv run python scripts/train.py method=$me "$@" $COMMON >"$P/$name.log" 2>&1
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$LOG; fi
}
run dil_colar_sbr_seed42 colar_sbr
echo "=== DIL SBR COMPLETE $(date) ===" >>$LOG
bash scripts/rca/audit_gates_2026-11.sh
