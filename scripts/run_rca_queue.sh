#!/bin/bash
# Amendment 7: Phase R (frozen-PTM references, probe run) then Phase S screens. Sequential.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/logs_vision/rca_queue.log; mkdir -p results/logs_vision
V="model=vit_b16 training=vision training.batch_size=16 training.num_workers=4 training.gradient_checkpointing=false training.amp=true"
run() {  # run <run-dir> <args...>
  local name="$1"; shift
  if [ -f "results/${name}/metrics.json" ]; then echo "$(date +%H:%M) SKIP $name" >>"$LOG"; return; fi
  echo "$(date +%H:%M) START $name" >>"$LOG"
  uv run python scripts/train.py "$@" $V > "results/logs_vision/${name}.log" 2>&1
  [ -f "results/${name}/metrics.json" ] && echo "$(date +%H:%M) DONE  $name" >>"$LOG" || echo "$(date +%H:%M) FAIL  $name" >>"$LOG"
}
echo "=== RCA QUEUE $(date) ===" >>"$LOG"
# R1 frozen-PTM references
for sc in cil_cifar100 cil_imagenet_r; do
  run "${sc}_simplecil_seed42_vit" method=simplecil scenario=$sc seed=42
  for sd in 42 7 123; do run "${sc}_ranpac_seed${sd}_vit" method=ranpac scenario=$sc seed=$sd method.rp_seed=$sd; done
done
# R2 probe run + probes
PP="method=colar_pp scenario=cil_cifar100 seed=42"
run cil_cifar100_rcaBank_seed42_vit $PP +method.run_alias=rcaBank method.dump_final=true
uv run python scripts/rca_colar_probe.py --run results/cil_cifar100_rcaBank_seed42_vit --k 4 \
  > results/logs_vision/rca_probe_bank.txt 2>&1 && echo "$(date +%H:%M) PROBES done" >>"$LOG" || echo "$(date +%H:%M) PROBES FAIL" >>"$LOG"
# S screens
run cil_cifar100_ppS1_seed42_vit $PP +method.run_alias=ppS1 method.analytic_head=rp
run cil_cifar100_ppS2_seed42_vit $PP +method.run_alias=ppS2 method.feature_anchor=1.0
run cil_cifar100_ppS3_seed42_vit $PP +method.run_alias=ppS3 method.analytic_head=rp method.feature_anchor=1.0
run cil_cifar100_ppS4_seed42_vit $PP +method.run_alias=ppS4 method.analytic_head=rp method.split_layer_k=12
echo "=== RCA QUEUE COMPLETE $(date) ===" >>"$LOG"
