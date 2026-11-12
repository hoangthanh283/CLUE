#!/usr/bin/env bash
# Follow-on queue (2026-11-12, user: "get the remaining ones started or put them in the queue").
# Starts only when no clean_protocol driver and no train.py is alive, then holds the clean-queue lock
# (results/clean_2026-11/.driver.lock) so a newly launched clean driver refuses instead of colliding.
#  A. image gates G1/G2 + CIFAR val-split (all three OOMed on Nov 10 from the DER++ s123 collision)
#  B. legacy rescore, remaining seed-123 rows (CIL-CORD + Mixed; resume-safe, done rows SKIP)
#  C. backbone-grid rescore BERT/BROS/LiLT (seed-outer; the Nov 10 'OOM' rows were a collision)
# G3's DIL CoLaR/ER val-split runs are superseded by the clean-protocol DIL cells (val_frac=0.1).
#   setsid nohup bash scripts/rca/after_clean_2026-11.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=results/gates/after_clean_2026-11.log
busy() { pgrep -f "clean_protocol_v[0-9]+_2026-11.sh" >/dev/null || pgrep -f "python scripts/train.py" >/dev/null; }
echo "$(date '+%m-%d %H:%M') WAIT  driver $$ for clean queue + GPU idle" >>$Q
exec 9>results/clean_2026-11/.driver.lock
while :; do
  while busy; do sleep 600; done
  sleep 120; busy && continue                # survive a driver swap gap
  flock -n 9 && ! busy && break
  flock -u 9; sleep 600
done
echo "$(date '+%m-%d %H:%M') GO    lock held" >>$Q

OUT=results/gates/audit_2026-11
VIS="model=vit_b16 training=vision training.batch_size=16 training.num_workers=1 training.gradient_checkpointing=false wandb.mode=offline output_dir=$OUT"
run() { local name=$1; shift
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP  $name" >>$Q; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$Q
  "$@" >"$OUT/$name.log" 2>&1 9>&-
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$Q; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$Q; fi
}
# A
run cil_cifar100_coda_masked_seed42_vit uv run python scripts/train.py method=coda_prompt +method.run_alias=coda_masked \
  +method.mask_old_logits=true method.epochs=100 scenario=cil_cifar100 seed=42 $VIS
run cil_cifar100_cl_lora_masked_seed42_vit uv run python scripts/train.py method=cl_lora +method.run_alias=cl_lora_masked \
  +method.mask_old_logits=true method.epochs=100 scenario=cil_cifar100 seed=42 $VIS
run cil_cifar100_aglr_fit500_seed42_vit uv run python scripts/train.py method=aglr_replay +method.run_alias=aglr_fit500 \
  +method.fit_docs_per_task=500 method.split_layer_k=8 method.carriers_per_task=500 method.attn_keep=1.0 method.epochs=100 \
  scenario=cil_cifar100 seed=42 $VIS
run cil_cifar100_colar500r128_val_seed42_vit uv run python scripts/train.py scenario=cil_cifar100 seed=42 method=colar_vit \
  method.rank_r=128 +method.run_alias=colar500r128_val method.epochs=100 method.split_layer_k=4 training.amp=true +training.val_frac=0.1 $VIS
# B, C (same flags as audit_gates stages R and RB)
R="GPUS=0 LANES=1 BATCH_SIZE=2 GRAD_CKPT=true AMP=true NUM_WORKERS=0 OUT=results/rescore"
echo "$(date '+%m-%d %H:%M') START rescore seed-123 remainder" >>$Q
env $R ROWS=main MODELS=layoutlmv3_base SEEDS=123 bash scripts/rescore_cil_2026-11.sh 9>&-
echo "$(date '+%m-%d %H:%M') START backbone-grid rescore" >>$Q
env $R ROWS=grid MODELS="bert_base bros_base lilt_base" bash scripts/rescore_cil_2026-11.sh 9>&-
echo "$(date '+%m-%d %H:%M') ALL DONE" >>$Q
