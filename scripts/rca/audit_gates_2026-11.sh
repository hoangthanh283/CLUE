#!/usr/bin/env bash
# Audit gates (2026-11-01, plan pure-zooming-jellyfish): one GPU, sequential, seed 42.
#  G1 CODA-Prompt / CL-LoRA on CIFAR-100 WITH old-class logit masking (official protocol)
#  G2 coverage control: AGLR Gaussians fit on 500 images/task (= the CoLaR bank budget)
#  G3 val-split delta: CoLaR CIFAR + DIL CoLaR/ER early-stopped on a 10 % train split
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/gates/audit_2026-11; LOG=$OUT/queue.log; mkdir -p $OUT
VIS="model=vit_b16 training=vision training.batch_size=16 training.num_workers=1 training.gradient_checkpointing=false wandb.mode=offline output_dir=$OUT"
DOC="training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$OUT"
run() { local name=$1; shift
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$LOG; return; fi
  echo "$(date '+%m-%d %H:%M') START $name" >>$LOG
  "$@" >"$OUT/$name.log" 2>&1
  if ls $OUT/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$LOG; else echo "$(date '+%m-%d %H:%M') FAIL  $name rc=$?" >>$LOG; fi
}
echo "=== AUDIT GATES START $(date) ===" >>$LOG
# S1 (2026-11-02, TOP PRIORITY): CoLaR-SBR session-blocked readout, CIL-CORD seed 42 + controls
P=results/gates/cil_protocol_2026-11
SBR_COMMON="scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
sbr_run() { local name=$1; shift
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') SKIP $name" >>$P/queue.log; return; fi
  echo "$(date '+%m-%d %H:%M') START $name (S1)" >>$P/queue.log
  uv run python scripts/train.py method=colar_sbr "$@" $SBR_COMMON >"$P/$name.log" 2>&1
  if ls $P/${name}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $name" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  $name" >>$P/queue.log; fi
}
sbr_run cil_cord_colar_sbr_seed42          method.docs_per_task=50
sbr_run cil_cord_colar_sbr_noreplay_seed42 method.docs_per_task=0 +method.run_alias=colar_sbr_noreplay
sbr_run cil_cord_colar_sbr_frozen_seed42   method.docs_per_task=0 method.split_layer_k=11 +method.run_alias=colar_sbr_frozen  # layers <11 frozen (ICE-like)
# M2 (2026-11-03): Mixed SBR with a patient stopper + lower entity threshold (SROIE block under-trained:
# val 57->50->53, recall 45-49, stopped at ep3)
if ! ls $P/mixed_colar_sbr_p4t3_seed42*/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START mixed_colar_sbr_p4t3_seed42 (M2)" >>$P/queue.log
  uv run python scripts/train.py method=colar_sbr +method.run_alias=colar_sbr_p4t3 method.sbr_threshold=0.3 +method.early_stop_patience=4 method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=mixed model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/mixed_colar_sbr_p4t3_seed42.log" 2>&1
  if ls $P/mixed_colar_sbr_p4t3_seed42*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  mixed_colar_sbr_p4t3_seed42" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  mixed_colar_sbr_p4t3_seed42" >>$P/queue.log; fi
fi
# E3 (2026-11-02): ER-ACE-asymmetric CoLaR-R (current_sliced: all) on CIL-CORD seed 42 + seed 123
P=results/gates/cil_protocol_2026-11
for SD in 42 123; do
if ! ls $P/cil_cord_colar_r_sliced3_seed${SD}*/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START cil_cord_colar_r_sliced3_seed${SD} (E3)" >>$P/queue.log
  uv run python scripts/train.py method=colar_r_sliced3 method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=cil_cord model=layoutlmv3_base seed=$SD training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/cil_cord_colar_r_sliced3_seed${SD}.log" 2>&1
  if ls $P/cil_cord_colar_r_sliced3_seed${SD}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  cil_cord_colar_r_sliced3_seed${SD}" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  cil_cord_colar_r_sliced3_seed${SD}" >>$P/queue.log; fi
fi
done
# G0b: Mixed joint under the corrected scorer (the follow-up batch's entry passes CoLaR keys and fails)
P=results/gates/cil_protocol_2026-11
if ! ls $P/mixed_joint_seed42/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START mixed_joint_seed42 (G0b)" >>$P/queue.log
  uv run python scripts/train.py method=joint method.epochs=100 scenario=mixed model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/mixed_joint_seed42.log" 2>&1
  if ls $P/mixed_joint_seed42/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  mixed_joint_seed42" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  mixed_joint_seed42" >>$P/queue.log; fi
fi
# G0c: stabiliser probe — double-sliced seed 123 at replay:current 1:1 (replay_batch 2) 
if ! ls $P/cil_cord_colar_r_sliced2_rb2_seed123*/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START cil_cord_colar_r_sliced2_rb2_seed123 (G0c)" >>$P/queue.log
  uv run python scripts/train.py method=colar_r_sliced2 method.run_alias=colar_r_sliced2_rb2 method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=2 scenario=cil_cord model=layoutlmv3_base seed=123 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/cil_cord_colar_r_sliced2_rb2_seed123.log" 2>&1
  if ls $P/cil_cord_colar_r_sliced2_rb2_seed123*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  cil_cord_colar_r_sliced2_rb2_seed123" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  cil_cord_colar_r_sliced2_rb2_seed123" >>$P/queue.log; fi
fi
# G0d: ceiling rerun with replay_sliced (the first ceiling run used the plain replay CE)
if ! ls $P/cil_cord_colar_r_sliced2_oracle_seed42*/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START cil_cord_colar_r_sliced2_oracle_seed42 (G0d)" >>$P/queue.log
  uv run python scripts/train.py method=colar_r_sliced2_oracle method.epochs=10 method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true >"$P/cil_cord_colar_r_sliced2_oracle_seed42.log" 2>&1
  if ls $P/cil_cord_colar_r_sliced2_oracle_seed42*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  cil_cord_colar_r_sliced2_oracle_seed42" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  cil_cord_colar_r_sliced2_oracle_seed42" >>$P/queue.log; fi
fi
# G0 (2026-11-02, highest priority): CoLaR-R double-sliced on CIL-CORD under the corrected scorer
P=results/gates/cil_protocol_2026-11
PDOC="scenario=cil_cord model=layoutlmv3_base seed=42 training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true"
PAX="method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4"
if ! ls $P/cil_cord_colar_r_sliced2_seed42*/metrics.json >/dev/null 2>&1; then
  echo "$(date '+%m-%d %H:%M') START cil_cord_colar_r_sliced2_seed42" >>$P/queue.log
  uv run python scripts/train.py method=colar_r_sliced2 method.epochs=10 $PAX $PDOC >"$P/cil_cord_colar_r_sliced2_seed42.log" 2>&1
  if ls $P/cil_cord_colar_r_sliced2_seed42*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  cil_cord_colar_r_sliced2_seed42" >>$P/queue.log; else echo "$(date '+%m-%d %H:%M') FAIL  cil_cord_colar_r_sliced2_seed42" >>$P/queue.log; fi
fi
# G1
run cil_cifar100_coda_masked_seed42_vit uv run python scripts/train.py method=coda_prompt +method.run_alias=coda_masked \
  +method.mask_old_logits=true method.epochs=100 scenario=cil_cifar100 seed=42 $VIS
run cil_cifar100_cl_lora_masked_seed42_vit uv run python scripts/train.py method=cl_lora +method.run_alias=cl_lora_masked \
  +method.mask_old_logits=true method.epochs=100 scenario=cil_cifar100 seed=42 $VIS
# G2
run cil_cifar100_aglr_fit500_seed42_vit uv run python scripts/train.py method=aglr_replay +method.run_alias=aglr_fit500 \
  +method.fit_docs_per_task=500 method.split_layer_k=8 method.carriers_per_task=500 method.attn_keep=1.0 method.epochs=100 \
  scenario=cil_cifar100 seed=42 $VIS
# G3
run dil_colar_e10_val_seed42 uv run python scripts/train.py method=colar +method.run_alias=colar_e10_val scenario=dil model=layoutlmv3_base \
  seed=42 method.epochs=10 training.batch_size=2 +training.val_frac=0.1 \
  method.split_layer_k=4 method.docs_per_task=50 method.rank_r=128 method.replay_batch_size=4 $DOC
run dil_er_val_seed42 uv run python scripts/train.py method=er +method.run_alias=er_val scenario=dil model=layoutlmv3_base \
  seed=42 method.epochs=100 training.batch_size=2 +training.val_frac=0.1 $DOC
run cil_cifar100_colar500r128_val_seed42_vit uv run python scripts/train.py scenario=cil_cifar100 seed=42 method=colar_vit \
  method.rank_r=128 +method.run_alias=colar500r128_val method.epochs=100 method.split_layer_k=4 training.amp=true +training.val_frac=0.1 $VIS
echo "=== AUDIT GATES COMPLETE $(date) ===" >>$LOG
