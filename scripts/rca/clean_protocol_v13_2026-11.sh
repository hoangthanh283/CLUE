#!/usr/bin/env bash
# Clean-protocol queue v13 (2026-11-14, Dev, plan rippling-pondering-comet s.1): CoLaR-SBR rank sweep
# first, then every v11 job in v11's order (run() skips finished dirs), then v12's DER++-250 block.
#  R1. CoLaR-SBR cil_cord rank 32: 1-epoch smoke, then x3 (dirs cil_cord_colar_sbr_r32_seed<s>).
#  R2. rank 64 x3 (colar_sbr_r64) ONLY IF some seed's r32 full_label_f1 is > 2 points below the
#      rank-128 run of the same seed (cil_cord_colar_sbr_seed<s>); a missing r32 value counts as worse.
#  v12 (pid 2053361) waits for the v11 pid and then takes the lock with flock -n, so once v13 holds
#  the lock it logs REFUSED and exits; its DER++ jobs (same names/overrides) are appended here.
# Helpers and variables below are v11's verbatim; LR32/LR64/LR1_32 differ from LR/LR1 only in rank_r.
#   setsid nohup bash scripts/rca/clean_protocol_v13_2026-11.sh [pid-to-wait-for] [its-run-name] &
set -u
cd "$(dirname "$0")/../.."
WAIT=${1:-}; ORPHAN=${2:-}; P=results/clean_2026-11; Q=$P/queue.log; mkdir -p $P
# ponytail: lock held by the driver only (children close fd 9), so killing the driver frees it.
exec 9>$P/.driver.lock
flock -n 9 || { echo "$(date '+%m-%d %H:%M') REFUSED another driver holds $P/.driver.lock (pid $$)" >>$Q; exit 1; }
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 60 9>&-; done; fi
if [ -n "$ORPHAN" ]; then
  if ls $P/${ORPHAN}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $ORPHAN" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $ORPHAN" >>$Q; fi
fi
T="training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 training.amp=false wandb.mode=offline output_dir=$P +training.cil_restrict_eval=true +training.val_frac=0.1 +training.full_label_eval=true +training.full_label_dump=true"
LR="method.epochs=10 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
RECB="method=colar_sbr_recall method.lora_rank=64 method.lora_alpha=128 method.target_modules=[query,key,value,dense]"
REC="$RECB +method.run_alias=colar_sbr_recall_r64 method.docs_per_task=0 $LR"
RECC="$RECB +method.run_alias=colar_sbr_recall_r64c method.docs_per_task=50 $LR"  # with the 50-doc memory
LR1="method.epochs=1 method.split_layer_k=4 method.rank_r=128 method.replay_batch_size=4"
run() {  # run <name> <args...>
  local NAME=$1; shift
  ls $P/${NAME}*/metrics.json >/dev/null 2>&1 && return 0
  echo "$(date '+%m-%d %H:%M') START $NAME" >>$Q
  uv run python scripts/train.py "$@" $T >"$P/$NAME.log" 2>&1 9>&-
  if ls $P/${NAME}*/metrics.json >/dev/null 2>&1; then echo "$(date '+%m-%d %H:%M') DONE  $NAME" >>$Q
  else echo "$(date '+%m-%d %H:%M') FAIL  $NAME" >>$Q; fi
}
smoke_ok() {  # smoke_ok <run-name>...: every smoke wrote full_label_f1
  local n; for n in "$@"; do grep -qs full_label_f1 $P/$n/metrics.json || return 1; done
}
L="model=layoutlmv3_base"; S="42 7 123"
ER250="method=er +method.run_alias=er250 method.buffer_size=250 method.epochs=10 scenario=cil_cord"
LR32="method.epochs=10 method.split_layer_k=4 method.rank_r=32 method.replay_batch_size=4"
LR64="method.epochs=10 method.split_layer_k=4 method.rank_r=64 method.replay_batch_size=4"
LR1_32="method.epochs=1 method.split_layer_k=4 method.rank_r=32 method.replay_batch_size=4"
r32_worse() {  # exit 0 if any seed's r32 full_label_f1 is > 2 points below rank 128 (or missing)
  uv run python - "$P" $S <<'PY'
import json, sys
p, seeds = sys.argv[1], sys.argv[2:]
def f1(d):
    try: return float(json.load(open(f"{p}/{d}/metrics.json"))["full_label_f1"])
    except Exception: return None
worse = False
for s in seeds:
    a, b = f1(f"cil_cord_colar_sbr_r32_seed{s}"), f1(f"cil_cord_colar_sbr_seed{s}")
    print(f"seed {s}: r32={a} r128={b}", file=sys.stderr)
    worse |= a is None or b is None or b - a > 2.0
sys.exit(0 if worse else 1)
PY
}
echo "$(date '+%m-%d %H:%M') v13 driver running (pid $$)" >>$Q
# R1. rank 32, smoke first
run cil_cord_smoke_colar_sbr_r32_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_r32_e1 method.docs_per_task=50 $LR1_32 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_sbr_r32_e1_seed42; then
  for s in $S; do run cil_cord_colar_sbr_r32_seed$s method=colar_sbr +method.run_alias=colar_sbr_r32 method.docs_per_task=50 $LR32 scenario=cil_cord seed=$s $L; done
  # R2. rank 64 only if rank 32 loses > 2 full-label-F1 points on some seed
  if r32_worse 2>>$Q; then
    echo "$(date '+%m-%d %H:%M') NOTE  r32 > 2 pts below r128 on some seed (or missing): running r64" >>$Q
    for s in $S; do run cil_cord_colar_sbr_r64_seed$s method=colar_sbr +method.run_alias=colar_sbr_r64 method.docs_per_task=50 $LR64 scenario=cil_cord seed=$s $L; done
  else echo "$(date '+%m-%d %H:%M') SKIP  r64 block (r32 within 2 pts of r128 on every seed)" >>$Q; fi
else echo "$(date '+%m-%d %H:%M') SKIP  rank-32 block (smoke has no full_label_f1)" >>$Q; fi
# v11 job list, verbatim from here (finished cells are skipped)
# 0. finished cells (skipped; re-run only if a metrics.json is missing)
for s in $S; do run cil_cord_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_recall_r64_seed$s $REC scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_sbr_noreplay_seed$s method=colar_sbr +method.run_alias=colar_sbr_noreplay method.docs_per_task=0 $LR scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# 1. CIL-CORD references
for s in $S; do run cil_cord_joint_seed$s method=joint method.epochs=100 scenario=cil_cord seed=$s $L; done
for s in $S; do run cil_cord_naive_seed$s method=naive method.epochs=10 scenario=cil_cord seed=$s $L; done
# 2. true-label memory refresh (oracle), recipe of results/gates/cil_protocol_2026-11/cil_cord_colar_r_oracle_seed42_k4_d50_r128
for s in $S; do run cil_cord_colar_r_oracle_seed$s method=colar_r_oracle method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
# 2b. D16c mechanism dumps (seed 42, distinct aliases; not table cells)
run cil_cord_colar_dump_seed42 method=colar +method.run_alias=colar_dump method.docs_per_task=50 $LR scenario=cil_cord seed=42 $L
run cil_cord_naive_dump_seed42 method=naive +method.run_alias=naive_dump method.epochs=10 scenario=cil_cord seed=42 $L
run cil_cord_joint_dump_seed42 method=joint +method.run_alias=joint_dump method.epochs=100 scenario=cil_cord seed=42 $L
# 2c. D17: CoLaR-SBR dump re-run (seed 42, distinct alias; not a table cell)
run cil_cord_colar_sbr_dump_seed42 method=colar_sbr +method.run_alias=colar_sbr_dump method.docs_per_task=50 $LR scenario=cil_cord seed=42 $L
# 2d. D17: true-label memory refreshed before each session (run_alias set by the config), smoke first
run cil_cord_smoke_colar_r_oracle_pre_e1_seed42 method=colar_r_oracle_pre method.run_alias=smoke_colar_r_oracle_pre_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_r_oracle_pre_e1_seed42_k4_d50_r128; then
  for s in $S; do run cil_cord_colar_r_oracle_pre_seed$s method=colar_r_oracle_pre method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  colar_r_oracle_pre block (smoke has no full_label_f1)" >>$Q; fi
# 3. MiB-style unbiased CE on the CoLaR memory, shared softmax (no MiB distillation/init)
run cil_cord_smoke_colar_mib_e1_seed42 method=colar_mib +method.run_alias=smoke_colar_mib_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_mib_e1_seed42; then
  for s in $S; do run cil_cord_colar_mib_seed$s method=colar_mib method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  colar_mib block (smoke has no full_label_f1)" >>$Q; fi
# 3b. D18: faithful ICE-O (Liu & Huang 2023) on LayoutLMv3, smoke first
run cil_cord_smoke_ice_o_e1_seed42 method=ice_o +method.run_alias=smoke_ice_o_e1 method.epochs=1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_ice_o_e1_seed42; then
  for s in $S; do run cil_cord_ice_o_seed$s method=ice_o method.epochs=10 scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  ice_o block (smoke has no full_label_f1)" >>$Q; fi
# 3c. D18: CoLaR-SBR with a constant (0) background logit per block, smoke first
run cil_cord_smoke_colar_sbr_const_o_e1_seed42 method=colar_sbr_const_o method.run_alias=smoke_colar_sbr_const_o_e1 method.docs_per_task=50 $LR1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_colar_sbr_const_o_e1_seed42; then
  for s in $S; do run cil_cord_colar_sbr_const_o_seed$s method=colar_sbr_const_o method.docs_per_task=50 $LR scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  colar_sbr_const_o block (smoke has no full_label_f1)" >>$Q; fi
# 4. E1: raw-document memory x blocked readout (er_sbr), smoke first
run cil_cord_smoke_er_sbr_e1_seed42 method=er_sbr +method.run_alias=smoke_er_sbr_e1 method.epochs=1 scenario=cil_cord seed=42 $L
if smoke_ok cil_cord_smoke_er_sbr_e1_seed42; then
  for s in $S; do run cil_cord_er_sbr_seed$s method=er_sbr method.docs_per_task=50 method.epochs=10 method.replay_batch_size=4 scenario=cil_cord seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  E1 er_sbr block (smoke has no full_label_f1)" >>$Q; fi
# 5. Mixed headline
for s in $S; do run mixed_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=mixed seed=$s $L; done
for s in $S; do run mixed_joint_seed$s method=joint method.epochs=100 scenario=mixed seed=$s $L; done
# 6. E2 smokes: joint pool, SBR (10-doc memory) and Recall on cil_funsd, 1 epoch, seed 42
FS="scenario=cil_funsd seed=42 $L"
run cil_funsd_smoke_joint_e1_seed42 method=joint +method.run_alias=smoke_joint_e1 method.epochs=1 $FS
run cil_funsd_smoke_colar_sbr_e1_seed42 method=colar_sbr +method.run_alias=smoke_colar_sbr_e1 method.docs_per_task=10 $LR1 $FS
run cil_funsd_smoke_recall_e1_seed42 $RECB +method.run_alias=smoke_recall_e1 method.docs_per_task=0 $LR1 $FS
# 6b. E2: cil_funsd, 10 docs/session for CoLaR and CoLaR-SBR (colar dirs get _k4_d10_r128)
if smoke_ok cil_funsd_smoke_{joint,colar_sbr,recall}_e1_seed42; then
  F="scenario=cil_funsd $L"
  for s in $S; do run cil_funsd_colar_sbr_d10_seed$s method=colar_sbr +method.run_alias=colar_sbr_d10 method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_colar_sbr_recall_r64_seed$s $REC $F seed=$s; done
  for s in $S; do run cil_funsd_colar_seed${s}_k4_d10 method=colar method.docs_per_task=10 $LR $F seed=$s; done
  for s in $S; do run cil_funsd_joint_seed$s method=joint method.epochs=100 $F seed=$s; done
  for s in $S; do run cil_funsd_naive_seed$s method=naive method.epochs=10 $F seed=$s; done
else echo "$(date '+%m-%d %H:%M') SKIP  E2 cil_funsd block (a smoke has no full_label_f1)" >>$Q; fi
# 7. second backbone for the method claim (run dir gets a _bros suffix; NAME stays a prefix)
for s in $S; do run cil_cord_colar_sbr_bros_seed$s method=colar_sbr +method.run_alias=colar_sbr_bros method.docs_per_task=50 $LR scenario=cil_cord seed=$s model=bros_base; done
# 8. DIL contrast (ER vs joint) under the clean protocol
for s in $S; do run dil_joint_seed$s method=joint method.epochs=100 scenario=dil seed=$s $L; done
for s in $S; do run dil_er_seed$s method=er method.epochs=10 scenario=dil seed=$s $L; done
# 10. Recall session-order check
for o in o2 o3; do run cil_cord_${o}_colar_sbr_recall_r64_seed42 $REC scenario=cil_cord_$o seed=42 $L; done
# 9. CL-LoRA (one shared adapter, shared softmax)
for s in $S; do run cil_cord_cl_lora_seed$s method=cl_lora method.epochs=10 scenario=cil_cord seed=$s $L; done
# 11. ER-250
for s in $S; do run cil_cord_er250_seed$s $ER250 seed=$s $L; done
# 12. LwF in fp32 (fp16 collapses it to 0)
for s in $S; do run cil_cord_lwf_seed$s method=lwf method.epochs=10 scenario=cil_cord seed=$s $L; done
# 13. rest of DIL
for s in $S; do run dil_colar_seed$s method=colar method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_colar_sbr_seed$s method=colar_sbr method.docs_per_task=50 $LR scenario=dil seed=$s $L; done
for s in $S; do run dil_naive_seed$s method=naive method.epochs=10 scenario=dil seed=$s $L; done
# v12: DER++ on CIL-CORD, 250-document budget (v12's names and overrides)
D="method=der_pp method.buffer_size=250 scenario=cil_cord"
run cil_cord_smoke_der_pp_e1_seed42 $D +method.run_alias=smoke_der_pp_e1 method.epochs=1 seed=42 $L
if grep -qs full_label_f1 $P/cil_cord_smoke_der_pp_e1_seed42*/metrics.json; then
  for s in 42 7 123; do run cil_cord_der_pp250_seed$s $D +method.run_alias=der_pp250 method.epochs=10 seed=$s $L; done
else echo "$(date '+%m-%d %H:%M') SKIP  der_pp block (smoke has no full_label_f1)" >>$Q; fi
echo "$(date '+%m-%d %H:%M') ALL DONE v13" >>$Q
