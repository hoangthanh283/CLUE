#!/usr/bin/env bash
# Multi-GPU parallel grid scheduler for a powerful box (e.g. 2x RTX 4090 24GB).
#
# Runs the COMPLETE experiment grid as fast as possible by keeping every GPU slot
# saturated: it builds the full job list (single-task baselines -> core -> prompt/LoRA
# -> DocCL + ablation), then dispatches jobs across N parallel slots, each pinned to a
# GPU via CUDA_VISIBLE_DEVICES. Resume-safe: a job whose results/<run>/.done exists is
# skipped, so re-running continues where it left off.
#
# Why slots, not in-process DDP: the grid is ~189 INDEPENDENT runs. One run per GPU
# (or several per GPU — the model is ~3 GB, a 4090 has 24 GB) gives near-linear
# speedup with zero code change to train.py, which already uses the single visible GPU.
#
# Tunables (env):
#   GPUS            space/comma GPU ids (default "0 1")
#   JOBS_PER_GPU    concurrent train.py per GPU (default 2 — model ~3GB, bs=16 ~ <10GB)
#   BATCH_SIZE      per-step batch (default 16; 4090 handles 32+ for non-replay methods)
#   EPOCHS_CAP      early-stop epoch ceiling (default 100)
#   SEEDS           default "42 123 7"
#   SCENARIOS       default "cil_cord dil mixed dil_xlingual cil_wildreceipt"
#   CORE_METHODS    default "naive joint ewc lwf er der_pp"
#   PROMPT_METHODS  default "l2p dualprompt coda_prompt o_lora"
#   CURRENCY_METHODS default "er_cflat cl_lora"  (2025 baselines, run AFTER DocCL)
#   RUN_BERT        default 1 (BERT text-only external comparator, classical tier)
#   RUN_DOCCL       default 1 (doccl across all scenarios)
#   RUN_LEXSLOT     default 1 (lexslot across all scenarios, primary backbone)
#   RUN_ABLATION    default 1 (doccl depth + lexslot slot_depth/sharing ablation on ABLATION_SCENARIOS)
#   ABLATION_SCENARIOS default "cil_cord"
#   DEPTH_TARGETS   default "head_only late_only uniform" ('all' = the main doccl run)
#   WANDB_MODE      default online
#   DRY_RUN         default 0 (print the job plan and exit)

set -uo pipefail
cd "$(dirname "$0")/.."
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; source .env 2>/dev/null || true; set +a
# Interpreter: honor $PYTHON (set by setup_remote.sh for pip-into-template-python), else
# 'python' (the uv .venv prepended above, or the system python).
PYBIN="${PYTHON:-python}"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN="python3"

GPUS="${GPUS:-0 1}"; GPUS="${GPUS//,/ }"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
# Optional VRAM-aware dispatch: when GPU_VRAM_BUDGET_GB>0, a job is dispatched only while the
# sum of running jobs' per-method estimates + the new job's estimate fits the budget (PER GPU),
# letting light methods (doccl ~5 GiB) pack 8-up while heavy ones (der_pp ~35) stay 1-2 — on the
# SAME card, no OOM. 0 (default) = pure slot scheduler (TOTAL_SLOTS workers), byte-identical to
# before. JOBS_PER_GPU still caps the hard concurrency ceiling per GPU. Estimates are measured
# peak_gpu_mem_mb maxima at bs16 (see metrics.json); override an entry via VRAM_EST_<method>=N.
GPU_VRAM_BUDGET_GB="${GPU_VRAM_BUDGET_GB:-0}"
declare -A VRAM_EST=(
  [der_pp]=35 [er]=27 [ewc]=20 [lwf]=20 [joint]=19 [cl_lora]=19 [o_lora]=18 [er_cflat]=19
  [l2p]=17 [dualprompt]=17 [naive]=11 [bert]=7 [doccl]=5 [coda_prompt]=4
)
# A job's estimate keyed off its run-name's method token (longest-match for *_cl_lora_*, etc.).
vram_est_for() {  # <run_name> -> GiB estimate (default 18 = conservative full-FT)
  local run="$1" m est=18
  for m in cl_lora o_lora er_cflat der_pp coda_prompt dualprompt l2p doccl joint ewc lwf er naive bert; do
    case "$run" in *_${m}_*) est="${VRAM_EST[$m]:-18}"; break ;; esac
  done
  local ov="VRAM_EST_${m}"; [ -n "${!ov:-}" ] && est="${!ov}"
  echo "$est"
}
# DataLoader workers PER JOB. Total worker procs = NUM_WORKERS x JOBS_PER_GPU x NGPU,
# and each worker forks a copy of the dataset working set. Heavy datasets (XFUND's 7
# languages, WildReceipt) can exhaust HOST RAM at the default x4 across all slots and
# trip the OOM killer (SIGKILL -> rc=137). Lower to 2 on RAM-constrained boxes.
NUM_WORKERS="${NUM_WORKERS:-4}"
# Gradient checkpointing: default off (big GPUs prefer the speed). Set GRAD_CKPT=true on
# VRAM-constrained boxes (e.g. only a few GB free) — it trades ~20-30%% compute for a large
# activation-memory saving, letting a job fit where it otherwise OOMs.
GRAD_CKPT="${GRAD_CKPT:-false}"
SEEDS="${SEEDS:-42 123 7}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
CORE_METHODS="${CORE_METHODS:-naive joint ewc lwf er der_pp}"
PROMPT_METHODS="${PROMPT_METHODS:-l2p dualprompt coda_prompt o_lora}"
# 2025 "currency" baselines — run AFTER DocCL (lowest priority tier).
CURRENCY_METHODS="${CURRENCY_METHODS:-er_cflat cl_lora}"
# BERT text-only external comparator (model=bert_base, naive method) — classical tier.
RUN_BERT="${RUN_BERT:-1}"
# Secondary-backbone GENERALIZATION study. The main grid runs LayoutLMv3 (the implicit
# default model). Every method is now backbone-agnostic (verified on LiLT/BROS/BERT — see
# clue-backbone-agnostic-audit), so the full method set runs on every family. BACKBONES =
# the families to add (config <family>_base must exist); BACKBONE_METHODS = which methods to
# run on each. Default is the full multi-backbone sweep (lilt bros bert); set BACKBONES=""
# for LayoutLMv3-only. Run-names get the train.py "_<family>" suffix so they never collide
# with the LayoutLMv3 run of the same scenario/method/seed. (BERT-naive is still covered by
# RUN_BERT above; 'bert' here adds the rest.)
BACKBONES="${BACKBONES-lilt bros bert}"
# Full method set per secondary backbone (all classical + prompt/LoRA + 2025 currency +
# proposed doccl + lexslot). Override to a subset to trade coverage for compute.
BACKBONE_METHODS="${BACKBONE_METHODS:-naive joint ewc lwf er der_pp l2p dualprompt coda_prompt o_lora cl_lora er_cflat doccl lexslot}"
RUN_DOCCL="${RUN_DOCCL:-1}"
RUN_LEXSLOT="${RUN_LEXSLOT:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
ABLATION_SCENARIOS="${ABLATION_SCENARIOS:-cil_cord}"
DEPTH_TARGETS="${DEPTH_TARGETS:-head_only late_only uniform}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"
# bf16 autocast: ON by default on this (big-GPU) path for ~1.5-2x speedup. Set AMP=0 for
# exact fp32. Exported as DOCCL_AMP so each train.py child enables it (CUDA-only; no-op on CPU).
AMP="${AMP:-1}"
export DOCCL_AMP="$AMP"

# Progress visibility
HEARTBEAT_SECS="${HEARTBEAT_SECS:-45}"              # 0 = disable the heartbeat block
PROGRESS_JSON="${PROGRESS_JSON:-results/logs/progress.json}"
TEE_TRAIN="${TEE_TRAIN:-1}"                         # 1 = mirror filtered train lines to the main log

# Durable resume for on-demand/ephemeral instances (all no-op when SYNC_REMOTE empty)
SYNC_REMOTE="${SYNC_REMOTE:-}"                      # e.g. "hf:datasets/<user>/doccl-results" — empty disables sync
SYNC_SECS="${SYNC_SECS:-300}"                       # background push interval
RCLONE_CONFIG_B64="${RCLONE_CONFIG_B64:-}"          # optional base64 rclone.conf injected at startup

LOG=results/logs/multigpu_grid.log
mkdir -p results/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# The tiny resume markers + TensorBoard event logs travel to/from durable storage.
# TB logs are small (~a few hundred KB/run, <~40 MB for the whole grid) but let the
# analysis be done LOCALLY after pulling from R2 (no need to keep a remote box alive).
SYNC_INCLUDES=(--include "*/.done" --include "*/metrics.json" --include "*/matrix.npy"
               --include "*/tb/**" --include "*/per_class_f1.json"
               --include "table_single_task_baselines.csv")

EXTRA="training.batch_size=${BATCH_SIZE} training.gradient_checkpointing=${GRAD_CKPT} \
training.num_workers=${NUM_WORKERS} method.epochs=${EPOCHS_CAP} wandb.project=${WANDB_PROJECT:-CL4IE}"

# ── Durable-resume sync (rclone; pure no-op unless sync is configured) ───────────
# SECURITY: credentials are NEVER written to disk. When R2_* env vars are present we
# configure the rclone 'obj' remote purely via RCLONE_CONFIG_<REMOTE>_<KEY> environment
# variables (rclone reads these in-process), so nothing sensitive lands on the rented
# instance's disk. Pass R2_* as runtime `-e` flags (NOT a persistent .env on the box).
R2_BUCKET="${R2_BUCKET:-doccl-results}"
maybe_build_r2_remote() {
  # Only when R2 creds are present AND the user hasn't already set SYNC_REMOTE explicitly.
  [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_ENDPOINT:-}" ] || return 0
  # rclone connection-via-env: defines an 'obj' s3 remote with no config file on disk.
  export RCLONE_CONFIG_OBJ_TYPE=s3
  export RCLONE_CONFIG_OBJ_PROVIDER=Cloudflare
  export RCLONE_CONFIG_OBJ_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
  export RCLONE_CONFIG_OBJ_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
  export RCLONE_CONFIG_OBJ_ENDPOINT="${R2_ENDPOINT}"
  export RCLONE_CONFIG_OBJ_REGION=auto
  [ -n "$SYNC_REMOTE" ] || SYNC_REMOTE="obj:${R2_BUCKET}/results"
  say "[sync] configured rclone 'obj' remote from R2_* env (in-process, no file on disk) -> ${SYNC_REMOTE}"
}

sync_setup() {
  maybe_build_r2_remote
  [ -n "$SYNC_REMOTE" ] || return 0
  if ! command -v rclone >/dev/null 2>&1; then
    say "[sync] rclone not installed — durable sync DISABLED"; SYNC_REMOTE=""; return 0
  fi
  if [ -n "$RCLONE_CONFIG_B64" ]; then
    # Optional fallback path: decode an injected rclone.conf to a TEMP file (cleaned up on
    # exit), not the persistent ~/.config — so a config-file user also leaves nothing behind.
    RCLONE_TMP_CONF=$(mktemp)
    echo "$RCLONE_CONFIG_B64" | base64 -d > "$RCLONE_TMP_CONF" 2>/dev/null \
      && { export RCLONE_CONFIG="$RCLONE_TMP_CONF"; chmod 600 "$RCLONE_TMP_CONF"; say "[sync] using injected rclone.conf (temp, auto-deleted on exit)"; } \
      || say "[sync] WARN: could not decode RCLONE_CONFIG_B64"
  fi
  # Cred-shape self-check FIRST: a blind paste into the secret prompt can silently drop
  # chars (read -s echoes nothing). R2 tokens are 32-char key / 64-char secret; warn if
  # the captured creds don't match so a paste typo is obvious before rclone even runs.
  local kl="${#R2_ACCESS_KEY_ID}" sl="${#R2_SECRET_ACCESS_KEY}"
  say "[sync] cred lengths: key=${kl} (expect 32), secret=${sl} (expect 64)"
  if [ "$kl" != "32" ] || [ "$sl" != "64" ]; then
    say "[sync] WARN: cred length mismatch — likely a mistyped/mis-pasted credential."
    say "[sync] Fix: export R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY in the env and relaunch (avoids the blind prompt)."
  fi
  # Fail-fast preflight: prove we can write+list+delete on the remote BEFORE running 189
  # ungated jobs. Run with -vv so the REAL error (403/Signature/skew/no-host) always prints.
  local tmp; tmp=$(mktemp -d)
  echo "ok $(date +%s)" > "$tmp/.synccheck"
  # The WRITE's exit code is the authoritative signal: rclone returns non-zero on a real
  # auth/network/permission failure and 0 on a real upload. We do NOT gate on an immediate
  # post-write `lsf` — S3/R2 is read-after-write *eventually* consistent, so a sub-second
  # list lag (seen on a fresh R2 region) made a SUCCESSFUL write (Transferred 1/1, rc=0)
  # falsely report PREFLIGHT FAILED. Trust rc; only flag failure on rc!=0 or an error line.
  local pf_err pf_rc
  pf_err=$(rclone copy "$tmp" "$SYNC_REMOTE" --include ".synccheck" -vv 2>&1); pf_rc=$?
  echo "$pf_err" >> "$LOG"
  if [ "$pf_rc" -eq 0 ] \
     && ! echo "$pf_err" | grep -qiE "fatal|forbidden|denied|signature|no such host|refused|AccessDenied|InvalidAccessKey"; then
    rclone delete "$SYNC_REMOTE/.synccheck" >> "$LOG" 2>&1 || true
    rm -rf "$tmp"
    say "[sync] preflight OK (write rc=0) — durable resume ON -> ${SYNC_REMOTE}"
  else
    rm -rf "$tmp"
    say "[sync] PREFLIGHT FAILED (rclone rc=${pf_rc}) on ${SYNC_REMOTE} — check bucket/token/endpoint."
    # Surface rclone's actual error on-screen — the real root cause (prefer error/fatal lines).
    local shown; shown=$(echo "$pf_err" | grep -iE "error|fail|denied|forbidden|signature|skew|no such|timeout|refused|403|400|401" | tail -3)
    [ -z "$shown" ] && shown=$(echo "$pf_err" | grep -v "^$" | tail -3)
    [ -n "$shown" ] && echo "$shown" | sed 's/^/  [rclone] /'
    say "[sync] hint: R2 token must have Object Read & Write on '${R2_BUCKET}'; key/secret must be 32/64 chars."
    if [ "${SYNC_STRICT:-1}" = "1" ]; then
      say "[sync] SYNC_STRICT=1 (default): aborting so you don't run un-persisted. Set SYNC_STRICT=0 to run anyway."
      exit 3
    fi
    say "[sync] SYNC_STRICT=0 — continuing WITHOUT durable sync (results live on the ephemeral disk only)."
    SYNC_REMOTE=""
  fi
}
sync_pull() {
  [ -n "$SYNC_REMOTE" ] || return 0
  say "[sync] pulling prior resume state from ${SYNC_REMOTE} ..."
  rclone copy "$SYNC_REMOTE" results/ "${SYNC_INCLUDES[@]}" >> "$LOG" 2>&1 \
    || say "[sync] pull nonzero (treated as fresh start, continuing)"
  # Partial-write safety: drop any metrics/matrix in a dir lacking .done (never read a half-write).
  for d in results/*/; do
    [ -f "${d}.done" ] && continue
    rm -f "${d}metrics.json" "${d}matrix.npy" 2>/dev/null || true
  done
}
sync_push_once() {
  [ -n "$SYNC_REMOTE" ] || return 0
  rclone copy results/ "$SYNC_REMOTE" "${SYNC_INCLUDES[@]}" >> "$LOG" 2>&1 || true
}
sync_push_loop() { while :; do sleep "$SYNC_SECS"; sync_push_once; done; }

# ── Heartbeat: filesystem-only live progress block (no dependency on slot arrays) ─
GRID_START_TS=$(date +%s)
# Unique per-launch marker written into the (persistent, appended) LOG. The heartbeat
# counts [FAIL lines only AFTER this marker, so the "failed" tally reflects THIS run's
# real failures — not every failure ever printed across relaunches (which made the
# count monotonically climb and never drop, even after the underlying bug was fixed).
SESSION_MARKER="=== GRID SESSION START ${GRID_START_TS} ==="
write_progress_json() {  # <total> <done> <failed>
  local tmp="${PROGRESS_JSON}.tmp" first=1 f rn
  {
    printf '{"ts":%s,"total":%s,"done":%s,"failed":%s,"running":[' "$(date +%s)" "$1" "$2" "$3"
    for f in results/logs/*.log; do
      [ -f "$f" ] || continue
      rn=$(basename "$f" .log)
      [ "$rn" = "multigpu_grid" ] && continue
      [ -f "results/${rn}/.done" ] && continue
      if [ -n "$(find "$f" -newermt "-$(( 2*HEARTBEAT_SECS )) seconds" 2>/dev/null)" ]; then
        [ $first -eq 0 ] && printf ','; first=0; printf '"%s"' "$rn"
      fi
    done
    printf ']}'
  } > "$tmp" 2>/dev/null && mv -f "$tmp" "$PROGRESS_JSON" 2>/dev/null || true
}
heartbeat_loop() {  # <total>
  local total="$1"
  while :; do
    sleep "$HEARTBEAT_SECS"
    local done_ct fail_ct elapsed eta="n/a" f rn last
    # Count .done ONLY for THIS launch's jobs (HB_MANIFEST), so done_ct can never exceed
    # total even when results/ holds other partitions'/stale .done markers (shared R2 bucket).
    if [ -f "$HB_MANIFEST" ]; then
      done_ct=0
      while IFS= read -r rn; do
        [ -n "$rn" ] && [ -f "results/${rn}/.done" ] && done_ct=$((done_ct+1))
      done < "$HB_MANIFEST"
    else
      done_ct=$(ls results/*/.done 2>/dev/null | wc -l)   # fallback (manifest absent)
    fi
    # Disk guard: online W&B buffers every run to wandb/, and on a small-disk rented box
    # (e.g. a 30 GB vast.ai volume) that fills the disk and WEDGES Docker ("No space left on
    # device" -> the container can't even restart). When free space on / drops below
    # DISK_GUARD_GB (default 5), prune the W&B run dirs (results live in results/<run>/ and
    # R2, never in wandb/) and warn loudly. Set DISK_GUARD_GB=0 to disable.
    local guard="${DISK_GUARD_GB:-5}" free_gb
    if [ "$guard" != "0" ]; then
      free_gb=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
      if [ -n "$free_gb" ] && [ "$free_gb" -lt "$guard" ]; then
        rm -rf wandb/run-* wandb/offline-run-* wandb/latest-run 2>/dev/null
        echo "[$(date '+%m-%d %H:%M:%S')] [disk] FREE ${free_gb}G < ${guard}G — pruned wandb/ run dirs (results unaffected; set WANDB_MODE=offline to avoid the bloat)" | tee -a "$LOG" 2>/dev/null || true
      fi
    fi
    # Count [FAIL only AFTER this launch's session marker (not stale failures from
    # prior relaunches appended to the same persistent log).
    fail_ct=$(awk -v m="$SESSION_MARKER" 'index($0,m){c=0;seen=1;next} seen&&/\[FAIL/{c++} END{print c+0}' "$LOG" 2>/dev/null || echo 0)
    elapsed=$(( $(date +%s) - GRID_START_TS ))
    if [ "$done_ct" -gt 0 ]; then
      local per=$(( elapsed / done_ct )) remain
      remain=$(( (total - done_ct) * per )); eta="~$(( remain/3600 ))h$(( (remain%3600)/60 ))m"
    fi
    {
      echo "[$(date '+%m-%d %H:%M:%S')] ── HEARTBEAT ── ${done_ct}/${total} done | ${fail_ct} failed | elapsed $(( elapsed/3600 ))h$(( (elapsed%3600)/60 ))m | ETA ${eta}"
      for f in results/logs/*.log; do
        [ -f "$f" ] || continue
        rn=$(basename "$f" .log)
        [ "$rn" = "multigpu_grid" ] && continue
        [ -f "results/${rn}/.done" ] && continue
        [ -n "$(find "$f" -newermt "-$(( 2*HEARTBEAT_SECS )) seconds" 2>/dev/null)" ] || continue
        last=$(grep -hE '=== Task|val_f1|Final: AA' "$f" 2>/dev/null | tail -1 | tr -s ' ')
        echo "    running: ${rn} | ${last:-starting...}"
      done
    } | tee -a "$LOG"
    write_progress_json "$total" "$done_ct" "$fail_ct"
  done
}

on_exit() {
  [ -n "${HB_PID:-}" ] && kill "$HB_PID" 2>/dev/null
  [ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null
  sync_push_once
  # Wipe any temp rclone config (RCLONE_CONFIG_B64 fallback path) so no creds linger on disk.
  [ -n "${RCLONE_TMP_CONF:-}" ] && shred -u "$RCLONE_TMP_CONF" 2>/dev/null || rm -f "${RCLONE_TMP_CONF:-}" 2>/dev/null
  [ -n "${HB_MANIFEST:-}" ] && rm -f "$HB_MANIFEST" 2>/dev/null
  say "[exit] heartbeat/sync stopped; final push done (or no-op)"
}

# ── Build the job list: each line = "<run_name>|<hydra overrides>" ───────────────
JOBS=()
add_job() { JOBS+=("$1|$2"); }

# Job-list builders, split by group so dispatch ORDER is controllable. Resume (.done skip)
# makes reordering harmless — already-finished runs are skipped regardless of order.
#
# CORE_BEFORE_DOCCL = methods that should run BEFORE DocCL (the cheap/medium baselines).
# CORE_AFTER_DOCCL  = heavy core methods to run AFTER DocCL (der_pp, the heaviest replay).
# Both derive from CORE_METHODS so overriding CORE_METHODS still works; der_pp is split out.
CORE_BEFORE_DOCCL="${CORE_BEFORE_DOCCL:-naive joint ewc lwf er}"
CORE_AFTER_DOCCL="${CORE_AFTER_DOCCL:-der_pp}"

add_singletask() {
  # Single-task baselines (provide b_i for FWT). One dataset per single scenario.
  # Scenario-INDEPENDENT, so they don't follow the SCENARIOS partition — set
  # RUN_SINGLETASK=0 on all-but-one box when partitioning the grid across machines,
  # so the single-task set runs on exactly one box (no cross-machine duplication).
  [ "${RUN_SINGLETASK:-1}" = "1" ] || return 0
  for sc in single_funsd single_cord single_sroie single_xfund single_wildreceipt; do
    for s in $SEEDS; do add_job "${sc}_naive_seed${s}" "method=naive scenario=${sc} seed=${s}"; done
  done
}
add_core_group() {  # $1 = space-separated method list
  for m in $1; do for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
  done; done; done
}
add_prompt() {
  for m in $PROMPT_METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
  done; done; done
}
add_doccl() {
  # DocCL main (full method = target_depth 'all', the config default) across scenarios
  if [ "$RUN_DOCCL" = "1" ]; then for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_doccl_seed${s}" "method=doccl scenario=${sc} seed=${s}"
  done; done; fi
  # DocCL depth ablation (head_only/late_only/uniform) on the ablation scenario(s)
  if [ "$RUN_ABLATION" = "1" ]; then for sc in $ABLATION_SCENARIOS; do for s in $SEEDS; do
    for tgt in $DEPTH_TARGETS; do
      add_job "${sc}_doccl_seed${s}_${tgt}" "method=doccl scenario=${sc} seed=${s} method.target_depth=${tgt}"
    done
  done; done; fi
}
add_lexslot() {
  # LexSlot main (full method = slot_depth 'head_late', slot_sharing 'soft', the config
  # defaults) across scenarios on the primary LayoutLMv3 backbone. Secondary backbones are
  # covered by add_backbones (lexslot is in BACKBONE_METHODS).
  if [ "$RUN_LEXSLOT" = "1" ]; then for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_lexslot_seed${s}" "method=lexslot scenario=${sc} seed=${s}"
  done; done; fi
  # LexSlot ablation (slot_depth x slot_sharing) on the ablation scenario(s): the go/no-go.
  if [ "$RUN_ABLATION" = "1" ]; then for sc in $ABLATION_SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_lexslot_seed${s}_off" \
      "method=lexslot scenario=${sc} seed=${s} method.slot_sharing=off"
    add_job "${sc}_lexslot_seed${s}_uniform" \
      "method=lexslot scenario=${sc} seed=${s} method.slot_depth=uniform"
    add_job "${sc}_lexslot_seed${s}_head_only" \
      "method=lexslot scenario=${sc} seed=${s} method.slot_depth=head_only"
  done; done; fi
}
add_bert() {
  # BERT text-only external comparator: naive method on the BERT backbone. The run
  # name carries the _bert family suffix (train.py) so it never collides with the
  # LayoutLMv3 naive run.
  if [ "$RUN_BERT" = "1" ]; then for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_naive_seed${s}_bert" "method=naive model=bert_base scenario=${sc} seed=${s}"
  done; done; fi
}

add_backbones() {
  # Secondary-backbone generalization study: BACKBONE_METHODS on each family in BACKBONES.
  # Run-name suffix "_<family>" mirrors train.py so the .done resume + analysis dedup work.
  # Skips the (bert, naive) combo since add_bert already covers it (no double-run).
  # BACKBONE_SCENARIOS lets the generalization block use a DIFFERENT scenario list than the
  # main grid (default: the same $SCENARIOS) — e.g. run backbones only on dil+mixed while the
  # main re-runs cover all 3 core scenarios, so one launch can do both without over-running.
  [ -n "$BACKBONES" ] || return 0
  local bb_scen="${BACKBONE_SCENARIOS:-$SCENARIOS}"
  for fam in $BACKBONES; do
    for m in $BACKBONE_METHODS; do
      [ "$fam" = "bert" ] && [ "$m" = "naive" ] && [ "$RUN_BERT" = "1" ] && continue
      for sc in $bb_scen; do for s in $SEEDS; do
        add_job "${sc}_${m}_seed${s}_${fam}" \
          "method=${m} model=${fam}_base scenario=${sc} seed=${s}"
      done; done
    done
  done
}
add_currency() {  # 2025 currency baselines (er_cflat, cl_lora) — lowest priority
  for m in $CURRENCY_METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
    # er_cflat with the config default (cflat_lambda=0.0) is plain ER+SAM. To run the
    # GENUINE C-Flat++ curvature variant, set CFLAT_LAMBDA>0 (needs >=16 GB VRAM): it
    # adds a distinct "_curv" run so the two never collide, and overrides the leaf key.
    if [ "$m" = "er_cflat" ] && [ -n "${CFLAT_LAMBDA:-}" ] && [ "${CFLAT_LAMBDA}" != "0" ] \
       && [ "${CFLAT_LAMBDA}" != "0.0" ]; then
      add_job "${sc}_${m}_seed${s}_curv" \
        "method=${m} scenario=${sc} seed=${s} +method.cflat_lambda=${CFLAT_LAMBDA}"
    else
      add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
    fi
  done; done; done
}

# Dispatch order (user-requested priority): CLASSICAL baselines -> DocCL (contribution)
# -> CURRENCY (2025 baselines: er_cflat, cl_lora). The classical tier is the full
# measured suite (single-task, core, der_pp, prompt/LoRA, BERT text-only); DocCL + its
# ablation land next so the headline is banked; the newest baselines fill in last.
# Resume (.done skip) makes the order purely a scheduling preference. (PRIORITY_DOCCL=1
# still forces DocCL absolutely first, before even the classical tier.)
if [ "${PRIORITY_DOCCL:-0}" = "1" ]; then
  add_doccl
  add_lexslot
  add_singletask
  add_core_group "$CORE_BEFORE_DOCCL"
  add_core_group "$CORE_AFTER_DOCCL"
  add_bert
  add_prompt
  add_currency
  add_backbones   # Tier 4 — secondary-backbone generalization (LiLT/BROS/BERT-rest)
else
  # Tier 1 — classical baselines (the measured suite).
  add_singletask
  add_core_group "$CORE_BEFORE_DOCCL"
  add_core_group "$CORE_AFTER_DOCCL"
  add_bert
  add_prompt
  # Tier 2 — the contributions (DocCL + LexSlot).
  add_doccl
  add_lexslot
  # Tier 3 — 2025 currency baselines.
  add_currency
  # Tier 4 — secondary-backbone generalization study (empty BACKBONES = no-op).
  add_backbones
fi

# This launch's run-names, one per line. The heartbeat counts ONLY these .done markers as
# "done" — NOT every results/*/.done on disk. On a shared R2 bucket the local results/ also
# holds the OTHER box's completed runs (pulled by sync) plus stale .done from prior launches,
# so `ls results/*/.done | wc -l` over-counts and produced the nonsensical 145/141 (done > total).
HB_MANIFEST="results/logs/.hb_jobs_$$"
printf '%s\n' "${JOBS[@]}" | cut -d'|' -f1 > "$HB_MANIFEST"

NUM_GPUS=$(echo $GPUS | wc -w)
TOTAL_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))
say "$SESSION_MARKER"   # per-launch boundary; heartbeat counts [FAIL only after this
say "=== MULTI-GPU GRID: ${#JOBS[@]} jobs, ${NUM_GPUS} GPU(s) x ${JOBS_PER_GPU} = ${TOTAL_SLOTS} slots, bs=${BATCH_SIZE}, cap=${EPOCHS_CAP}ep ==="

if [ "$DRY_RUN" = "1" ]; then
  printf '%s\n' "${JOBS[@]}" | sed 's/|/  ->  /' | nl
  say "DRY_RUN=1 — plan above, not executed."
  exit 0
fi

# Durable resume: inject creds + pull prior .done/metrics BEFORE the scheduler reads them,
# so a fresh on-demand instance skips already-completed runs (no-op if SYNC_REMOTE unset).
sync_setup
sync_pull

# Prepare data once (SROIE local artifact; XFUND/WildReceipt pull from HF on first use).
if [ ! -f data/sroie/train.json ]; then
  say "Preparing SROIE from HF mirror ..."
  "$PYBIN" scripts/prepare_sroie.py --source hf >> "$LOG" 2>&1 || say "prepare_sroie nonzero (continuing)"
fi

# Install exit trap + start background heartbeat & periodic push (after sync_pull).
trap on_exit EXIT INT TERM
if [ "$HEARTBEAT_SECS" != "0" ]; then heartbeat_loop "${#JOBS[@]}" & HB_PID=$!; fi
if [ -n "$SYNC_REMOTE" ]; then sync_push_loop & SYNC_PID=$!; fi

# ── Slot scheduler: round-robin assign jobs to (gpu, slot) workers ──────────────
# Each slot id maps to a GPU: slot k -> gpu = GPUS[k % NUM_GPUS]. We launch up to
# TOTAL_SLOTS jobs concurrently and refill a slot as soon as its job finishes.
GPU_ARR=($GPUS)
declare -A SLOT_PID   # slot -> pid of running job
declare -A SLOT_RUN   # slot -> run name (for logging)

run_one_bg() {  # <slot> <run_name> <overrides...>
  local slot="$1" run="$2"; shift 2
  local gpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
  local marker="results/${run}/.done"
  if [ -f "$marker" ]; then echo "skip"; return 0; fi
  # Two run-name-targeted overrides, appended AFTER $EXTRA so Hydra's last-wins makes them
  # authoritative:
  #
  #   1. gradient_checkpointing=true for the FULL-FINETUNE heavy methods ONLY — the LoRA
  #      methods (o_lora/cl_lora), doccl + its depth-ablation variants, and er_cflat (SAM does
  #      2 backbone forwards). These train the full backbone and hold ~17-19 GiB/process at
  #      bs16 with no ckpt, so 2-3 co-resident OOM a 44-48 GiB card. Ckpt drops them enough to
  #      pack safely. The PROMPT methods (dualprompt/l2p/coda_prompt) are deliberately EXCLUDED:
  #      they FREEZE the backbone (only the prompt pool + classifier train), so measured peak is
  #      ~3.6 GiB — they never approach the VRAM ceiling, and checkpointing them was pure ~20-30%
  #      slowdown for zero benefit (a real regression: ~26 epochs/task x slow epochs on big
  #      datasets like cil_wildreceipt = hours/task). Classical methods (naive/ewc/lwf/er/der_pp/
  #      joint/bert) are also un-checkpointed (light). HEAVY_GRAD_CKPT=0 disables entirely.
  #   2. epochs=PROMPT_EPOCHS_CAP (default 30) for PROMPT methods ONLY — the global EPOCHS_CAP
  #      (100) is wasteful for them: they top out at AA 0-30 and dualprompt was still inching up
  #      at ep28 (~25->30, diminishing). 30 captures ~all realistic gain. This cap must NOT apply
  #      to o_lora/cl_lora/doccl — those need full convergence (they're LoRA baselines / the
  #      contribution). PROMPT_EPOCHS_CAP= (empty) keeps the global cap.
  local heavy_ckpt="" prompt_epochs=""
  case "$run" in
    *_o_lora_*|*_cl_lora_*|*_doccl_*|*_er_cflat_*)
      [ "${HEAVY_GRAD_CKPT:-1}" = "1" ] && heavy_ckpt="training.gradient_checkpointing=true" ;;
  esac
  case "$run" in
    *_dualprompt_*|*_l2p_*|*_coda_prompt_*)
      # ${VAR-30}: 30 only when UNSET; an explicit PROMPT_EPOCHS_CAP= (empty) keeps the global cap.
      local _pe="${PROMPT_EPOCHS_CAP-30}"
      [ -n "$_pe" ] && prompt_epochs="method.epochs=${_pe}" ;;
  esac
  (
    local rc
    if [ "$TEE_TRAIN" = "1" ]; then
      # Full log -> per-run file; a filtered view -> the main log (visible via `docker logs`).
      # CRITICAL: .done must gate on train.py's exit (PIPESTATUS[0]), NOT the pipeline's last
      # element (sed always exits 0), or a failed run would be falsely marked done.
      CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA $heavy_ckpt $prompt_epochs 2>&1 \
        | tee "results/logs/${run}.log" \
        | grep --line-buffered -E '=== Task|val_f1|STOP|Final: AA|Zero-shot' \
        | sed -u "s#^#    [${run} gpu${gpu}] #" >> "$LOG"
      rc=${PIPESTATUS[0]}
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA $heavy_ckpt $prompt_epochs \
        >> "results/logs/${run}.log" 2>&1
      rc=$?
    fi
    [ "$rc" -eq 0 ] && mkdir -p "results/${run}" && touch "$marker"
    exit "$rc"
  ) &
  SLOT_PID[$slot]=$!
  SLOT_RUN[$slot]="$run (gpu${gpu})"
  echo "launched"
}

ji=0
DONE_CT=0; SKIP_CT=0; FAIL_CT=0
declare -A SLOT_VRAM   # slot -> GiB estimate of the job it holds (budget mode)
# Per-GPU live VRAM tally (budget mode). budget_ok <gpu> <est>: would this job still fit?
declare -A GPU_VRAM
budget_ok() {  # <gpu> <est> -> 0 (fits) / 1 (would overflow)
  [ "$GPU_VRAM_BUDGET_GB" = "0" ] && return 0          # budget disabled -> always "fits"
  local gpu="$1" est="$2" used="${GPU_VRAM[$gpu]:-0}"
  # Always allow at least one job per GPU even if its est alone exceeds the budget (never deadlock).
  [ "$used" = "0" ] && return 0
  [ $(( used + est )) -le "$GPU_VRAM_BUDGET_GB" ] && return 0 || return 1
}
# Find the next not-yet-done job that FITS gpu's remaining budget; dispatch into slot. Sets a
# global LAST_DISPATCHED=1 on success, 0 if none fit / queue drained (budget mode scans ahead).
dispatch_into() {  # <slot> <gpu>
  local slot="$1" gpu="$2" k rn ov est
  LAST_DISPATCHED=0
  for (( k=ji; k<${#JOBS[@]}; k++ )); do
    IFS='|' read -r rn ov <<< "${JOBS[$k]}"
    [ -f "results/${rn}/.done" ] && continue
    est=$(vram_est_for "$rn")
    if [ "$GPU_VRAM_BUDGET_GB" != "0" ] && ! budget_ok "$gpu" "$est"; then
      [ "$k" -eq "$ji" ] && return 0   # head doesn't fit now; wait for a slot to free (don't reorder past it greedily beyond budget)
      continue                          # try a lighter job further down the queue
    fi
    # consume queue entries up to k (mark the skipped-done ones)
    while [ $ji -le $k ]; do
      IFS='|' read -r _rn _ov <<< "${JOBS[$ji]}"; ji=$((ji+1))
      [ "$_rn" = "$rn" ] && break
      [ -f "results/${_rn}/.done" ] && SKIP_CT=$((SKIP_CT+1))
    done
    # shellcheck disable=SC2086
    run_one_bg "$slot" "$rn" $ov >/dev/null
    SLOT_VRAM[$slot]=$est; GPU_VRAM[$gpu]=$(( ${GPU_VRAM[$gpu]:-0} + est ))
    say "[run] ${SLOT_RUN[$slot]} (${est}G; gpu${gpu} ${GPU_VRAM[$gpu]}/${GPU_VRAM_BUDGET_GB}G; $ji/${#JOBS[@]})"
    LAST_DISPATCHED=1; return 0
  done
  return 0
}
# Prime each slot 0..TOTAL_SLOTS-1 with the next fitting job.
slot=0
while [ $slot -lt $TOTAL_SLOTS ] && [ $ji -lt ${#JOBS[@]} ]; do
  gpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
  dispatch_into "$slot" "$gpu"
  [ "$LAST_DISPATCHED" = "1" ] || break   # budget full or queue drained
  slot=$((slot+1))
done

# Main loop: wait for any slot to free, then refill from the queue (budget-aware).
# In budget mode a freed slot may dispatch MULTIPLE light jobs (if VRAM freed allows) or none
# (if the queue head still doesn't fit) — so after each completion we re-scan every free slot.
while [ ${#SLOT_PID[@]} -gt 0 ] || [ $ji -lt ${#JOBS[@]} ]; do
  for slot in "${!SLOT_PID[@]}"; do
    pid=${SLOT_PID[$slot]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      if [ $rc -eq 0 ]; then DONE_CT=$((DONE_CT+1)); say "[done] ${SLOT_RUN[$slot]}"; \
        else FAIL_CT=$((FAIL_CT+1)); say "[FAIL rc=$rc] ${SLOT_RUN[$slot]}"; fi
      # release this slot's VRAM reservation back to its GPU
      fgpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
      GPU_VRAM[$fgpu]=$(( ${GPU_VRAM[$fgpu]:-0} - ${SLOT_VRAM[$slot]:-0} ))
      [ "${GPU_VRAM[$fgpu]}" -lt 0 ] && GPU_VRAM[$fgpu]=0
      unset 'SLOT_PID[$slot]'; unset 'SLOT_RUN[$slot]'; unset 'SLOT_VRAM[$slot]'
    fi
  done
  # (Re)fill every currently-free slot with any job that fits its GPU's remaining budget.
  for (( slot=0; slot<TOTAL_SLOTS; slot++ )); do
    [ -n "${SLOT_PID[$slot]:-}" ] && continue
    [ $ji -lt ${#JOBS[@]} ] || break
    gpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
    dispatch_into "$slot" "$gpu"
  done
  sleep 5
done

say "=== COMPLETE. done=$DONE_CT skip=$SKIP_CT fail=$FAIL_CT (this launch) | markers in results/ (all partitions)=$(ls results/*/.done 2>/dev/null | wc -l) ==="
say "Next: \$PYBIN scripts/analyze_results.py && \$PYBIN scripts/ingest_to_thesis.py"
