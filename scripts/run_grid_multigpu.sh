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
#   RUN_ABLATION    default 1 (doccl depth ablation on ABLATION_SCENARIOS)
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
# DataLoader workers PER JOB. Total worker procs = NUM_WORKERS x JOBS_PER_GPU x NGPU,
# and each worker forks a copy of the dataset working set. Heavy datasets (XFUND's 7
# languages, WildReceipt) can exhaust HOST RAM at the default x4 across all slots and
# trip the OOM killer (SIGKILL -> rc=137). Lower to 2 on RAM-constrained boxes.
NUM_WORKERS="${NUM_WORKERS:-4}"
# Gradient checkpointing: default off (big GPUs prefer the speed). Set GRAD_CKPT=true on
# VRAM-constrained boxes (e.g. only a few GB free) — it trades ~20-30%% compute for a large
# activation-memory saving, letting a job fit where it otherwise OOMs.
GRAD_CKPT="${GRAD_CKPT:-false}"
# VRAM-aware admission control. The scheduler is otherwise method-blind: at a high
# JOBS_PER_GPU it can co-schedule several HEAVY jobs (dualprompt ~17 GB, der_pp ~35 GB,
# er ~26 GB at bs=16) into the same slots and OOM, while LIGHT jobs (naive/bert/doccl
# ~5-10 GB) would happily pack. GPU_VRAM_GB caps the SUM of estimated per-job VRAM that
# may run concurrently on one GPU; a job is deferred (re-queued) until it fits. Set to
# the card's usable VRAM minus headroom (e.g. 44 for a 46/48 GB L40/A6000, 0 disables).
# This lets you raise JOBS_PER_GPU for throughput on light jobs WITHOUT OOMing on heavy.
GPU_VRAM_GB="${GPU_VRAM_GB:-0}"
SEEDS="${SEEDS:-42 123 7}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
CORE_METHODS="${CORE_METHODS:-naive joint ewc lwf er der_pp}"
PROMPT_METHODS="${PROMPT_METHODS:-l2p dualprompt coda_prompt o_lora}"
# 2025 "currency" baselines — run AFTER DocCL (lowest priority tier).
CURRENCY_METHODS="${CURRENCY_METHODS:-er_cflat cl_lora}"
# BERT text-only external comparator (model=bert_base, naive method) — classical tier.
RUN_BERT="${RUN_BERT:-1}"
RUN_DOCCL="${RUN_DOCCL:-1}"
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
    done_ct=$(ls results/*/.done 2>/dev/null | wc -l)
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
  say "[exit] heartbeat/sync stopped; final push done (or no-op)"
}

# ── Per-method VRAM estimate (GB) at bs=16, from measured peak_gpu_mem across the grid.
# Used only by GPU_VRAM_GB admission control. Conservative (rounded up) so we under-pack
# rather than OOM. der_pp/er/dualprompt are the heavies; naive/bert/doccl are light.
vram_est() {  # <run_name> -> estimated GB
  case "$1" in
    *_der_pp_*)      echo 36 ;;
    *_er_seed*)      echo 27 ;;   # plain ER (not er_cflat)
    *_dualprompt_*)  echo 18 ;;
    *_l2p_*|*_coda_prompt_*) echo 17 ;;
    *_ewc_*|*_lwf_*) echo 21 ;;
    *_er_cflat_*)    echo 16 ;;   # SAM 2x graph; conservative
    *_cl_lora_*|*_o_lora_*)  echo 14 ;;
    *_joint_*)       echo 12 ;;
    *_bert)          echo 8  ;;   # BERT text-only (run-name ends _bert)
    *_doccl_*)       echo 8  ;;
    *_naive_*)       echo 11 ;;
    *)               echo 12 ;;   # unknown -> mid estimate
  esac
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
add_bert() {
  # BERT text-only external comparator: naive method on the BERT backbone. The run
  # name carries the _bert family suffix (train.py) so it never collides with the
  # LayoutLMv3 naive run.
  if [ "$RUN_BERT" = "1" ]; then for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_naive_seed${s}_bert" "method=naive model=bert_base scenario=${sc} seed=${s}"
  done; done; fi
}
add_currency() {  # 2025 currency baselines (er_cflat, cl_lora) — lowest priority
  for m in $CURRENCY_METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
    add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
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
  add_singletask
  add_core_group "$CORE_BEFORE_DOCCL"
  add_core_group "$CORE_AFTER_DOCCL"
  add_bert
  add_prompt
  add_currency
else
  # Tier 1 — classical baselines (the measured suite).
  add_singletask
  add_core_group "$CORE_BEFORE_DOCCL"
  add_core_group "$CORE_AFTER_DOCCL"
  add_bert
  add_prompt
  # Tier 2 — the contribution.
  add_doccl
  # Tier 3 — 2025 currency baselines.
  add_currency
fi

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
declare -A SLOT_GPU   # slot -> gpu id (for VRAM accounting)
declare -A SLOT_VRAM  # slot -> estimated GB this job uses

# Sum of estimated VRAM (GB) of jobs currently running on a given GPU.
gpu_vram_used() {  # <gpu_id> -> GB
  local g="$1" total=0 s
  for s in "${!SLOT_GPU[@]}"; do
    [ "${SLOT_GPU[$s]}" = "$g" ] && total=$(( total + ${SLOT_VRAM[$s]:-0} ))
  done
  echo "$total"
}
# Does <run_name> fit on <gpu_id> within GPU_VRAM_GB given what's already running?
# Always true when GPU_VRAM_GB=0 (admission control disabled). A job whose OWN estimate
# exceeds the whole budget can never fit alongside anything -> it "fits" only on an EMPTY
# GPU (run it solo) rather than deferring forever (which would hang the scheduler).
job_fits() {  # <gpu_id> <run_name>
  [ "${GPU_VRAM_GB:-0}" -gt 0 ] || return 0
  local need; need=$(vram_est "$2")
  local used; used=$(gpu_vram_used "$1")
  if [ "$need" -gt "$GPU_VRAM_GB" ]; then
    [ "$used" -eq 0 ]   # bigger than budget -> only on an idle GPU, alone
  else
    [ $(( used + need )) -le "$GPU_VRAM_GB" ]
  fi
}

run_one_bg() {  # <slot> <run_name> <overrides...>
  local slot="$1" run="$2"; shift 2
  local gpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
  local marker="results/${run}/.done"
  if [ -f "$marker" ]; then echo "skip"; return 0; fi
  (
    local rc
    if [ "$TEE_TRAIN" = "1" ]; then
      # Full log -> per-run file; a filtered view -> the main log (visible via `docker logs`).
      # CRITICAL: .done must gate on train.py's exit (PIPESTATUS[0]), NOT the pipeline's last
      # element (sed always exits 0), or a failed run would be falsely marked done.
      CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA 2>&1 \
        | tee "results/logs/${run}.log" \
        | grep --line-buffered -E '=== Task|val_f1|STOP|Final: AA|Zero-shot' \
        | sed -u "s#^#    [${run} gpu${gpu}] #" >> "$LOG"
      rc=${PIPESTATUS[0]}
    else
      CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA \
        >> "results/logs/${run}.log" 2>&1
      rc=$?
    fi
    [ "$rc" -eq 0 ] && mkdir -p "results/${run}" && touch "$marker"
    exit "$rc"
  ) &
  SLOT_PID[$slot]=$!
  SLOT_RUN[$slot]="$run (gpu${gpu})"
  SLOT_GPU[$slot]="$gpu"
  SLOT_VRAM[$slot]=$(vram_est "$run")
  echo "launched"
}

# Pull the next not-done, VRAM-fitting job for <slot>'s GPU from the queue. Skips .done
# jobs (counts them) and DEFERS jobs that don't currently fit (rotates them to the back
# of the queue so a heavy job waits for room instead of OOMing). Returns 0 if it launched.
dispatch_next() {  # <slot>
  local slot="$1" gpu="${GPU_ARR[$(( slot % NUM_GPUS ))]}"
  local scanned=0 njobs=${#JOBS[@]}
  while [ "$scanned" -lt "$njobs" ] && [ $ji -lt ${#JOBS[@]} ]; do
    IFS='|' read -r rn ov <<< "${JOBS[$ji]}"; ji=$((ji+1)); scanned=$((scanned+1))
    if [ -f "results/${rn}/.done" ]; then SKIP_CT=$((SKIP_CT+1)); continue; fi
    if ! job_fits "$gpu" "$rn"; then
      JOBS+=("${rn}|${ov}")          # defer: requeue at the back, try a lighter job now
      continue
    fi
    # shellcheck disable=SC2086
    run_one_bg "$slot" "$rn" $ov >/dev/null
    say "[run] ${SLOT_RUN[$slot]} vram~${SLOT_VRAM[$slot]}GB used=$(gpu_vram_used "$gpu")/${GPU_VRAM_GB:-inf}GB ($ji/${#JOBS[@]})"
    return 0
  done
  return 1   # nothing fit right now (all remaining are heavier than free budget)
}

ji=0
DONE_CT=0; SKIP_CT=0; FAIL_CT=0
[ "${GPU_VRAM_GB:-0}" -gt 0 ] && say "[sched] VRAM admission ON: cap ${GPU_VRAM_GB}GB/GPU (heavy jobs deferred to avoid OOM)"
# Prime each slot 0..TOTAL_SLOTS-1 with the next not-yet-done, VRAM-fitting job.
slot=0
while [ $slot -lt $TOTAL_SLOTS ] && [ $ji -lt ${#JOBS[@]} ]; do
  dispatch_next "$slot" || break   # nothing fits yet -> remaining slots fill in the main loop
  slot=$((slot+1))
done

# Main loop: wait for any slot to free, then refill from the queue. Also retries empty
# slots every tick — under VRAM admission a slot may sit idle until a running heavy job
# frees enough budget for the next deferred job.
while [ ${#SLOT_PID[@]} -gt 0 ] || [ $ji -lt ${#JOBS[@]} ]; do
  # Reap finished slots.
  for slot in "${!SLOT_PID[@]}"; do
    pid=${SLOT_PID[$slot]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      if [ $rc -eq 0 ]; then DONE_CT=$((DONE_CT+1)); say "[done] ${SLOT_RUN[$slot]}"; \
        else FAIL_CT=$((FAIL_CT+1)); say "[FAIL rc=$rc] ${SLOT_RUN[$slot]}"; fi
      unset 'SLOT_PID[$slot]'; unset 'SLOT_RUN[$slot]'; unset 'SLOT_GPU[$slot]'; unset 'SLOT_VRAM[$slot]'
    fi
  done
  # Fill every free slot with the next fitting job (budget freed by reaped jobs).
  for slot in $(seq 0 $((TOTAL_SLOTS-1))); do
    [ -n "${SLOT_PID[$slot]:-}" ] && continue            # slot busy
    [ $ji -lt ${#JOBS[@]} ] || break                      # queue exhausted
    dispatch_next "$slot" || true                         # may not fit yet -> retry next tick
  done
  sleep 5
done

say "=== COMPLETE. done=$DONE_CT skip=$SKIP_CT fail=$FAIL_CT | total .done=$(ls results/*/.done 2>/dev/null | wc -l) ==="
say "Next: \$PYBIN scripts/analyze_results.py && \$PYBIN scripts/ingest_to_thesis.py"
