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

# Only the tiny resume markers travel to/from durable storage (~700 B/run).
SYNC_INCLUDES=(--include "*/.done" --include "*/metrics.json" --include "*/matrix.npy"
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
  # Fail-fast preflight: prove we can write+list+delete on the remote BEFORE running 189
  # ungated jobs. A creds/bucket/endpoint mistake aborts in seconds, not after hours.
  local tmp; tmp=$(mktemp -d)
  echo "ok $(date +%s)" > "$tmp/.synccheck"
  if rclone copy "$tmp" "$SYNC_REMOTE" --include ".synccheck" >> "$LOG" 2>&1 \
     && rclone lsf "$SYNC_REMOTE" 2>/dev/null | grep -q ".synccheck"; then
    rclone delete "$SYNC_REMOTE/.synccheck" >> "$LOG" 2>&1 || true
    rm -rf "$tmp"
    say "[sync] preflight OK — durable resume ON -> ${SYNC_REMOTE}"
  else
    rm -rf "$tmp"
    say "[sync] PREFLIGHT FAILED on ${SYNC_REMOTE} — check bucket/token/endpoint."
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
  echo "launched"
}

ji=0
DONE_CT=0; SKIP_CT=0; FAIL_CT=0
# Prime each slot 0..TOTAL_SLOTS-1 with the next not-yet-done job.
slot=0
while [ $slot -lt $TOTAL_SLOTS ] && [ $ji -lt ${#JOBS[@]} ]; do
  IFS='|' read -r rn ov <<< "${JOBS[$ji]}"; ji=$((ji+1))
  if [ -f "results/${rn}/.done" ]; then SKIP_CT=$((SKIP_CT+1)); say "[skip] $rn"; continue; fi
  # shellcheck disable=SC2086
  run_one_bg "$slot" "$rn" $ov >/dev/null
  say "[run] ${SLOT_RUN[$slot]} ($ji/${#JOBS[@]} dispatched)"
  slot=$((slot+1))
done

# Main loop: wait for any slot to free, then refill from the queue.
while [ ${#SLOT_PID[@]} -gt 0 ]; do
  for slot in "${!SLOT_PID[@]}"; do
    pid=${SLOT_PID[$slot]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      if [ $rc -eq 0 ]; then DONE_CT=$((DONE_CT+1)); say "[done] ${SLOT_RUN[$slot]}"; \
        else FAIL_CT=$((FAIL_CT+1)); say "[FAIL rc=$rc] ${SLOT_RUN[$slot]}"; fi
      unset 'SLOT_PID[$slot]'; unset 'SLOT_RUN[$slot]'
      # refill this slot from the queue (skipping already-done)
      while [ $ji -lt ${#JOBS[@]} ]; do
        IFS='|' read -r rn ov <<< "${JOBS[$ji]}"; ji=$((ji+1))
        if [ -f "results/${rn}/.done" ]; then SKIP_CT=$((SKIP_CT+1)); continue; fi
        # shellcheck disable=SC2086
        run_one_bg "$slot" "$rn" $ov >/dev/null
        say "[run] ${SLOT_RUN[$slot]} ($ji/${#JOBS[@]} dispatched)"
        break
      done
    fi
  done
  sleep 5
done

say "=== COMPLETE. done=$DONE_CT skip=$SKIP_CT fail=$FAIL_CT | total .done=$(ls results/*/.done 2>/dev/null | wc -l) ==="
say "Next: \$PYBIN scripts/analyze_results.py && \$PYBIN scripts/ingest_to_thesis.py"
