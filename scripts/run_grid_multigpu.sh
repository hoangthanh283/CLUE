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

GPUS="${GPUS:-0 1}"; GPUS="${GPUS//,/ }"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS_CAP="${EPOCHS_CAP:-100}"
SEEDS="${SEEDS:-42 123 7}"
SCENARIOS="${SCENARIOS:-cil_cord dil mixed dil_xlingual cil_wildreceipt}"
CORE_METHODS="${CORE_METHODS:-naive joint ewc lwf er der_pp}"
PROMPT_METHODS="${PROMPT_METHODS:-l2p dualprompt coda_prompt o_lora}"
RUN_DOCCL="${RUN_DOCCL:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
ABLATION_SCENARIOS="${ABLATION_SCENARIOS:-cil_cord}"
DEPTH_TARGETS="${DEPTH_TARGETS:-head_only late_only uniform}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

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

EXTRA="training.batch_size=${BATCH_SIZE} training.gradient_checkpointing=false \
training.num_workers=4 method.epochs=${EPOCHS_CAP} wandb.project=${WANDB_PROJECT:-CL4IE}"

# ── Durable-resume sync (rclone; pure no-op unless SYNC_REMOTE is set) ───────────
sync_setup() {
  [ -n "$SYNC_REMOTE" ] || return 0
  if ! command -v rclone >/dev/null 2>&1; then
    say "[sync] rclone not installed — durable sync DISABLED"; SYNC_REMOTE=""; return 0
  fi
  if [ -n "$RCLONE_CONFIG_B64" ]; then
    mkdir -p "$HOME/.config/rclone"
    echo "$RCLONE_CONFIG_B64" | base64 -d > "$HOME/.config/rclone/rclone.conf" 2>/dev/null \
      && say "[sync] injected rclone.conf" || say "[sync] WARN: could not decode RCLONE_CONFIG_B64"
  fi
  say "[sync] durable resume ON -> ${SYNC_REMOTE}"
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
    fail_ct=$(grep -c '\[FAIL' "$LOG" 2>/dev/null || echo 0)
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
  say "[exit] heartbeat/sync stopped; final push done (or no-op)"
}

# ── Build the job list: each line = "<run_name>|<hydra overrides>" ───────────────
JOBS=()
add_job() { JOBS+=("$1|$2"); }

# 1) Single-task baselines FIRST (provide b_i for FWT). One dataset per single scenario.
for sc in single_funsd single_cord single_sroie single_xfund single_wildreceipt; do
  for s in $SEEDS; do add_job "${sc}_naive_seed${s}" "method=naive scenario=${sc} seed=${s}"; done
done
# 2) Core methods x scenarios x seeds
for m in $CORE_METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
  add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
done; done; done
# 3) Prompt/LoRA methods
for m in $PROMPT_METHODS; do for sc in $SCENARIOS; do for s in $SEEDS; do
  add_job "${sc}_${m}_seed${s}" "method=${m} scenario=${sc} seed=${s}"
done; done; done
# 4) DocCL main (full method = target_depth 'all', the config default) across scenarios
if [ "$RUN_DOCCL" = "1" ]; then for sc in $SCENARIOS; do for s in $SEEDS; do
  add_job "${sc}_doccl_seed${s}" "method=doccl scenario=${sc} seed=${s}"
done; done; fi
# 5) DocCL depth ablation (head_only/late_only/uniform) on the ablation scenario(s)
if [ "$RUN_ABLATION" = "1" ]; then for sc in $ABLATION_SCENARIOS; do for s in $SEEDS; do
  for tgt in $DEPTH_TARGETS; do
    add_job "${sc}_doccl_seed${s}_${tgt}" "method=doccl scenario=${sc} seed=${s} method.target_depth=${tgt}"
  done
done; done; fi

NUM_GPUS=$(echo $GPUS | wc -w)
TOTAL_SLOTS=$(( NUM_GPUS * JOBS_PER_GPU ))
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
  python scripts/prepare_sroie.py --source hf >> "$LOG" 2>&1 || say "prepare_sroie nonzero (continuing)"
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
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA 2>&1 \
        | tee "results/logs/${run}.log" \
        | grep --line-buffered -E '=== Task|val_f1|STOP|Final: AA|Zero-shot' \
        | sed -u "s#^#    [${run} gpu${gpu}] #" >> "$LOG"
      rc=${PIPESTATUS[0]}
    else
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/train.py "$@" "wandb.mode=${WANDB_MODE}" $EXTRA \
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
say "Next: python scripts/analyze_results.py && python scripts/ingest_to_thesis.py"
