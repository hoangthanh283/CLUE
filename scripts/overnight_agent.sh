#!/bin/bash
# ==============================================================================
# overnight_agent.sh
# Launches an autonomous Claude Code agent to run all pending CL experiments.
#
# Usage:
#   bash scripts/overnight_agent.sh              # run immediately
#   bash scripts/overnight_agent.sh --dry-run    # just print what would happen
#
# Prerequisites:
#   1. Run seed_experiment_tasks.sh first to create beads tasks
#   2. `claude` CLI must be in PATH (claude.ai/code)
#   3. ANTHROPIC_API_KEY must be set (or claude must be authenticated)
# ==============================================================================

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="$REPO/scripts/AGENT_PROMPT.md"
DATE_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/tmp"
LOG="$LOG_DIR/overnight_agent_${DATE_TAG}.log"
RESULTS_DIR="$REPO/results/nightly_$(date +%Y%m%d)"
PYTHON="/home/thanh/anaconda3/envs/cl4ie/bin/python"

# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

if [ ! -f "$PROMPT_FILE" ]; then
  echo "ERROR: Agent prompt not found at $PROMPT_FILE"
  echo "       Run from the cl4ie repo root or check the file exists."
  exit 1
fi

if ! command -v claude &>/dev/null; then
  echo "ERROR: 'claude' CLI not found in PATH."
  echo "       Install: https://claude.ai/code"
  exit 1
fi

if [ ! -f "$PYTHON" ]; then
  echo "ERROR: Python not found at $PYTHON"
  echo "       Activate the cl4ie conda env or update PYTHON path in this script."
  exit 1
fi

if [ "${1}" = "--dry-run" ]; then
  echo "=== DRY RUN ==="
  echo "Would run: claude -p \"\$(cat $PROMPT_FILE)\" --dangerously-skip-permissions"
  echo "Log file:  $LOG"
  echo "Results:   $RESULTS_DIR"
  exit 0
fi

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

mkdir -p "$RESULTS_DIR"

# GPU memory fragmentation fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure beads commands resolve correctly
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO"

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------

echo "======================================================================"
echo "  CL4IE Overnight Experiment Agent"
echo "  Started:  $(date)"
echo "  Log:      $LOG"
echo "  Results:  $RESULTS_DIR"
echo "  Prompt:   $PROMPT_FILE"
echo "======================================================================"
echo ""
echo "Tip: Monitor progress with:"
echo "  tail -f $LOG"
echo "  cd /home/thanh/Workspace/Master-HUST/Thesis && bd list --status=in_progress"
echo ""

{
  echo "=== Overnight Agent Started: $(date) ==="
  echo "=== Repo: $REPO ==="
  echo "=== Results dir: $RESULTS_DIR ==="
  echo ""
} | tee "$LOG"

# Run Claude Code non-interactively with full permissions
# --dangerously-skip-permissions: skips all tool confirmation prompts
claude --dangerously-skip-permissions \
  -p "$(cat "$PROMPT_FILE")" \
  2>&1 | tee -a "$LOG"

EXIT_CODE=${PIPESTATUS[0]}

{
  echo ""
  echo "=== Overnight Agent Finished: $(date) ==="
  echo "=== Exit code: $EXIT_CODE ==="
} | tee -a "$LOG"

# --------------------------------------------------------------------------
# Quick morning summary
# --------------------------------------------------------------------------

echo ""
echo "======================================================================"
echo "  Agent finished. Quick summary:"
echo "======================================================================"

if [ -d "$RESULTS_DIR" ]; then
  echo ""
  echo "Result directories:"
  ls -la "$RESULTS_DIR/" 2>/dev/null | grep "^d" | awk '{print "  "$NF}' || echo "  (none)"
  echo ""
  echo "cl_results.json files found:"
  find "$RESULTS_DIR" -name "cl_results.json" 2>/dev/null | while read f; do
    exp=$(dirname "$f" | xargs basename)
    acc=$(python3 -c "import json; r=json.load(open('$f')); print(f'{r[\"cl_metrics\"][\"ACC\"]:.4f}')" 2>/dev/null || echo "?")
    echo "  ✅ $exp: ACC=$acc"
  done
fi

echo ""
echo "Full log: $LOG"
echo "Run 'cd /home/thanh/Workspace/Master-HUST/Thesis && bd stats' to see task status."
echo ""

exit $EXIT_CODE
