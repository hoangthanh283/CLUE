#!/bin/bash
# ==============================================================================
# seed_experiment_tasks.sh
# Populates beads with one task per CL experiment config (idempotent).
# Run this before launching overnight_agent.sh.
# ==============================================================================

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# beads root is the Thesis directory (where .beads/ lives)
BEADS_ROOT="$(cd "$REPO/.." && pwd)"

echo "=== Seeding experiment tasks in: $BEADS_ROOT ==="
echo "=== Repo: $REPO ==="

# Experiment definitions: "config_path|experiment_name|duration_estimate"
CONFIGS=(
  "configs/layoutlmv3_class_il.yaml|sequential_baseline|~2h|Sequential Fine-tuning (catastrophic forgetting baseline)"
  "configs/layoutlmv3_joint_class_il.yaml|joint_upper_bound|~3h|Joint Training (upper bound — all data at once)"
  "configs/layoutlmv3_er_class_il.yaml|experience_replay|~2h|Experience Replay (memory-based rehearsal)"
  "configs/layoutlmv3_ewc_class_il.yaml|elastic_weight_consolidation|~2h|Elastic Weight Consolidation (Fisher regularization)"
  "configs/layoutlmv3_gem_class_il.yaml|gradient_episodic_memory|~2h|Gradient Episodic Memory (task-IL, QP projection)"
  "configs/layoutlmv3_agem_class_il.yaml|averaged_gem|~2h|Averaged GEM (class-IL, gradient projection)"
  "configs/layoutlmv3_lwf_class_il.yaml|learning_without_forgetting|~2h|Learning without Forgetting (knowledge distillation)"
)

CREATED=0
SKIPPED=0

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r config_path exp_name duration description <<< "$entry"

  abs_config="$REPO/$config_path"

  # Check if config file exists
  if [ ! -f "$abs_config" ]; then
    echo "  [WARN] Config not found, skipping: $abs_config"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Check if a task for this experiment already exists (idempotent).
  # Filter to lines that look like beads issue IDs (e.g. "Thesis-abc1234")
  existing=$(cd "$BEADS_ROOT" && bd search "$exp_name" 2>/dev/null | grep -E "^[A-Za-z]+-[a-z0-9]+" | grep -c "$exp_name" || true)
  if [ "$existing" -gt 0 ]; then
    echo "  [SKIP] Task already exists: $exp_name"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Create the beads task
  TASK_DESC="Run CL experiment: $exp_name

Config: $abs_config
Output dir: $REPO/results/nightly_\$(date +%Y%m%d)/$exp_name
Duration estimate: $duration
Description: $description

Success criteria:
- Exit code 0
- File exists: results/nightly_*/$exp_name/cl_results.json
- cl_metrics.ACC is a finite float

Agent instructions:
1. Run: PYTHONPATH=$REPO /home/thanh/anaconda3/envs/cl4ie/bin/python scripts/train_cl.py --config $abs_config --output_dir \$OUTPUT_DIR
2. On failure: read logs, investigate src/, patch code, retry (max 3 attempts)
3. git commit any src/ patches with descriptive message
4. Close this issue with ACC/BWT/FWT metrics from cl_results.json"

  echo "  [CREATE] $exp_name ($duration)"
  (cd "$BEADS_ROOT" && bd create \
    --title="Run experiment: $exp_name" \
    --description="$TASK_DESC" \
    --type=task \
    --priority=2) 2>/dev/null

  # Label as experiment for filtering
  TASK_ID=$(cd "$BEADS_ROOT" && bd list --status=open 2>/dev/null | grep "Run experiment: $exp_name" | head -1 | awk '{print $1}' || true)
  if [ -n "$TASK_ID" ]; then
    (cd "$BEADS_ROOT" && bd label "$TASK_ID" add experiment 2>/dev/null || true)
    echo "    → Created $TASK_ID"
  fi

  CREATED=$((CREATED + 1))
done

echo ""
echo "=== Seeding complete: $CREATED created, $SKIPPED skipped ==="
echo ""
echo "To view all experiment tasks:"
echo "  cd $BEADS_ROOT && bd list --status=open"
echo ""
echo "To launch the overnight agent:"
echo "  bash $REPO/scripts/overnight_agent.sh"
