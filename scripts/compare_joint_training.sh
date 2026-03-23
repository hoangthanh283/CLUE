#!/bin/bash

# Script to compare True Joint vs Progressive Joint training
# This helps validate the accuracy matrix fix and understand the differences

set -e  # Exit on error

echo "======================================================================"
echo "Comparing Joint Training Modes"
echo "======================================================================"
echo ""
echo "This script will run:"
echo "  1. True Joint Training (all datasets combined in ONE training run)"
echo "  2. Progressive Joint Training (cumulative datasets per task)"
echo ""
echo "Expected differences:"
echo "  - True Joint: All accuracy matrix rows should be IDENTICAL"
echo "  - Progressive Joint: Matrix rows should IMPROVE over tasks"
echo "======================================================================"
echo ""

# Configuration
GPU=${1:-0}
OUTPUT_BASE="results/joint_comparison_$(date +%Y%m%d_%H%M%S)"

echo "Using GPU: ${GPU}"
echo "Output directory: ${OUTPUT_BASE}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Function to extract metrics from cl_results.json
extract_metrics() {
    local results_file=$1
    if [ -f "$results_file" ]; then
        python3 -c "
import json
import sys

with open('${results_file}', 'r') as f:
    data = json.load(f)

print('CL Metrics:')
print(f\"  ACC:       {data['cl_metrics']['ACC']:.4f}\")
print(f\"  AAA:       {data['cl_metrics']['AAA']:.4f}\")
print(f\"  Forgetting: {data['cl_metrics']['Forgetting']:.4f}\")
print(f\"  BWT:       {data['cl_metrics']['BWT']:.4f}\")
print(f\"  FWT:       {data['cl_metrics']['FWT']:.4f}\")

print('\nAccuracy Matrix:')
for i, row in enumerate(data['accuracy_matrix']):
    task_name = data['task_names'][i]
    row_str = ' '.join([f'{v:.3f}' for v in row])
    print(f\"  {task_name:12s}: [{row_str}]\")
"
    else
        echo "Results file not found: $results_file"
    fi
}

# ======================================================================
# 1. Run True Joint Training
# ======================================================================
echo "======================================================================"
echo "STEP 1: Running True Joint Training"
echo "======================================================================"
echo "Config: configs/layoutlmv3_joint_class_il.yaml (true_joint: true)"
echo ""

TRUE_JOINT_DIR="${OUTPUT_BASE}/true_joint"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_cl.py \
    --config configs/layoutlmv3_joint_class_il.yaml \
    --output_dir "${TRUE_JOINT_DIR}" \
    2>&1 | tee "${TRUE_JOINT_DIR}.log"

echo ""
echo "True Joint Training completed!"
echo "----------------------------------------------------------------------"
# Find the actual results directory (train_cl.py creates a subdirectory with config name)
TRUE_JOINT_RESULTS=$(find "${TRUE_JOINT_DIR}" -name "cl_results.json" -type f | head -n 1)
if [ -n "$TRUE_JOINT_RESULTS" ]; then
    extract_metrics "$TRUE_JOINT_RESULTS"
else
    echo "Warning: cl_results.json not found in ${TRUE_JOINT_DIR}"
fi
echo "======================================================================"
echo ""

# ======================================================================
# 2. Run Progressive Joint Training
# ======================================================================
echo "======================================================================"
echo "STEP 2: Running Progressive Joint Training"
echo "======================================================================"
echo "Config: configs/layoutlmv3_progressive_joint_class_il.yaml (true_joint: false)"
echo ""

PROG_JOINT_DIR="${OUTPUT_BASE}/progressive_joint"
CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_cl.py \
    --config configs/layoutlmv3_progressive_joint_class_il.yaml \
    --output_dir "${PROG_JOINT_DIR}" \
    2>&1 | tee "${PROG_JOINT_DIR}.log"

echo ""
echo "Progressive Joint Training completed!"
echo "----------------------------------------------------------------------"
# Find the actual results directory (train_cl.py creates a subdirectory with config name)
PROG_JOINT_RESULTS=$(find "${PROG_JOINT_DIR}" -name "cl_results.json" -type f | head -n 1)
if [ -n "$PROG_JOINT_RESULTS" ]; then
    extract_metrics "$PROG_JOINT_RESULTS"
else
    echo "Warning: cl_results.json not found in ${PROG_JOINT_DIR}"
fi
echo "======================================================================"
echo ""

# ======================================================================
# 3. Generate Comparison Report
# ======================================================================
echo "======================================================================"
echo "STEP 3: Generating Comparison Report"
echo "======================================================================"

REPORT_FILE="${OUTPUT_BASE}/comparison_report.md"

python3 << EOF
import json
import os
import glob

output_base = "${OUTPUT_BASE}"
report_file = "${REPORT_FILE}"

# Find the actual results files (train_cl.py creates subdirectories with config names)
true_joint_files = glob.glob(f"{output_base}/true_joint/**/cl_results.json", recursive=True)
prog_joint_files = glob.glob(f"{output_base}/progressive_joint/**/cl_results.json", recursive=True)

if not true_joint_files:
    raise FileNotFoundError(f"No cl_results.json found in {output_base}/true_joint/")
if not prog_joint_files:
    raise FileNotFoundError(f"No cl_results.json found in {output_base}/progressive_joint/")

# Load results
with open(true_joint_files[0], 'r') as f:
    true_joint = json.load(f)

with open(prog_joint_files[0], 'r') as f:
    prog_joint = json.load(f)

# Generate markdown report
with open(report_file, 'w') as f:
    f.write("# Joint Training Comparison Report\n\n")
    f.write(f"Generated: $(date)\n\n")

    f.write("## Configuration\n\n")
    f.write("| Mode | Description | Tasks |\n")
    f.write("|------|-------------|-------|\n")
    f.write("| **True Joint** | All datasets combined in ONE training run | 1 training task |\n")
    f.write("| **Progressive Joint** | Cumulative datasets per task | 5 training tasks |\n\n")

    f.write("## CL Metrics Comparison\n\n")
    f.write("| Metric | True Joint | Progressive Joint | Difference |\n")
    f.write("|--------|-----------|-------------------|------------|\n")

    for metric in ['ACC', 'AAA', 'Forgetting', 'BWT', 'FWT']:
        tj_val = true_joint['cl_metrics'][metric]
        pj_val = prog_joint['cl_metrics'][metric]
        diff = pj_val - tj_val
        f.write(f"| **{metric}** | {tj_val:.4f} | {pj_val:.4f} | {diff:+.4f} |\n")

    f.write("\n## Accuracy Matrix Comparison\n\n")

    f.write("### True Joint (Expected: All rows identical)\n\n")
    f.write("```\n")
    f.write("After Task    | " + " | ".join([f"{t:6s}" for t in true_joint['task_names']]) + " |\n")
    f.write("------------- | " + " | ".join(["------" for _ in true_joint['task_names']]) + " |\n")
    for i, row in enumerate(true_joint['accuracy_matrix']):
        task = true_joint['task_names'][i]
        row_str = " | ".join([f"{v:.4f}" for v in row])
        f.write(f"{task:13s} | {row_str} |\n")
    f.write("```\n\n")

    f.write("### Progressive Joint (Expected: Rows improve over time)\n\n")
    f.write("```\n")
    f.write("After Task    | " + " | ".join([f"{t:6s}" for t in prog_joint['task_names']]) + " |\n")
    f.write("------------- | " + " | ".join(["------" for _ in prog_joint['task_names']]) + " |\n")
    for i, row in enumerate(prog_joint['accuracy_matrix']):
        task = prog_joint['task_names'][i]
        row_str = " | ".join([f"{v:.4f}" for v in row])
        f.write(f"{task:13s} | {row_str} |\n")
    f.write("```\n\n")

    f.write("## Analysis\n\n")

    # Check if true joint has identical rows
    tj_matrix = true_joint['accuracy_matrix']
    all_identical = all(tj_matrix[0] == row for row in tj_matrix)

    f.write("### True Joint Validation\n\n")
    if all_identical:
        f.write("✅ **PASS**: All accuracy matrix rows are identical (as expected)\n\n")
    else:
        f.write("❌ **FAIL**: Accuracy matrix rows are NOT identical (bug detected!)\n\n")

    f.write("### Progressive Joint Validation\n\n")
    pj_matrix = prog_joint['accuracy_matrix']
    # Check if rows generally improve
    avg_accs = [sum(row)/len(row) for row in pj_matrix]
    improving = all(avg_accs[i] <= avg_accs[i+1] for i in range(len(avg_accs)-1))

    if improving:
        f.write("✅ **PASS**: Average accuracy improves (or stays stable) with each task\n\n")
        f.write(f"Average accuracy progression: {' -> '.join([f'{a:.3f}' for a in avg_accs])}\n\n")
    else:
        f.write("⚠️  **WARNING**: Average accuracy does not consistently improve\n\n")
        f.write(f"Average accuracy progression: {' -> '.join([f'{a:.3f}' for a in avg_accs])}\n\n")

    f.write("## Key Findings\n\n")

    tj_acc = true_joint['cl_metrics']['ACC']
    pj_acc = prog_joint['cl_metrics']['ACC']

    f.write(f"1. **Final Accuracy**: True Joint ({tj_acc:.4f}) vs Progressive Joint ({pj_acc:.4f})\n")

    if abs(tj_acc - pj_acc) < 0.01:
        f.write("   - ✅ Very similar final performance (expected)\n\n")
    elif tj_acc > pj_acc:
        f.write(f"   - True Joint performs better by {(tj_acc-pj_acc)*100:.2f}%\n\n")
    else:
        f.write(f"   - Progressive Joint performs better by {(pj_acc-tj_acc)*100:.2f}%\n\n")

    tj_forg = true_joint['cl_metrics']['Forgetting']
    pj_forg = prog_joint['cl_metrics']['Forgetting']

    f.write(f"2. **Forgetting**: True Joint ({tj_forg:.4f}) vs Progressive Joint ({pj_forg:.4f})\n")
    f.write(f"   - True Joint should have near-zero forgetting (all rows identical)\n")
    f.write(f"   - Progressive Joint may have some forgetting between training phases\n\n")

    f.write("## Files Generated\n\n")
    f.write(f"- True Joint results: `{output_base}/true_joint/`\n")
    f.write(f"- Progressive Joint results: `{output_base}/progressive_joint/`\n")
    f.write(f"- Comparison report: `{report_file}`\n")

print(f"Comparison report generated: {report_file}")
EOF

echo ""
echo "======================================================================"
echo "COMPARISON COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to: ${OUTPUT_BASE}"
echo ""
echo "Summary:"
cat "${REPORT_FILE}"
echo ""
echo "======================================================================"
echo "View full report: ${REPORT_FILE}"
echo "True Joint logs: ${TRUE_JOINT_DIR}.log"
echo "Progressive Joint logs: ${PROG_JOINT_DIR}.log"
echo "======================================================================"
