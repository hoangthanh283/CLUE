#!/bin/bash

# ==============================================================================
# Comprehensive Continual Learning Experiments Runner
# ==============================================================================
# This script runs all continual learning strategies for LayoutLM-based IE
# covering all major CL algorithms: Sequential, Joint, ER, EWC, GEM, A-GEM, LwF
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD="python"
SCRIPT_PATH="scripts/train_cl.py"

# Set Python path to include src directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"
BASE_OUTPUT_DIR="results/cl_experiments"
LOG_FILE="$BASE_OUTPUT_DIR/experiment_runner.log"

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Initialize log file
echo "=== Continual Learning Experiments Started at $(date) ===" | tee "$LOG_FILE"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOG_FILE"
}

# Function to run a single experiment
run_experiment() {
    local config_file=$1
    local experiment_name=$2
    local description=$3
    
    print_status "$BLUE" ""
    print_status "$BLUE" "🚀 Starting Experiment: $experiment_name"
    print_status "$BLUE" "📝 Description: $description"
    print_status "$BLUE" "⚙️  Config: $config_file"
    print_status "$BLUE" "⏰ Started at: $(date)"
    print_status "$BLUE" "======================================================"
    
    # Create experiment-specific output directory
    local exp_output_dir="$BASE_OUTPUT_DIR/$experiment_name"
    
    # Run the experiment
    if $PYTHON_CMD "$SCRIPT_PATH" --config "$config_file" --output_dir "$exp_output_dir" 2>&1 | tee -a "$LOG_FILE"; then
        print_status "$GREEN" "✅ SUCCESS: $experiment_name completed successfully!"
        print_status "$GREEN" "📁 Results saved to: $exp_output_dir"
    else
        print_status "$RED" "❌ FAILED: $experiment_name failed with exit code $?"
        print_status "$RED" "📋 Check logs above for details"
        # Don't exit here - continue with other experiments
    fi
    
    print_status "$BLUE" "⏰ Finished at: $(date)"
    print_status "$BLUE" "======================================================"
    echo "" | tee -a "$LOG_FILE"
}

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        print_status "$CYAN" "🖥️  GPU Available: $gpu_info"
        
        # Check GPU memory (assuming RTX 2060 with 6GB as mentioned in memory)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        if [ "$gpu_memory" -lt "6000" ]; then
            print_status "$YELLOW" "⚠️  Warning: Limited GPU memory ($gpu_memory MB). Some experiments may run slower."
        fi
    else
        print_status "$YELLOW" "⚠️  Warning: nvidia-smi not found. Running on CPU will be very slow."
    fi
}

# Function to estimate experiment duration
print_experiment_info() {
    print_status "$PURPLE" ""
    print_status "$PURPLE" "📊 EXPERIMENT OVERVIEW"
    print_status "$PURPLE" "======================================================"
    print_status "$PURPLE" "🎯 Total Experiments: 7 major CL strategies"
    print_status "$PURPLE" "📋 Strategies: Sequential, Joint, ER, EWC, GEM, A-GEM, LwF"
    print_status "$PURPLE" "🗂️  Tasks: FUNSD → CORD → SROIE → WildReceipt → XFUND-zh"
    print_status "$PURPLE" "🔄 Setting: Class-IL (single head, growing label space)"
    print_status "$PURPLE" "⏱️  Estimated Duration: 8-12 hours total (varies by hardware)"
    print_status "$PURPLE" "💾 Expected Output Size: ~2-5GB results per experiment"
    print_status "$PURPLE" "======================================================"
    print_status "$PURPLE" ""
}

# Function to check prerequisites
check_prerequisites() {
    print_status "$CYAN" "🔍 Checking Prerequisites..."
    
    # Check if Python script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        print_status "$RED" "❌ Error: Training script not found at $SCRIPT_PATH"
        exit 1
    fi
    
    # Check if config files exist
    local missing_configs=0
    local configs=(
        "configs/layoutlmv3_class_il.yaml"
        "configs/layoutlmv3_joint_class_il.yaml"
        "configs/layoutlmv3_er_class_il.yaml"
        "configs/layoutlmv3_ewc_class_il.yaml"
        "configs/layoutlmv3_gem_class_il.yaml"
        "configs/layoutlmv3_agem_class_il.yaml"
        "configs/layoutlmv3_lwf_class_il.yaml"
    )
    
    for config in "${configs[@]}"; do
        if [ ! -f "$config" ]; then
            print_status "$RED" "❌ Missing config: $config"
            missing_configs=$((missing_configs + 1))
        fi
    done
    
    if [ $missing_configs -gt 0 ]; then
        print_status "$RED" "❌ Error: $missing_configs config files are missing"
        exit 1
    fi
    
    # Check Python and dependencies (basic check)
    if ! $PYTHON_CMD -c "import torch, transformers" 2>/dev/null; then
        print_status "$RED" "❌ Error: Required Python packages (torch, transformers) not available"
        exit 1
    fi
    
    print_status "$GREEN" "✅ All prerequisites check passed!"
    print_status "$CYAN" ""
}

# Main execution
main() {
    print_status "$CYAN" "======================================================"
    print_status "$CYAN" "🧠 CONTINUAL LEARNING FOR INFORMATION EXTRACTION"
    print_status "$CYAN" "🔬 Comprehensive CL Strategy Evaluation"
    print_status "$CYAN" "======================================================"
    
    # Run checks
    check_prerequisites
    check_gpu
    print_experiment_info
    
    # Confirm with user (optional - uncomment if you want manual confirmation)
    # print_status "$YELLOW" "⚠️  This will run all 7 CL experiments. Continue? (y/N)"
    # read -r response
    # if [[ ! "$response" =~ ^[Yy]$ ]]; then
    #     print_status "$YELLOW" "🛑 Experiments cancelled by user"
    #     exit 0
    # fi
    
    print_status "$CYAN" "🏁 Starting all experiments..."
    echo "" | tee -a "$LOG_FILE"
    
    # =================================================================
    # BASELINE EXPERIMENTS
    # =================================================================
    
    run_experiment \
        "configs/layoutlmv3_class_il.yaml" \
        "sequential_baseline" \
        "Sequential Fine-tuning (Baseline) - Expected catastrophic forgetting"
    
    run_experiment \
        "configs/layoutlmv3_joint_class_il.yaml" \
        "joint_upper_bound" \
        "Joint Training (Upper Bound) - All data available, best possible performance"
    
    # =================================================================
    # CONTINUAL LEARNING STRATEGIES
    # =================================================================
    
    run_experiment \
        "configs/layoutlmv3_er_class_il.yaml" \
        "experience_replay" \
        "Experience Replay - Memory-based rehearsal strategy"
    
    run_experiment \
        "configs/layoutlmv3_ewc_class_il.yaml" \
        "elastic_weight_consolidation" \
        "Elastic Weight Consolidation - Fisher Information regularization"
    
    run_experiment \
        "configs/layoutlmv3_gem_class_il.yaml" \
        "gradient_episodic_memory" \
        "Gradient Episodic Memory - Gradient-based memory constraints"
    
    run_experiment \
        "configs/layoutlmv3_agem_class_il.yaml" \
        "averaged_gem" \
        "Averaged GEM - Memory-efficient gradient projection"
    
    run_experiment \
        "configs/layoutlmv3_lwf_class_il.yaml" \
        "learning_without_forgetting" \
        "Learning without Forgetting - Knowledge distillation approach"
    
    # =================================================================
    # EXPERIMENT COMPLETION
    # =================================================================
    
    print_status "$GREEN" ""
    print_status "$GREEN" "🎉 ALL EXPERIMENTS COMPLETED!"
    print_status "$GREEN" "======================================================"
    print_status "$GREEN" "📁 Results Location: $BASE_OUTPUT_DIR"
    print_status "$GREEN" "📋 Log File: $LOG_FILE"
    print_status "$GREEN" "⏰ Total Duration: Started at $(head -n1 "$LOG_FILE" | cut -d' ' -f6-)"
    print_status "$GREEN" "⏰              Finished at $(date)"
    print_status "$GREEN" ""
    
    # Print summary of results
    print_status "$CYAN" "📊 EXPERIMENT SUMMARY:"
    print_status "$CYAN" "======================================================"
    
    local total_experiments=0
    local successful_experiments=0
    
    for exp_dir in "$BASE_OUTPUT_DIR"/*/; do
        if [ -d "$exp_dir" ]; then
            total_experiments=$((total_experiments + 1))
            exp_name=$(basename "$exp_dir")
            
            # Check if experiment completed successfully (basic heuristic)
            if [ -f "$exp_dir/continual_metrics.json" ] || [ -f "$exp_dir/experiment_results.json" ]; then
                print_status "$GREEN" "✅ $exp_name: SUCCESS"
                successful_experiments=$((successful_experiments + 1))
            else
                print_status "$RED" "❌ $exp_name: FAILED or INCOMPLETE"
            fi
        fi
    done
    
    print_status "$CYAN" ""
    print_status "$CYAN" "📈 Success Rate: $successful_experiments/$total_experiments experiments completed"
    
    if [ $successful_experiments -eq $total_experiments ]; then
        print_status "$GREEN" "🏆 Perfect! All experiments completed successfully!"
    elif [ $successful_experiments -gt 0 ]; then
        print_status "$YELLOW" "⚠️  Some experiments failed. Check logs for details."
    else
        print_status "$RED" "❌ All experiments failed. Please check configuration and dependencies."
    fi
    
    print_status "$CYAN" ""
    print_status "$CYAN" "🔍 Next Steps:"
    print_status "$CYAN" "1. Review individual experiment results in: $BASE_OUTPUT_DIR"
    print_status "$CYAN" "2. Analyze CL metrics: ACC, BWT, FWT, AAA curves"
    print_status "$CYAN" "3. Compare strategy performance across all methods"
    print_status "$CYAN" "4. Generate comparison plots and tables for your thesis"
    print_status "$CYAN" "======================================================"
}

# Handle Ctrl+C gracefully
trap 'print_status "$YELLOW" "\n⚠️  Experiment interrupted by user. Results so far saved in: $BASE_OUTPUT_DIR"; exit 130' INT

# Run main function
main "$@"