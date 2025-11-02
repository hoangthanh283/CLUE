#!/bin/bash

# ==============================================================================
# Independent Setting Experiments Runner
# ==============================================================================
# This script runs independent training for each dataset separately
# Each dataset is trained and evaluated in isolation (not continual learning)
# Purpose: Establish baseline performance for each dataset individually
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
SCRIPT_PATH="scripts/train.py"

# Set Python path to include src directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"
BASE_OUTPUT_DIR="results/independent_experiments"
LOG_FILE="$BASE_OUTPUT_DIR/independent_runner.log"

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Initialize log file
echo "=== Independent Setting Experiments Started at $(date) ===" | tee "$LOG_FILE"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}" | tee -a "$LOG_FILE"
}

# Function to check and enable Neptune tracking
check_and_enable_neptune() {
    print_status "$CYAN" "🔍 Checking Neptune.ai configuration..."
    
    # Load .env file if it exists
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
        print_status "$CYAN" "📄 Loaded environment variables from .env file"
    fi
    
    # Check if Neptune credentials are available
    if [ -n "$NEPTUNE_PROJECT" ] && [ -n "$NEPTUNE_API_TOKEN" ]; then
        print_status "$GREEN" "✅ Neptune credentials found!"
        print_status "$GREEN" "   Project: $NEPTUNE_PROJECT"
        print_status "$GREEN" "   Token: ${NEPTUNE_API_TOKEN:0:10}...${NEPTUNE_API_TOKEN: -10}"
        print_status "$GREEN" "🔧 Enabling Neptune tracking in all configs..."
        
        # Enable Neptune in all config files
        local configs=(
            "configs/layoutlmv3_funsd.yaml"
            "configs/layoutlmv3_cord.yaml"
            "configs/layoutlmv3_sroie.yaml"
            "configs/layoutlmv3_wildreceipt.yaml"
            "configs/layoutlmv3_xfund.yaml"
        )
        
        for config in "${configs[@]}"; do
            if [ -f "$config" ]; then
                # Use sed to change use_neptune from false to true
                sed -i 's/use_neptune: false/use_neptune: true/g' "$config"
                print_status "$GREEN" "   ✓ Enabled Neptune in $config"
            fi
        done
        
        NEPTUNE_ENABLED=true
        print_status "$GREEN" ""
    else
        print_status "$YELLOW" "⚠️  Neptune credentials not found in environment"
        print_status "$YELLOW" "   Set NEPTUNE_PROJECT and NEPTUNE_API_TOKEN in .env file to enable tracking"
        print_status "$YELLOW" "   Experiments will run without Neptune tracking"
        NEPTUNE_ENABLED=false
        print_status "$YELLOW" ""
    fi
}

# Function to restore Neptune config after experiments
restore_neptune_config() {
    if [ "$NEPTUNE_ENABLED" = true ]; then
        print_status "$CYAN" "🔧 Restoring original Neptune configuration..."
        
        local configs=(
            "configs/layoutlmv3_funsd.yaml"
            "configs/layoutlmv3_cord.yaml"
            "configs/layoutlmv3_sroie.yaml"
            "configs/layoutlmv3_wildreceipt.yaml"
            "configs/layoutlmv3_xfund.yaml"
        )
        
        for config in "${configs[@]}"; do
            if [ -f "$config" ]; then
                # Restore use_neptune to false
                sed -i 's/use_neptune: true/use_neptune: false/g' "$config"
            fi
        done
        
        print_status "$CYAN" "✅ Configuration restored"
    fi
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
        
        # Check GPU memory
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
        if [ "$gpu_memory" -lt "6000" ]; then
            print_status "$YELLOW" "⚠️  Warning: Limited GPU memory ($gpu_memory MB). Experiments may run slower."
        fi
    else
        print_status "$YELLOW" "⚠️  Warning: nvidia-smi not found. Running on CPU will be very slow."
    fi
}

# Function to print experiment overview
print_experiment_info() {
    print_status "$PURPLE" ""
    print_status "$PURPLE" "📊 EXPERIMENT OVERVIEW"
    print_status "$PURPLE" "======================================================"
    print_status "$PURPLE" "🎯 Total Experiments: 5 datasets (independent training)"
    print_status "$PURPLE" "📋 Datasets: FUNSD, CORD, SROIE, WildReceipt, XFUND-de"
    print_status "$PURPLE" "🔄 Setting: Independent (each dataset trained separately)"
    print_status "$PURPLE" "🎓 Purpose: Baseline performance for each dataset"
    print_status "$PURPLE" "📐 Split: Train (90%), Validation (10%), Test (held-out)"
    print_status "$PURPLE" "⏱️  Estimated Duration: 4-8 hours total (varies by hardware)"
    print_status "$PURPLE" "💾 Expected Output Size: ~500MB-1GB per experiment"
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
        "configs/layoutlmv3_funsd.yaml"
        "configs/layoutlmv3_cord.yaml"
        "configs/layoutlmv3_sroie.yaml"
        "configs/layoutlmv3_wildreceipt.yaml"
        "configs/layoutlmv3_xfund.yaml"
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
    if ! $PYTHON_CMD -c "import torch, transformers, datasets" 2>/dev/null; then
        print_status "$RED" "❌ Error: Required Python packages (torch, transformers, datasets) not available"
        exit 1
    fi
    
    print_status "$GREEN" "✅ All prerequisites check passed!"
    print_status "$CYAN" ""
}

# Function to extract dataset statistics from config
print_dataset_info() {
    print_status "$CYAN" ""
    print_status "$CYAN" "📊 DATASET INFORMATION"
    print_status "$CYAN" "======================================================"
    print_status "$CYAN" "Dataset       | Labels | Task Type           | Split Strategy"
    print_status "$CYAN" "--------------|--------|---------------------|-------------------"
    print_status "$CYAN" "FUNSD         | 7      | Form Understanding  | 90/10 train/val"
    print_status "$CYAN" "CORD          | 31     | Receipt Parsing     | 90/10 train/val"
    print_status "$CYAN" "SROIE         | 9      | Receipt Extraction  | 90/10 train/val"
    print_status "$CYAN" "WildReceipt   | 35     | Receipt Fields      | 90/10 train/val"
    print_status "$CYAN" "XFUND (de)    | 7      | Multilingual Forms  | 90/10 train/val"
    print_status "$CYAN" "======================================================"
    print_status "$CYAN" ""
}

# Main execution
main() {
    print_status "$CYAN" "======================================================"
    print_status "$CYAN" "📚 INDEPENDENT SETTING EXPERIMENTS"
    print_status "$CYAN" "🔬 Baseline Performance Evaluation"
    print_status "$CYAN" "======================================================"
    
    # Run checks
    check_prerequisites
    check_gpu
    check_and_enable_neptune
    print_experiment_info
    print_dataset_info
    
    print_status "$CYAN" "🏁 Starting all independent experiments..."
    echo "" | tee -a "$LOG_FILE"
    
    # =================================================================
    # INDEPENDENT DATASET EXPERIMENTS
    # =================================================================
    
    run_experiment \
        "configs/layoutlmv3_funsd.yaml" \
        "funsd_independent" \
        "FUNSD - Form Understanding (independent training)"
    
    run_experiment \
        "configs/layoutlmv3_cord.yaml" \
        "cord_independent" \
        "CORD - Receipt Parsing (independent training)"
    
    run_experiment \
        "configs/layoutlmv3_sroie.yaml" \
        "sroie_independent" \
        "SROIE - Receipt Information Extraction (independent training)"
    
    run_experiment \
        "configs/layoutlmv3_wildreceipt.yaml" \
        "wildreceipt_independent" \
        "WildReceipt - Diverse Receipt Fields (independent training)"
    
    run_experiment \
        "configs/layoutlmv3_xfund.yaml" \
        "xfund_de_independent" \
        "XFUND (German) - Multilingual Form Understanding (independent training)"
    
    # =================================================================
    # EXPERIMENT COMPLETION
    # =================================================================
    
    print_status "$GREEN" ""
    print_status "$GREEN" "🎉 ALL INDEPENDENT EXPERIMENTS COMPLETED!"
    print_status "$GREEN" "======================================================"
    print_status "$GREEN" "📁 Results Location: $BASE_OUTPUT_DIR"
    print_status "$GREEN" "📋 Log File: $LOG_FILE"
    print_status "$GREEN" "⏰ Total Duration: Started at $(head -n1 "$LOG_FILE" | cut -d' ' -f5-)"
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
            
            # Check if experiment completed successfully (search recursively for model files)
            if find "$exp_dir" -name "config.json" -path "*/best_model/*" -o -name "config.json" -path "*/final_model/*" | grep -q .; then
                # Try to extract F1 score from logs if available
                local f1_score=$(find "$exp_dir" -name "*.log" -exec grep -oP "eval_f1['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" {} \; 2>/dev/null | tail -1 || echo "N/A")
                print_status "$GREEN" "✅ $exp_name: SUCCESS (Best F1: $f1_score)"
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
    print_status "$CYAN" "1. Review individual results in: $BASE_OUTPUT_DIR"
    print_status "$CYAN" "2. Compare performance across datasets"
    print_status "$CYAN" "3. Use these as baselines for continual learning experiments"
    print_status "$CYAN" "4. Analyze per-dataset metrics: F1, Precision, Recall"
    print_status "$CYAN" "5. Create comparison tables for your thesis"
    print_status "$CYAN" ""
    print_status "$CYAN" "📝 To run continual learning experiments next:"
    print_status "$CYAN" "   ./scripts/run_all_cl_experiments.sh"
    print_status "$CYAN" "======================================================"
    
    # Restore Neptune configuration
    restore_neptune_config
}

# Handle Ctrl+C gracefully - restore configs before exit
cleanup_on_interrupt() {
    print_status "$YELLOW" "\n⚠️  Experiment interrupted by user."
    restore_neptune_config
    print_status "$YELLOW" "Results so far saved in: $BASE_OUTPUT_DIR"
    exit 130
}

trap cleanup_on_interrupt INT

# Run main function
main "$@"
