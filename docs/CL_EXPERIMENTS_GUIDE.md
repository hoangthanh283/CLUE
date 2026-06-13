# Continual Learning Experiments Guide

This directory contains comprehensive bash scripts to run all continual learning experiments for your LayoutLM-based Information Extraction thesis.

## 🚀 Quick Start

### Prerequisites Check
```bash
# Quick setup validation
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -c "import src.data.label_space; print('✅ Setup OK')"
```

### Available Scripts

1. **`run_all_cl_experiments.sh`** - Complete automated suite
   - Runs all 7 CL strategies sequentially
   - Comprehensive logging and error handling
   - Estimated time: 8-12 hours total

2. **`run_cl_experiments.sh`** - Flexible selective runner
   - Run specific strategies or groups
   - Support for quick testing mode
   - Fine-grained control over experiments

## 📋 Experiment Strategies

| Strategy | Config File | Description |
|----------|-------------|-------------|
| `sequential` | `configs/layoutlmv3_class_il.yaml` | Sequential Fine-tuning (Baseline) |
| `joint` | `configs/layoutlmv3_joint_class_il.yaml` | Joint Training (Upper Bound) |
| `er` | `configs/layoutlmv3_er_class_il.yaml` | Experience Replay |
| `ewc` | `configs/layoutlmv3_ewc_class_il.yaml` | Elastic Weight Consolidation |
| `gem` | `configs/layoutlmv3_gem_class_il.yaml` | Gradient Episodic Memory |
| `agem` | `configs/layoutlmv3_agem_class_il.yaml` | Averaged GEM |
| `lwf` | `configs/layoutlmv3_lwf_class_il.yaml` | Learning without Forgetting |

## 🎯 Usage Examples

### Run All Experiments
```bash
# Complete experiment suite (8-12 hours)
./run_all_cl_experiments.sh
```

### Run Specific Groups
```bash
# Baseline experiments only (Sequential + Joint)
./run_cl_experiments.sh --baseline

# Memory-based strategies (ER, GEM, A-GEM)  
./run_cl_experiments.sh --memory

# Regularization strategies (EWC, LwF)
./run_cl_experiments.sh --regularization
```

### Run Individual Strategies
```bash
# Single strategy
./run_cl_experiments.sh agem

# Multiple specific strategies
./run_cl_experiments.sh gem agem ewc

# Quick test mode (reduced epochs)
./run_cl_experiments.sh --quick agem
```

### Help and Options
```bash
# Show all available options
./run_cl_experiments.sh --help
```

## 📊 Expected Outputs

Each experiment generates:
- **Model weights**: Final trained model for each task
- **Metrics**: Continual learning metrics (ACC, BWT, FWT, AAA)
- **Logs**: Detailed training logs with hyperparameters
- **Plots**: AAA curves and performance visualizations

### Results Structure
```
results/
├── layoutlmv3_class_il/           # Sequential baseline
├── layoutlmv3_joint_class_il/     # Joint upper bound  
├── layoutlmv3_agem_class_il/      # A-GEM results
├── layoutlmv3_ewc_class_il/       # EWC results
├── layoutlmv3_gem_class_il/       # GEM results
├── layoutlmv3_er_class_il/        # Experience Replay
└── layoutlmv3_lwf_class_il/       # LwF results
```

## ⚙️ Configuration Details

### Hardware Requirements
- **GPU**: NVIDIA RTX 2060 6GB or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: ~20GB for results

### Task Sequence
All experiments follow the same 5-task sequence:
1. **FUNSD** → 2. **CORD** → 3. **SROIE** → 4. **WildReceipt** → 5. **XFUND-zh**

### Key Settings
- **CL Setting**: Class-IL (single growing head)
- **Batch Size**: 1 with gradient accumulation
- **Epochs**: 10 per task
- **Learning Rate**: 5e-5 with linear warmup
- **Early Stopping**: 3 epochs patience

## 🔍 Monitoring Progress

### Real-time Monitoring
```bash
# Monitor specific experiment
tail -f results/layoutlmv3_agem_class_il/logs/training.log

# Check GPU utilization
watch nvidia-smi
```

### Check Experiment Status
```bash
# View experiment summary
ls -la results/*/continual_metrics.json

# Check for completion
find results -name "final_model" -type d
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   ```

2. **GPU Memory Issues**  
   - Reduce batch size in configs
   - Enable gradient checkpointing
   - Use A-GEM instead of GEM

3. **Missing Dependencies**
   ```bash
   pip install torch transformers datasets evaluate
   ```

4. **Config File Not Found**
   - Ensure you're in the project root directory
   - Check config file paths in the scripts

### Performance Optimization

For **RTX 2060 6GB** constraints:
- **A-GEM** is optimized for your hardware
- **EWC** uses efficient Fisher computation  
- **ER** with small memory buffer (1000 samples)
- **GEM** redesigned to prevent OOM

## 📈 Expected Results

### Typical Performance Ranges (F1 Score)
- **Sequential**: 40-60% (catastrophic forgetting)
- **Joint**: 80-90% (upper bound)
- **A-GEM**: 65-75% (good balance)
- **EWC**: 60-70% (regularization)
- **ER**: 70-80% (memory replay)
- **LwF**: 55-65% (distillation)
- **GEM**: 70-75% (constraint satisfaction)

### Key Metrics to Compare
- **ACC**: Average accuracy across all tasks
- **BWT**: Backward transfer (forgetting measure)
- **FWT**: Forward transfer (knowledge reuse)
- **AAA**: Area under accuracy curve

## 🎓 Thesis Integration

These experiments provide comprehensive evaluation for your thesis:

1. **Baseline Comparison**: Sequential vs Joint bounds
2. **Method Analysis**: Detailed comparison of CL approaches  
3. **Memory Efficiency**: Resource usage analysis
4. **Task Sequence Effects**: Impact of task ordering
5. **Ablation Studies**: Component-wise analysis

## 📞 Support

If experiments fail:
1. Check the generated log files in `results/*/logs/`
2. Verify GPU memory usage with `nvidia-smi`
3. Ensure all config files exist and are valid YAML
4. Check Python path and imports

---

**Note**: All implementations have been validated against original papers and optimized for RTX 2060 6GB constraints based on your previous optimization work.