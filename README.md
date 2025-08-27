# Continual Learning for Information Extraction (CL4IE)

This project provides a comprehensive framework for conducting continual learning experiments on information extraction tasks using LayoutLM models.

## Features

- **Multiple LayoutLM Models**: Support for LayoutLM v1, v2, and v3
- **Rich Dataset Support**: FUNSD, CORD, SROIE, XFUND, WildReceipt
- **Flexible Configuration**: YAML-based experiment configuration
- **Best Practices**: Following SOLID principles with clean, maintainable code
- **Comprehensive Evaluation**: Token-level and entity-level metrics
- **Experiment Tracking**: Support for Neptune integration

## Supported Datasets

| Dataset | Task | Languages | Entities | HuggingFace Dataset |
|---------|------|-----------|----------|-------------------|
| **FUNSD** | Form Understanding | English | 7 types | `nielsr/funsd` |
| **CORD** | Receipt Understanding | English | 30 types | `naver-clova-ix/cord-v2` |
| **SROIE** | Receipt Information Extraction | English | 4 types | `darentang/sroie` |
| **XFUND** | Multilingual Forms | 7 languages | 7 types | `nielsr/xfund` |
| **WildReceipt** | Receipt Understanding | English | 26 types | `microsoft/wildreceipt` |

## Project Structure

```
cl4ie/
├── configs/              # Experiment configuration files
│   ├── layoutlm_funsd.yaml
│   ├── layoutlm_cord.yaml
│   ├── layoutlm_sroie.yaml
│   ├── layoutlm_xfund.yaml
│   └── layoutlm_wildreceipt.yaml
├── src/                  # Core source code
│   ├── data/             # Dataset handling
│   │   ├── layoutlm_datasets.py
│   │   └── additional_datasets.py
│   ├── models/           # LayoutLM model implementations
│   │   └── layoutlm_models.py
│   ├── training/         # Training procedures
│   │   └── layoutlm_trainer.py
│   ├── cl_strategies/    # Continual learning strategies
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utilities
├── scripts/              # Training and evaluation scripts
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   └── validate.py       # Code validation
├── experiments/          # Experiment runners
├── data/                 # Raw and processed data
├── results/              # Experiment outputs
└── notebooks/            # Analysis notebooks
```

## Quick Start

### 1. Prerequisites

- **Anaconda or Miniconda**: Download from [anaconda.com](https://www.anaconda.com/products/distribution) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Python 3.9+**: Recommended for best compatibility
- **CUDA** (optional): For GPU acceleration with PyTorch

> **Why Conda?** Conda provides better package management for scientific computing libraries, handles binary dependencies more reliably than pip alone, and offers superior environment isolation for machine learning projects.

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd cl4ie

# Create conda environment with Python 3.9
conda create -n cl4ie python=3.9 -y
conda activate cl4ie

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

#### Alternative: Using environment.yml (Recommended)

For a more reproducible setup, you can also use:

```bash
# For GPU support
conda env create -f environment.yml
conda activate cl4ie

# For CPU-only (lighter installation)
conda env create -f environment-cpu.yml
conda activate cl4ie-cpu
```

#### GPU Support

For CUDA-enabled training:

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Train a Model

```bash
# Train LayoutLM on FUNSD dataset
python scripts/train.py --config configs/layoutlm_funsd.yaml --output_dir results/funsd

# Train with Neptune logging (configure in YAML file)
python scripts/train.py --config configs/layoutlm_cord.yaml --output_dir results/cord

# Resume from checkpoint
python scripts/train.py --config configs/layoutlm_sroie.yaml --resume_from_checkpoint results/sroie/checkpoint_epoch_10
```

### 4. Evaluate a Model

```bash
# Evaluate on test set
python scripts/evaluate.py 
    --config configs/layoutlm_funsd.yaml 
    --model_path results/funsd/best_model_epoch_15 
    --output_dir results/funsd_eval

# Save predictions
python scripts/evaluate.py 
    --config configs/layoutlm_cord.yaml 
    --model_path results/cord/final_model 
    --save_predictions
```

### 4. Validate Code Quality

```bash
# Run all validation checks
python scripts/validate.py
```

## Configuration

Each experiment is defined by a YAML configuration file. Here's an example:

```yaml
experiment_name: "layoutlm_funsd"

# Dataset configuration
dataset:
  name: "funsd"
  hf_dataset_name: "nielsr/funsd"
  task_type: "token_classification"
  preprocessing:
    max_seq_length: 512
    image_size: [224, 224]
    normalize_bbox: true
    include_image: true

# Model configuration  
model:
  name: "layoutlm"
  model_type: "layoutlm-base-uncased"
  pretrained_model_name: "microsoft/layoutlm-base-uncased"
  config:
    num_labels: 7
    hidden_dropout_prob: 0.1

# Training configuration
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 5e-5
  weight_decay: 0.01
  early_stopping_patience: 10
```

## Neptune Experiment Tracking

The framework supports Neptune for experiment tracking and monitoring. To enable Neptune:

1. **Install Neptune**: Already included in requirements.txt
2. **Get Neptune credentials**: Sign up at [neptune.ai](https://neptune.ai) and get your API token
3. **Configure in YAML**:

```yaml
neptune:
  use_neptune: true
  neptune_project: "your-workspace/cl4ie"  # Replace with your project
  neptune_api_token: "your-api-token"      # Or set NEPTUNE_API_TOKEN env var
  tags: ["layoutlm", "funsd", "information_extraction"]
```

4. **Set environment variable** (recommended):
```bash
export NEPTUNE_API_TOKEN="your-api-token"
```

Neptune will automatically log:
- Training and validation metrics (loss, accuracy, F1)
- Model configuration and hyperparameters
- Training progress and timing
- Best model checkpoints

## Expected Performance

Based on paper reports and community benchmarks:

| Dataset | Model | F1 Score | Notes |
|---------|-------|----------|-------|
| FUNSD | LayoutLM v1 | ~79% | Form understanding |
| CORD | LayoutLM v2 | ~94% | Receipt parsing |
| SROIE | LayoutLM v1 | ~95% | Key information extraction |
| XFUND | LayoutLM v2 | ~89% | Multilingual forms |

## Development

### Code Quality

The project follows strict code quality standards:

- **PEP 8** compliance via flake8
- **Type hints** for better code documentation
- **SOLID principles** for maintainable architecture
- **Comprehensive logging** for debugging

### Running Tests

```bash
# Activate environment first
conda activate cl4ie

# Run flake8 linting
flake8 .

# Run validation script
python scripts/validate.py
```

### Environment Management

```bash
# List conda environments
conda env list

# Update environment from requirements.txt
pip install -r requirements.txt --upgrade

# Export current environment (for sharing)
conda env export > environment_backup.yml

# Remove environment (if needed)
conda env remove -n cl4ie

# Deactivate environment
conda deactivate
```
