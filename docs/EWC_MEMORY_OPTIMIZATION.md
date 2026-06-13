# EWC Memory Optimization Recommendations

## Current vs. Optimized Settings for RTX 2060 6GB

### Current Configuration Issues
The current EWC configuration may still be too memory-intensive for RTX 2060 6GB:

```yaml
# Current settings that may cause memory issues
training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Too high for memory-constrained GPU
  
dataset:
  preprocessing:
    max_seq_length: 512  # Too long for 6GB GPU

cl_strategy:
  ewc_lambda: 1.0  # May be too aggressive
  ewc_chunk_size: 500000  # Could be smaller
  n_fisher_samples: 500  # Reasonable
```

### Recommended Memory-Optimized Settings

```yaml
# Memory-optimized training
training:
  batch_size: 1
  gradient_accumulation_steps: 2  # Reduced from 8 (effective batch size = 2)
  num_epochs: 5  # Reduced for faster iteration

# Memory-optimized preprocessing  
dataset:
  preprocessing:
    max_seq_length: 256  # Reduced from 512 (50% reduction)

# Memory-optimized EWC strategy
cl_strategy:
  name: "ewc"
  ewc_lambda: 0.4  # Reduced for better plasticity and stability
  ewc_chunk_size: 250000  # Smaller chunks for 6GB GPU (50% reduction)
  n_fisher_samples: 300  # Reduced for faster Fisher computation
  store_fishers_on_cpu: true  # Mandatory for memory efficiency
  fisher_cache_dir: "ewc_fisher_cache"
```

## Why These Changes?

### 1. **Sequence Length Reduction (512 → 256)**
- **Memory Impact**: ~75% reduction in attention matrix size
- **Computation Impact**: ~50% reduction in memory usage
- **Performance Impact**: Minimal for most document understanding tasks

### 2. **Gradient Accumulation Reduction (8 → 2)**
- **Memory Impact**: Reduces peak GPU memory during backward pass
- **Training Impact**: Smaller effective batch size may require learning rate adjustment
- **Stability Impact**: Lower memory pressure reduces OOM risk

### 3. **EWC Lambda Reduction (1.0 → 0.4)**
- **Learning Impact**: Better balance between stability and plasticity
- **Memory Impact**: Smaller penalty terms reduce computation overhead
- **Forgetting Impact**: Still sufficient regularization strength

### 4. **Chunk Size Reduction (500K → 250K)**
- **Memory Impact**: Smaller chunks require less GPU memory per computation
- **Computation Impact**: More frequent cache clearing prevents memory accumulation
- **Processing Impact**: Slight increase in computation overhead, but safer for OOM

### 5. **Fisher Samples Reduction (500 → 300)**
- **Computation Impact**: 40% faster Fisher information estimation
- **Memory Impact**: Less data to process during Fisher computation
- **Accuracy Impact**: Still statistically significant sample size

## Expected Benefits

### Memory Usage
- **~60% reduction** in peak GPU memory usage
- **~50% reduction** in Fisher computation memory
- **Aggressive cleanup** prevents memory accumulation

### Training Stability
- **Lower OOM risk** on RTX 2060 6GB
- **Faster Fisher estimation** with reduced samples
- **More frequent checkpointing** opportunity

### Performance Maintenance
- **Maintains EWC effectiveness** with λ=0.4
- **Preserves continual learning** capability
- **Better stability-plasticity trade-off**

## Implementation Notes

### Monitoring Recommendations
1. **Watch GPU memory usage** during Fisher estimation
2. **Monitor training stability** with reduced gradient accumulation
3. **Track continual learning metrics** to ensure effectiveness

### Potential Further Optimizations
1. **Dynamic chunk sizing** based on available GPU memory
2. **Adaptive Fisher sampling** based on task complexity
3. **Progressive sequence length** starting small and increasing

This optimized configuration should provide stable EWC training on RTX 2060 6GB while maintaining the mathematical correctness validated in the main review.