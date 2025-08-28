#!/usr/bin/env python3
"""Test LayoutLMv3 model creation"""

import sys
sys.path.append('.')

from src.utils import load_config
from src.models.layoutlm_models import get_model
import torch

# Load config
config = load_config('configs/layoutlm_funsd.yaml')

print("Config model section:")
print(config['model'])

print("\nAttempting to create model...")
try:
    model = get_model(config)
    print(f"Model created successfully: {type(model)}")
    print(f"Backbone type: {type(model.backbone)}")
    
    # Test a forward pass
    batch_size = 2
    seq_len = 10
    sample_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'bbox': torch.randint(0, 1000, (batch_size, seq_len, 4)),
        'labels': torch.randint(0, 7, (batch_size, seq_len))
    }
    
    print("\nTesting forward pass...")
    outputs = model(**sample_input)
    print(f"Forward pass successful! Output keys: {list(outputs.keys())}")
    
except Exception as e:
    import traceback
    print(f"Error creating model: {e}")
    print("Full traceback:")
    traceback.print_exc()
