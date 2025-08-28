#!/usr/bin/env python3
"""Test script for LayoutLM model functionality"""
import torch
import yaml

from src.models.layoutlm_models import create_layoutlm_model


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main test function"""
    # Load config
    config = load_config('configs/layoutlmv3_funsd.yaml')
    print("Testing model creation...")

    try:
        # Test model creation
        model = create_layoutlm_model(config)
        print(f"Model created successfully: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")

        # Test model forward pass with dummy data
        batch_size = 2
        seq_length = 10
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length),
            'bbox': torch.randint(0, 1000, (batch_size, seq_length, 4)),
            'labels': torch.randint(0, 7, (batch_size, seq_length))
        }

        print("Testing forward pass...")
        with torch.no_grad():
            outputs = model(**dummy_input)

        print("Forward pass successful!")
        print(f"Loss: {outputs.loss}")
        print(f"Logits shape: {outputs.logits.shape}")

        print("Model test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
