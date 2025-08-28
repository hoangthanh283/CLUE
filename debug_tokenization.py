#!/usr/bin/env python3
"""Debug script to check what data is being passed to LayoutLMv3 tokenizer"""

import sys
sys.path.append('.')

from src.data.layoutlm_datasets import get_dataset_loader, create_data_loader
from src.utils import load_config
import yaml

# Load config
config = load_config('configs/layoutlm_funsd.yaml')

print("Creating dataset...")
try:
    # Load dataset
    dataset_loader = get_dataset_loader(config)
    train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

    # Get label information
    label_list = dataset_loader.get_label_list()
    label2id = {label: i for i, label in enumerate(label_list)}

    print(f"Dataset: {config['dataset']['name']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Labels: {label_list}")

    # Create examples
    print("Creating examples...")
    train_examples = dataset_loader.create_examples(train_dataset)
    print(f"Train examples: {len(train_examples)}")
    
    # Check first raw example
    first_example = train_examples[0]
    print(f"First example words: {first_example.words[:5]}...")
    print(f"First example labels: {first_example.labels[:5]}...")
    print(f"First example bboxes: {first_example.bboxes[:5]}...")
    print(f"Labels type: {type(first_example.labels)}")
    print(f"Labels item type: {type(first_example.labels[0]) if first_example.labels else 'Empty'}")
    
    # Try tokenizing manually
    print("\nManual tokenization test...")
    tokenizer = dataset_loader.tokenizer
    print(f"Tokenizer type: {type(tokenizer)}")
    
    # Convert string labels to integers
    integer_labels = [label2id.get(label, 0) for label in first_example.labels]
    print(f"String labels: {first_example.labels[:5]}")
    print(f"Integer labels: {integer_labels[:5]}")
    
    encoding = tokenizer(
        first_example.words,
        boxes=first_example.bboxes,
        word_labels=integer_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    print("Manual tokenization successful!")
    print(f"Encoding keys: {list(encoding.keys())}")
    print(f"Labels shape: {encoding['labels'].shape}")
    
    # Try creating a data loader (this is where the error happens)
    print("\nTesting data loader creation...")
    train_dataloader = create_data_loader(
        train_examples[:1], tokenizer, label2id, config, is_training=True
    )
    print("Data loader created successfully!")
    
    # Try getting one batch
    print("\nTesting batch retrieval...")
    for batch in train_dataloader:
        print("Batch retrieved successfully!")
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        break
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc()
