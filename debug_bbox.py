#!/usr/bin/env python3
import torch
import yaml
from transformers import LayoutLMv3Tokenizer

from src.data.layoutlm_datasets import FUNSDDatasetLoader


def main():
    print("Loading configuration...")
    with open("configs/layoutlm_funsd.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Creating FUNSD dataset loader...")
    dataset_loader = FUNSDDatasetLoader(config)

    print("Loading dataset...")
    train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

    # Check if we're in streaming mode
    streaming_mode = config["dataset"].get("streaming", False)
    print(f"Streaming mode: {streaming_mode}")

    print("Creating examples...")
    if streaming_mode:
        # In streaming mode, we can't use .select(), so we'll iterate and take first 3
        print("Using streaming mode - taking first 3 examples...")
        train_examples = []
        for i, item in enumerate(train_dataset):
            if i >= 3:
                break
            example = dataset_loader._process_single_item(item)
            train_examples.append(example)
    else:
        # In memory mode, we can use .select()
        sample_dataset = train_dataset.select(range(3))  # Just first 3
        train_examples = dataset_loader.create_examples(sample_dataset)

    print("Loading LayoutLMv3 tokenizer...")
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

    print("Checking examples after our processing...")
    for i, example in enumerate(train_examples):
        print(f"\nExample {i}:")
        print(f"  Number of words: {len(example.words)}")
        print(f"  Number of bboxes: {len(example.bboxes)}")

        if example.bboxes:
            bbox_array = torch.tensor(example.bboxes)
            print("  Processed bbox ranges:")
            print(f"    x0: {bbox_array[:, 0].min().item()} - {bbox_array[:, 0].max().item()}")
            print(f"    y0: {bbox_array[:, 1].min().item()} - {bbox_array[:, 1].max().item()}")
            print(f"    x1: {bbox_array[:, 2].min().item()} - {bbox_array[:, 2].max().item()}")
            print(f"    y1: {bbox_array[:, 3].min().item()} - {bbox_array[:, 3].max().item()}")

            # Check for out-of-range values
            out_of_range = (bbox_array < 0) | (bbox_array > 1000)
            if out_of_range.any():
                print("  WARNING: Found out-of-range values!")
                problematic_indices = torch.where(out_of_range)
                for idx in range(min(10, len(problematic_indices[0]))):
                    row, col = problematic_indices[0][idx].item(), problematic_indices[1][idx].item()
                    print(f"    Row {row}, Col {col}: {bbox_array[row, col].item()}")
            else:
                print("  ✓ All bbox values are within [0, 1000] range!")

            # Test tokenization
            try:
                print("  Testing tokenization...")

                # Create label mapping
                label_list = example.labels
                label2id = dataset_loader.label2id
                integer_labels = [label2id.get(label, 0) for label in label_list]

                encoding = tokenizer(
                    example.words,
                    boxes=example.bboxes,
                    word_labels=integer_labels,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )

                print("  Tokenization successful!")
                print(f"    Input shape: {encoding['input_ids'].shape}")
                print(f"    Bbox shape: {encoding['bbox'].shape}")

                # Check final bbox ranges after tokenization
                final_bbox = encoding['bbox']
                print("  Final tokenized bbox ranges:")
                print(f"    x0: {final_bbox[0, :, 0].min().item()} - {final_bbox[0, :, 0].max().item()}")
                print(f"    y0: {final_bbox[0, :, 1].min().item()} - {final_bbox[0, :, 1].max().item()}")
                print(f"    x1: {final_bbox[0, :, 2].min().item()} - {final_bbox[0, :, 2].max().item()}")
                print(f"    y1: {final_bbox[0, :, 3].min().item()} - {final_bbox[0, :, 3].max().item()}")

                # Check for problematic values
                final_out_of_range = (final_bbox < 0) | (final_bbox > 1000)
                if final_out_of_range.any():
                    print("  WARNING: Found out-of-range values in final tokenized bbox!")
                    problem_positions = torch.where(final_out_of_range)
                    for idx in range(min(10, len(problem_positions[0]))):
                        batch, seq, coord = (problem_positions[0][idx].item(),
                                             problem_positions[1][idx].item(),
                                             problem_positions[2][idx].item())
                        print(f"    Position [{batch}, {seq}, {coord}]: {final_bbox[batch, seq, coord].item()}")
                else:
                    print("  ✓ All final bbox values are within [0, 1000] range!")

            except Exception as e:
                print(f"  Tokenization failed: {e}")


if __name__ == "__main__":
    main()
