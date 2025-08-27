#!/usr/bin/env python3
"""
Data preprocessing script
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess data for IE experiments"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--dataset_type", type=str, default="conll",
        help="Type of dataset (conll, custom, etc.)"
    )

    args = parser.parse_args()

    print(f"Preprocessing {args.dataset_type} data...")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # TODO: Implement preprocessing logic
    # 1. Load raw data
    # 2. Clean and format data
    # 3. Create train/dev/test splits
    # 4. Save processed data

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
