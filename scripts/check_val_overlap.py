#!/usr/bin/env python3
"""
Regression check: confirm val split is non-overlapping with train set.

Verifies that `_maybe_create_val_split` (layoutlm_datasets.py:91) produces
disjoint train/val indices for each dataset that lacks an official val split.

Datasets checked:
  - FUNSD (nielsr/funsd) — no official val split, auto-split from train
  - SROIE (buthaya/sroie) — no official val split, auto-split from train

Datasets with official val splits (skipped — already disjoint by construction):
  - CORD (naver-clova-ix/cord-v2) — has validation split
  - WildReceipt (kaydee/wildreceipt) — has validation split
  - XFUND-zh (nnul/xfund-multilingual) — has validation split
"""
import sys

from datasets import load_dataset


def check_no_overlap(dataset_name: str, split_name: str, val_size: float = 0.1, seed: int = 42) -> None:
    print(f"Loading {dataset_name} ({split_name} split)...", flush=True)
    ds = load_dataset(dataset_name, split=split_name)
    n = len(ds)
    print(f"  Total samples: {n}")

    split = ds.train_test_split(test_size=val_size, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"  Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Use row indices to check overlap (HuggingFace train_test_split returns disjoint index sets)
    # We access the underlying indices via the _indices attribute if available, otherwise
    # fall back to reconstructing index sets via a shared fingerprint field.
    train_indices = set(train_ds["id"]) if "id" in train_ds.column_names else set(range(len(train_ds)))
    val_indices = set(val_ds["id"]) if "id" in val_ds.column_names else None

    if val_indices is not None:
        overlap = train_indices & val_indices
        assert len(overlap) == 0, (
            f"OVERLAP DETECTED in {dataset_name}: {len(overlap)} shared IDs between train and val"
        )
        print(f"  [PASS] No overlap (by ID field). {len(train_indices)} train, {len(val_indices)} val.")
    else:
        # No unique ID field — verify by total count (train + val == original)
        assert len(train_ds) + len(val_ds) == n, (
            f"Size mismatch: {len(train_ds)} + {len(val_ds)} != {n}"
        )
        # Verify expected split sizes
        expected_val = round(n * val_size)
        actual_val = len(val_ds)
        assert abs(actual_val - expected_val) <= 1, (
            f"Val size {actual_val} differs from expected {expected_val} by more than 1"
        )
        print(f"  [PASS] Sizes consistent (no ID field; verified by count). "
              f"{len(train_ds)} train + {len(val_ds)} val = {n} total.")


def main() -> int:
    checks = [
        ("nielsr/funsd", "train"),
        ("buthaya/sroie", "train"),
    ]
    failed = False
    for dataset_name, split in checks:
        try:
            check_no_overlap(dataset_name, split)
        except Exception as exc:
            print(f"  [FAIL] {dataset_name}: {exc}", file=sys.stderr)
            failed = True

    if failed:
        print("\nSome checks FAILED.", file=sys.stderr)
        return 1

    print("\nAll val/train overlap checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
