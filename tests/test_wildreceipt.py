"""Tests for the WildReceipt loader (large receipts) and cil_wildreceipt scenario.

Fast tests cover pure logic (flat→BIO run-encoding, label-name construction). Slow
tests (@pytest.mark.slow) download the dataset and verify loader output + scenario.
"""

from __future__ import annotations

import pytest

from doccl.data.wildreceipt import _BACKGROUND, _build_label_names


def test_build_label_names_drops_background_and_pairs_bio() -> None:
    native = ["Ignore", "Total_value", "Total_key", "Others"]
    names = _build_label_names(native)
    # O first, then B-/I- for each non-background class in upstream order.
    assert names == ["O", "B-Total_value", "I-Total_value", "B-Total_key", "I-Total_key"]
    # Background classes never produce a tag.
    assert all("Ignore" not in n and "Others" not in n for n in names)


def test_background_set() -> None:
    assert _BACKGROUND == {"Ignore", "Others"}


def test_flat_to_bio_run_encoding() -> None:
    # A tiny stand-in that exercises _flat_to_bio without a download: build the
    # translation tables the method relies on, then call it.
    from doccl.data.wildreceipt import WildReceiptDataset

    obj = WildReceiptDataset.__new__(WildReceiptDataset)  # no __init__ (no download)
    names = _build_label_names(["Ignore", "Total_value", "Total_key", "Others"])
    obj.label_to_id = {l: i for i, l in enumerate(names)}

    flat = ["Ignore", "Total_value", "Total_value", "Others", "Total_key"]
    bio = obj._flat_to_bio(flat)
    decoded = [names[i] for i in bio]
    # First token of a span → B-, consecutive same-class → I-, background → O.
    assert decoded == ["O", "B-Total_value", "I-Total_value", "O", "B-Total_key"]


@pytest.mark.slow
def test_wildreceipt_loader_returns_layoutlmv3_batch() -> None:
    from doccl.data.wildreceipt import WildReceiptDataset

    ds = WildReceiptDataset(split="test")
    assert len(ds) > 0
    # 24 entity classes (Ignore/Others dropped) → 1 + 2*24 = 49 BIO tags.
    assert WildReceiptDataset.NUM_LABELS == 49
    item = ds[0]
    for key in ("input_ids", "bbox", "pixel_values", "labels"):
        assert key in item
    assert int(item["bbox"].max()) <= 1023
    assert int(item["bbox"].min()) >= 0


@pytest.mark.slow
def test_cil_wildreceipt_scenario_grows_head() -> None:
    from doccl.data.scenarios import get_scenario

    scenario = get_scenario("cil_wildreceipt")
    assert len(scenario.tasks) == 4
    # Cumulative label sets grow monotonically across sessions (class-IL).
    sizes = [len(t.label_set) for t in scenario.tasks]
    assert sizes == sorted(sizes)
    assert scenario.joint_train_datasets is not None  # joint pool present
