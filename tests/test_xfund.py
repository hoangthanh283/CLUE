"""Tests for the XFUND loader (cross-lingual forms) and dil_xlingual scenario.

Fast tests cover pure logic (bbox normalisation, DIL mapping, registry). Slow tests
(@pytest.mark.slow) download the dataset and verify the loader output shapes.
"""

from __future__ import annotations

import pytest

from doccl.data.dil_remapping import DIL_NAME_MAPPING
from doccl.data.xfund import XFUND_LANGS, XFUNDDataset, _normalize_box


def test_xfund_label_names_match_funsd() -> None:
    # XFUND reuses FUNSD's exact 7-tag schema.
    assert XFUNDDataset.LABEL_NAMES == [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]
    assert XFUNDDataset.NUM_LABELS == 7


def test_normalize_box_scales_to_0_1000() -> None:
    # Half-size box on a square page → midpoint coords.
    assert _normalize_box([50, 50, 100, 100], 100, 100) == [500, 500, 1000, 1000]
    # Clamp: out-of-page coords are bounded to [0, 1000].
    assert _normalize_box([-10, 0, 200, 100], 100, 100) == [0, 0, 1000, 1000]
    # Degenerate (zero dim) → zeros, never divide-by-zero.
    assert _normalize_box([10, 10, 20, 20], 0, 0) == [0, 0, 0, 0]


def test_xfund_dil_mapping_is_funsd_mapping() -> None:
    # Cross-lingual DIL relies on XFUND mapping == FUNSD mapping (same schema).
    assert DIL_NAME_MAPPING["xfund"]["B-QUESTION"] == "B-KEY"
    assert DIL_NAME_MAPPING["xfund"]["B-ANSWER"] == "B-VALUE"
    assert DIL_NAME_MAPPING["xfund"] == DIL_NAME_MAPPING["funsd"]


def test_unknown_lang_raises() -> None:
    with pytest.raises(ValueError):
        XFUNDDataset(split="test", lang="en")  # XFUND has no English (English == FUNSD)


def test_xlingual_default_langs_known() -> None:
    from doccl.data.scenarios import XLINGUAL_DEFAULT_LANGS

    assert all(lang in XFUND_LANGS for lang in XLINGUAL_DEFAULT_LANGS)
    assert "zh" in XLINGUAL_DEFAULT_LANGS  # script-shift task present


@pytest.mark.slow
def test_xfund_loader_returns_layoutlmv3_batch() -> None:
    ds = XFUNDDataset(split="test", lang="fr")
    assert len(ds) > 0
    item = ds[0]
    for key in ("input_ids", "bbox", "pixel_values", "labels"):
        assert key in item
    # bbox must be within LayoutLMv3's 0–1023 range.
    assert int(item["bbox"].max()) <= 1023
    assert int(item["bbox"].min()) >= 0


@pytest.mark.slow
def test_dil_xlingual_scenario_fixed_label_space() -> None:
    from doccl.data.dil_remapping import DIL_UNIFIED_LABELS
    from doccl.data.scenarios import get_scenario

    scenario = get_scenario("dil_xlingual")
    assert len(scenario.tasks) == 5
    # Domain = language; label space is constant across all tasks (no head growth).
    assert all(t.label_set == DIL_UNIFIED_LABELS for t in scenario.tasks)
    assert [t.metadata["lang"] for t in scenario.tasks] == ["de", "es", "fr", "it", "zh"]
