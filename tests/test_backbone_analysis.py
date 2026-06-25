"""Regression tests for the secondary-backbone analysis path in analyze_results.

The bug these guard against: ``load_local_runs`` only special-cased the BERT family,
so LiLT/BROS runs silently averaged into the LayoutLMv3 method cells of the main
table. The fix carries ``model_family`` as a real column, scopes the main/ablation/
compute tables to the primary backbone via ``_primary_only``, and emits a dedicated
secondary-backbone table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_results import (  # noqa: E402
    PRIMARY_FAMILY,
    _primary_only,
    write_backbone_table,
)


def _toy_df() -> pd.DataFrame:
    """One LayoutLMv3, one LiLT, one BERT-textonly naive run on the same scenario."""
    return pd.DataFrame(
        [
            {
                "name": "dil_naive_seed42",
                "state": "finished",
                "method": "naive",
                "model_family": "layoutlmv3",
                "scenario": "dil",
                "seed": 42,
                "AA": 41.0,
                "BWT": -10.0,
                "AF": 5.0,
                "FWT": float("nan"),
                "target": None,
            },
            {
                "name": "dil_naive_seed42_lilt",
                "state": "finished",
                "method": "naive",
                "model_family": "lilt",
                "scenario": "dil",
                "seed": 42,
                "AA": 40.0,
                "BWT": -9.0,
                "AF": 4.0,
                "FWT": float("nan"),
                "target": None,
            },
            {
                "name": "dil_naive_seed42_bert",
                "state": "finished",
                "method": "bert_textonly",
                "model_family": "bert",
                "scenario": "dil",
                "seed": 42,
                "AA": 39.0,
                "BWT": -8.0,
                "AF": 3.0,
                "FWT": float("nan"),
                "target": None,
            },
        ]
    )


def test_primary_only_keeps_layoutlmv3_and_bert_comparator():
    """LiLT/BROS dropped from the primary tables; the BERT comparator row stays."""
    prim = _primary_only(_toy_df())
    families = set(prim["model_family"])
    assert "lilt" not in families  # secondary backbone excluded from primary tables
    assert PRIMARY_FAMILY in families
    # The text-only comparator survives despite its non-primary family (own method row).
    assert "bert_textonly" in set(prim["method"])


def test_primary_only_prevents_lilt_contaminating_naive_cell():
    """The LayoutLMv3 naive AA must not be averaged with the LiLT naive AA."""
    prim = _primary_only(_toy_df())
    layoutlm_naive = prim[(prim["method"] == "naive") & (prim["model_family"] == PRIMARY_FAMILY)]
    assert layoutlm_naive["AA"].mean() == 41.0  # not (41+40)/2


def test_backbone_table_emitted_when_secondary_runs_exist(tmp_path):
    out = tmp_path / "table_backbone_AA.tex"
    write_backbone_table(_toy_df(), out, metric="AA")
    assert out.exists()
    text = out.read_text()
    assert "LayoutLMv3" in text and "LiLT" in text  # reference + secondary both shown


def test_backbone_table_skipped_without_secondary_runs(tmp_path):
    """LayoutLMv3-only result trees write no backbone table (graceful no-op)."""
    primary_only = _toy_df()
    primary_only = primary_only[primary_only["model_family"] == PRIMARY_FAMILY]
    out = tmp_path / "table_backbone_AA.tex"
    write_backbone_table(primary_only, out, metric="AA")
    assert not out.exists()
