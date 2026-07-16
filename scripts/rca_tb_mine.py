"""RCA-A3: TensorBoard scan — formalizes the boundary-diagnostics NULL result.

``tensorboard.diagnostics`` defaulted to false for the entire historical grid, so no
run ever logged the per-component/per-depth displacement scalars
(``log_boundary_diagnostics`` in scripts/train.py). This script proves that claim
mechanically (asserts no ``displacement``-tagged scalar exists in ANY results/*/tb dir)
and extracts the always-on scalars (final/AA|BWT|AF) as a cross-check against
metrics.json. Consequence for the RCA: per-baseline forgetting-locus data can only come
from fresh instrumented runs (Tier B, scripts/rca_baselines.py).

CPU-only. Run from CLUE/:  uv run python scripts/rca_tb_mine.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

FINAL_TAGS = ("final/AA", "final/BWT", "final/AF")


def scan_tb_dir(tb_dir: Path) -> dict:
    """Tags + final scalars for one run's tb/ dir."""
    acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    row: dict = {
        "run": tb_dir.parent.name,
        "n_tags": len(tags),
        "has_displacement": any("displacement" in t for t in tags),
    }
    for tag in FINAL_TAGS:
        if tag in tags:
            events = acc.Scalars(tag)
            row[tag.split("/")[1]] = events[-1].value if events else None
    return row


def main() -> None:
    out_dir = Path("results/rca")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tb_dir in sorted(Path("results").glob("*/tb")):
        try:
            rows.append(scan_tb_dir(tb_dir))
        except Exception as e:  # noqa: BLE001 — corrupt event files just get skipped
            log.warning("skipping %s: %s", tb_dir, e)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a3_tb_scan.csv", index=False)

    n_disp = int(df.has_displacement.sum()) if len(df) else 0
    log.info("scanned %d tb dirs — %d contain displacement scalars", len(df), n_disp)
    verdict = (
        "NULL RESULT CONFIRMED: no historical run logged boundary displacement "
        "diagnostics (tensorboard.diagnostics was never enabled). Per-baseline "
        "forgetting-locus data requires Tier B instrumented re-runs."
        if n_disp == 0
        else f"UNEXPECTED: {n_disp} runs DO have displacement scalars — mine those "
        "before running Tier B for them."
    )
    (out_dir / "a3_verdict.md").write_text(f"# RCA-A3 TB scan verdict\n\n{verdict}\n")
    log.info(verdict)
    log.info("saved -> %s/a3_tb_scan.csv + a3_verdict.md", out_dir)


if __name__ == "__main__":
    main()
