"""Copy generated result artifacts into the thesis tree (committed, CI-safe).

The thesis builds from committed files under ``thesis/figures/`` and
``thesis/generated/`` (``results/`` is gitignored, so figures/tables there would
not exist in CI). This script copies the latest pilot figures, main-grid figure,
and LaTeX table fragments into the thesis tree. ``chapters/chapter6.tex`` uses
``\\IfFileExists`` so the build stays green whether or not these files exist yet —
run this after the pilot (Stage 4) and again after the main grid (Stage 5).

Usage:
    python scripts/analyze_results.py            # writes table/figure artifacts
    python -m doccl.pilot.analyze                 # writes pilot figures
    python scripts/ingest_to_thesis.py            # copies them into thesis/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# (source relative to --results_dir, destination relative to --thesis_dir)
FIGURES = [
    ("pilot/figures/fisher_bars.pdf", "figures/pilot_fisher_bars.pdf"),
    ("pilot/figures/cka_heatmap.pdf", "figures/pilot_cka_heatmap.pdf"),
    ("figure_forgetting_curves.pdf", "figures/forgetting_curves.pdf"),
]
TABLES = [
    ("table_main.tex", "generated/table_main.tex"),
    ("table_ablation.tex", "generated/table_ablation.tex"),
    ("table_compute.tex", "generated/table_compute.tex"),
]


def ingest(results_dir: Path, thesis_dir: Path) -> tuple[list[str], list[str]]:
    (thesis_dir / "figures").mkdir(parents=True, exist_ok=True)
    (thesis_dir / "generated").mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for src_rel, dst_rel in FIGURES + TABLES:
        src, dst = results_dir / src_rel, thesis_dir / dst_rel
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(f"{src}  ->  {dst}")
        else:
            missing.append(str(src))
    return copied, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--thesis_dir", type=Path, default=Path("thesis"))
    args = parser.parse_args()

    copied, missing = ingest(args.results_dir, args.thesis_dir)
    print(f"Ingested {len(copied)} artifact(s) into {args.thesis_dir}:")
    for c in copied:
        print(f"  + {c}")
    if missing:
        print(f"\n{len(missing)} not yet generated (skipped — rerun after the relevant stage):")
        for m in missing:
            print(f"  - {m}")
    print("\nRebuild the thesis (latexmk) to pick up the new artifacts.")


if __name__ == "__main__":
    main()
