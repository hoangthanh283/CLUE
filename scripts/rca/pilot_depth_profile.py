"""Review C1 (2026-11-11): Fisher-weighted displacement by depth bucket at the first task boundary,
per pilot condition, mean over seeds, reported WITH and WITHOUT the head. The Fisher weighting
favours the head by construction, so the encoder-only share is the informative part. Also prints
the mean BWT per condition over the valid seeds.

A run is DEAD (excluded everywhere, listed with the reason) when its final AA is below 1.0 or its
retention matrix is all zeros. Runs with a non-default task order (``*_ord210``) are their own group.

    uv run python scripts/rca/pilot_depth_profile.py [results/pilot]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEAD_AA = 1.0  # final AA below this (0-100 scale) = dead run
ENC = ("input", "early", "mid", "late")


def dead_reason(d: dict) -> str | None:
    aa = d.get("cl_metrics", {}).get("AA")
    if aa is None or not aa >= DEAD_AA:
        return f"final AA {aa} < {DEAD_AA}"
    if not np.any(np.asarray(d.get("matrix", []), dtype=float)):
        return "retention matrix all zeros"
    return None


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pilot")
    disp, bwt, excluded = defaultdict(list), defaultdict(list), []
    for f in sorted(root.glob("c*_seed*.json")):
        d = json.loads(f.read_text())
        why = dead_reason(d)
        if why:
            excluded.append((f.name, why))
            continue
        group = re.sub(r"_seed\d+", "", f.stem)  # c4_full, c4_full_ord210, ...
        bwt[group].append((d["seed"], d["cl_metrics"]["BWT"]))
        recs = [r for r in d.get("displacement_records", []) if r["task_boundary"] == "0_to_1"]
        if recs:  # older pilot JSONs carry no displacement records
            disp[group].append([recs[0]["by_depth"].get(k, 0.0) for k in (*ENC, "head")])

    print(f"Excluded dead runs ({len(excluded)}):")
    for name, why in excluded:
        print(f"  {name:28s} {why}")

    print(
        f"\n{'condition':16s} n  head/max-encoder   encoder share (input/early/mid/late), "
        "head excluded"
    )
    for cond, rows in sorted(disp.items()):
        m = np.mean(rows, axis=0)
        enc = m[:4] / max(m[:4].sum(), 1e-30)
        print(
            f"{cond:16s} {len(rows)}  {m[4] / max(m[:4].max(), 1e-30):12.0f}x      "
            + " / ".join(f"{x:.2f}" for x in enc)
        )

    print(f"\n{'condition':16s} n  mean BWT   per-seed BWT")
    for cond, vals in sorted(bwt.items()):
        per = ", ".join(f"s{s}={v:.2f}" for s, v in vals)
        print(f"{cond:16s} {len(vals)}  {np.mean([v for _, v in vals]):8.2f}   {per}")


if __name__ == "__main__":
    main()
