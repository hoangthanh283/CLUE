"""Review C1 (2026-11-11): Fisher-weighted displacement by depth bucket at the first task boundary,
per pilot condition, mean over seeds, reported WITH and WITHOUT the head. The Fisher weighting
favours the head by construction, so the encoder-only share is the informative part.

    uv run python scripts/rca/pilot_depth_profile.py [results/pilot]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/pilot")
ENC = ("input", "early", "mid", "late")
acc = defaultdict(list)
for f in sorted(root.glob("c*_seed*.json")):
    d = json.loads(f.read_text())
    recs = [r for r in d.get("displacement_records", []) if r["task_boundary"] == "0_to_1"]
    if not recs:
        continue  # older pilot JSONs carry no displacement records
    rec = recs[0]
    acc[d["condition"]].append([rec["by_depth"].get(k, 0.0) for k in (*ENC, "head")])
print(
    f"{'condition':14s} n  head/max-encoder   encoder share (input/early/mid/late), head excluded"
)
for cond, rows in sorted(acc.items()):
    m = np.mean(rows, axis=0)
    enc = m[:4] / max(m[:4].sum(), 1e-30)
    print(
        f"{cond:14s} {len(rows)}  {m[4] / max(m[:4].max(), 1e-30):12.0f}x      "
        + " / ".join(f"{x:.2f}" for x in enc)
    )
