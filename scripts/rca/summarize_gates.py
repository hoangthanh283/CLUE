"""Summarise every run under a gates directory: AA / BWT / final row, grouped by (scenario, method)
across seeds with mean ± sample s.d.  Usage: uv run python scripts/rca/summarize_gates.py [dir]"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gates/cil_protocol_2026-11")
rows = defaultdict(list)
for d in sorted(root.iterdir()):
    m = d / "metrics.json"
    if not m.exists():
        continue
    j = json.loads(m.read_text())
    mat = np.load(d / "matrix.npy")
    key = re.sub(r"_seed\d+", "", d.name)
    rows[key].append((j["AA"], j.get("BWT", float("nan")), np.round(mat[-1], 1).tolist()))
for key, vals in sorted(rows.items()):
    aa = np.array([v[0] for v in vals])
    sd = f" ± {aa.std(ddof=1):.1f}" if len(aa) > 1 else ""
    print(f"{key:48s} n={len(aa)}  AA {aa.mean():5.1f}{sd}   final {vals[-1][2]}")
