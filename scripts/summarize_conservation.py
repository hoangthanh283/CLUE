#!/usr/bin/env python3
"""Summarize the conservation grid (EXPLORE.md §7) across seeds: AA/BWT/row3 mean±sd."""
import json
import os

import numpy as np

VARIANTS = [
    ("CoLaR r128", "colar", ""),
    ("CoLaR+kcenter", "colar", "_kc"),
    ("CoLaR-Bal", "colar_bal", ""),
    ("CoLaR-Bal+kcenter", "colar_bal", "_kc"),
]
SEEDS = (42, 7, 123)

for name, method, suffix in VARIANTS:
    aa, bwt, rows = [], [], []
    for s in SEEDS:
        p = f"results/dil_{method}_seed{s}_k4_d50_r128{suffix}/metrics.json"
        if not os.path.exists(p):
            continue
        with open(p) as f:
            m = json.load(f)
        aa.append(m["AA"])
        bwt.append(m["BWT"])
        rows.append(np.array(m["matrix"])[-1])
    if not aa:
        print(f"{name:20s} (no runs yet)")
        continue
    r = np.mean(rows, axis=0)
    sd = f"±{np.std(aa):.1f}" if len(aa) > 1 else ""
    print(
        f"{name:20s} n={len(aa)}  AA={np.mean(aa):.1f}{sd}  BWT={np.mean(bwt):+.1f}  "
        f"row=[{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}]"
    )
