"""R2 programme status + pre-registered gate verdicts (read-only, no GPU).

Reads results/<scenario>_<method>_seed<N>[_k4_d50_r128]/metrics.json for the CoLaR-R
programme and prints per-cell AA/BWT, three-seed means, and the gate verdicts from
docs/RESEARCH_PLAN_2026-10_next_mechanism.md §0'. Run: uv run python scripts/r2_status.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 7, 123)
SCENARIOS = ("cil_cord", "mixed")
METHODS = {  # label -> (run-name method token, suffix)
    "colar": ("colar", "_k4_d50_r128"),
    "colar_r_oracle": ("colar_r_oracle", "_k4_d50_r128"),
    "colar_r_pseudo": ("colar_r_pseudo", "_k4_d50_r128"),
    "cpfd": ("cpfd", ""),
    "is3": ("is3", ""),
}
# Measured three-seed LayoutLMv3 references (paper Table 1 / thesis table_main).
REF = {
    "cil_cord": {"naive": 19.0, "er": 16.0, "best": 19.0, "joint": 33.3},
    "mixed": {"naive": 33.9, "er": 60.3, "best": 62.2, "joint": 67.5},
}
BAR = {"cil_cord": REF["cil_cord"]["naive"] + 4.0, "mixed": REF["mixed"]["best"] + 1.0}


def cell(sc: str, label: str, seed: int) -> dict | None:
    tok, suf = METHODS[label]
    p = ROOT / "results" / f"{sc}_{tok}_seed{seed}{suf}" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    mat = p.with_name("matrix.npy")
    row = np.load(mat)[-1].tolist() if mat.exists() else None  # final-state retention row
    return {"AA": float(m["AA"]), "BWT": float(m.get("BWT", float("nan"))), "row": row}


def row_summary(cells: dict) -> str:
    """mean±sd of the final retention row over the seeds that have a matrix.npy."""
    rows = [c["row"] for c in cells.values() if c and c["row"]]
    if not rows:
        return ""
    cols = list(zip(*rows, strict=True))
    sd = [st.stdev(c) if len(c) > 1 else 0.0 for c in cols]
    return " ".join(f"{st.mean(c):.1f}±{d:.1f}" for c, d in zip(cols, sd, strict=True))


def summarise(sc: str) -> dict[str, dict]:
    out = {}
    for label in METHODS:
        cells = {s: cell(sc, label, s) for s in SEEDS}
        aas = [c["AA"] for c in cells.values() if c]
        out[label] = {
            "cells": cells,
            "n": len(aas),
            "mean": st.mean(aas) if aas else None,
            "sd": st.stdev(aas) if len(aas) > 1 else 0.0,
        }
    return out


def gate(sc: str, s: dict) -> list[str]:
    """Pre-registered verdicts; PENDING until all three seeds of the cells involved exist."""
    v = []
    plain, orc, pse = s["colar"], s["colar_r_oracle"], s["colar_r_pseudo"]
    comps = [s["cpfd"]["mean"], s["is3"]["mean"]]
    if pse["n"] < 3 or plain["n"] < 3:
        v.append(f"R1 pseudo-vs-bar: PENDING ({pse['n']}/3 pseudo, {plain['n']}/3 plain)")
    else:
        same_sign = all(
            (pse["cells"][k]["AA"] - plain["cells"][k]["AA"]) > 0 for k in SEEDS
        ) or all((pse["cells"][k]["AA"] - plain["cells"][k]["AA"]) < 0 for k in SEEDS)
        beats_bar = pse["mean"] >= BAR[sc]
        beats_comp = all(c is None or pse["mean"] >= c for c in comps)
        kill = pse["mean"] <= plain["mean"] + 1.0
        verdict = (
            "PASS"
            if (beats_bar and same_sign and beats_comp and not kill)
            else ("KILL" if kill else "FAIL")
        )
        v.append(
            f"R1 pseudo {pse['mean']:.1f}±{pse['sd']:.1f} vs bar {BAR[sc]:.1f} "
            f"(plain {plain['mean']:.1f}, comparators {[round(c,1) if c else None for c in comps]}, "
            f"sign-agree={same_sign}) -> {verdict}"
        )
    if orc["n"] == 3 and pse["n"] == 3:
        v.append(
            f"oracle>pseudo: {'OK' if orc['mean'] > pse['mean'] else 'VIOLATED (pseudo-labeller is noise or oracle broken)'}"
            f" ({orc['mean']:.1f} vs {pse['mean']:.1f})"
        )
    elif orc["n"]:
        v.append(
            f"oracle: {orc['mean']:.1f} (n={orc['n']}) — stale-label ceiling vs joint {REF[sc]['joint']}"
        )
    return v


def main() -> int:
    pending = 0
    for sc in SCENARIOS:
        s = summarise(sc)
        print(
            f"\n== {sc}  (naive {REF[sc]['naive']}, ER {REF[sc]['er']}, best {REF[sc]['best']}, joint {REF[sc]['joint']}; bar {BAR[sc]:.1f})"
        )
        for label, r in s.items():
            row = " ".join(
                f"s{k}={c['AA']:.1f}/{c['BWT']:+.1f}" if c else f"s{k}=--"
                for k, c in r["cells"].items()
            )
            mean = f"{r['mean']:.2f}±{r['sd']:.2f}" if r["mean"] is not None else "   --  "
            print(f"  {label:15s} n={r['n']} mean={mean:>12s}  {row}")
            if rs := row_summary(r["cells"]):
                print(f"  {'':15s} final row [{rs}]")
            pending += 3 - r["n"]
        for line in gate(sc, s):
            print("  GATE", line)
    log = ROOT / "results/logs_vision/r2_queue.log"
    if log.exists():
        tail = [ln for ln in log.read_text().splitlines() if ln.strip()][-3:]
        print("\nqueue tail:", *tail, sep="\n  ")
        fails = [ln for ln in log.read_text().splitlines() if " FAIL " in ln]
        if fails:
            print("FAILED CELLS:", *fails, sep="\n  ")
    print(f"\ncells pending: {pending}/30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
