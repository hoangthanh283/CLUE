"""Stale-label audit: refresh counts per CoLaR-R run next to its final AA/BWT.

Usage: uv run python scripts/stale_label_audit.py [results/logs_vision] [results]
One row per run: AA, BWT (from metrics.json when the run finished) and the tokens relabelled
after each task. The oracle row is the stale-label budget; a deployable rule is judged by how
much of it it recovers. Candidate counts (new runs only) print in brackets.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"colar_r\[(?P<mode>\w+)\]: refreshed (?P<docs>\d+) docs, (?P<tok>\d+) tokens relabelled"
    r"(?: \((?P<cand>\d+|n/a) candidates\))?"
)


def refresh_steps(log: Path) -> list[str]:
    steps = []
    for m in LINE.finditer(log.read_text(errors="ignore")):
        cand = f"[{m['cand']}]" if m["cand"] not in (None, "n/a") else ""
        steps.append(f"{m['tok']}{cand}")
    return steps


def final_metrics(run_dir: Path) -> tuple[str, str]:
    f = run_dir / "metrics.json"
    if not f.exists():
        return "run.", "--"
    m = json.loads(f.read_text())
    return f"{m['AA']:.1f}", f"{m.get('BWT', float('nan')):.1f}"


def audit(log_dir: Path, results: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    for log in sorted(log_dir.glob("*colar_r_*.log")):
        aa, bwt = final_metrics(results / log.stem)
        rows.append((log.stem, aa, bwt, " / ".join(refresh_steps(log)) or "-"))
    return rows


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/logs_vision")
    results = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    rows = audit(log_dir, results)
    if not rows:
        sys.exit(f"no colar_r logs under {log_dir}")
    print(f"{'run':<52} {'AA':>5} {'BWT':>6}  tokens relabelled after T1 / T2 / ...")
    for run, aa, bwt, steps in rows:
        print(f"{run:<52} {aa:>5} {bwt:>6}  {steps}")


if __name__ == "__main__":
    main()
