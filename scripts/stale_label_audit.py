"""Stale-label audit: refresh counts per CoLaR-R run and task, parsed from the training logs.

Usage: uv run python scripts/stale_label_audit.py [results/logs_vision]
Prints one row per (run, refresh step): docs refreshed, tokens relabelled, candidates (when
logged). The oracle row is the stale-label budget; a deployable rule is judged by how much of
it it recovers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINE = re.compile(
    r"colar_r\[(?P<mode>\w+)\]: refreshed (?P<docs>\d+) docs, (?P<tok>\d+) tokens relabelled"
    r"(?: \((?P<cand>\d+|n/a) candidates\))?"
)


def audit(log_dir: Path) -> list[tuple[str, str, int, int, int, str]]:
    rows = []
    for log in sorted(log_dir.glob("*colar_r_*.log")):
        for step, m in enumerate(LINE.finditer(log.read_text(errors="ignore")), start=1):
            rows.append(
                (log.stem, m["mode"], step, int(m["docs"]), int(m["tok"]), m["cand"] or "-")
            )
    return rows


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/logs_vision")
    rows = audit(log_dir)
    if not rows:
        sys.exit(f"no colar_r refresh lines under {log_dir}")
    print(f"{'run':<52} {'mode':<8} {'step':>4} {'docs':>5} {'tokens':>7} {'cand':>6}")
    for run, mode, step, docs, tok, cand in rows:
        print(f"{run:<52} {mode:<8} {step:>4} {docs:>5} {tok:>7} {cand:>6}")


if __name__ == "__main__":
    main()
