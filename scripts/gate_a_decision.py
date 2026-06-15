#!/usr/bin/env python3
"""GATE A: map the pilot diagnosis to the proposed-method decision.

Reads results/pilot/findings_summary.md and prints one of:

    DOCCL       -> proceed with the depth/head-targeted DocCL method
    FALLBACK    -> characterization-only paper (no method)

Revised decision rule (review M4 — only *measurable* branches). The earlier
3-candidate tree (fusion-dominant -> A, layout-drift -> B, per-modality -> C) is
removed because a single-stream LayoutLMv3 cannot produce a separable fusion or
visual-attention signal (review C2). What the corrected instrument CAN show is
*where along depth* forgetting concentrates:

  - forgetting concentrated in the classifier head / late encoder layers
        -> DOCCL (depth/head-targeted consolidation)
  - forgetting uniform across depth, or no concentration (fail to reject H0)
        -> FALLBACK (characterization-only)

Evidence used: the per-component verdict + dominant group (Fisher importance) and,
when present, the displacement-by-depth signal — both written by
``doccl.pilot.analyze``. Output-facing dominant groups are ``classifier`` (and the
``late`` / ``head`` depth buckets).

Exit code 0 always; the decision token is the last stdout line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Dominant signals that indicate an output-facing (head/late) forgetting locus.
HEAD_LATE_SIGNALS = {"classifier", "late", "head"}


def decide(summary_path: Path) -> tuple[str, str]:
    if not summary_path.exists():
        return "FALLBACK", f"findings_summary.md not found at {summary_path}"

    text = summary_path.read_text()
    comp_section = text.split("### Per-component", 1)
    if len(comp_section) < 2:
        return "FALLBACK", "no per-component section in summary"
    comp = comp_section[1]

    rejected = "Verdict" in comp and "reject H0" in comp.split("Verdict")[1] \
        and "fail to reject H0" not in comp.split("Verdict")[1]

    m = re.search(r"[Dd]ominant(?: component)?:\s*\*\*([a-z0-9_]+)\*\*", comp)
    dominant = m.group(1) if m else None

    if rejected and dominant in HEAD_LATE_SIGNALS:
        return "DOCCL", f"head/late-dominant forgetting (dominant={dominant}) -> DocCL"
    if not rejected:
        return "FALLBACK", f"no concentration (fail to reject H0); dominant={dominant!r}"
    return "FALLBACK", (
        f"dominant={dominant!r} is not output-facing; depth/head targeting "
        "not indicated -> characterization-only"
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pilot/findings_summary.md")
    decision, reason = decide(path)
    print(f"[gate_a] {reason}", file=sys.stderr)
    print(decision)


if __name__ == "__main__":
    main()
