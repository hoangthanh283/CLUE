#!/usr/bin/env python3
"""GATE A: map the pilot's dominant-component verdict to the proposed method.

Reads results/pilot/findings_summary.md, extracts the per-component (C4) verdict
and dominant Fisher group, and prints one of:

    A   -> Candidate A (doccl_a, H-LoRA)         [fusion-dominant]
    B   -> Candidate B (doccl_b, Layout-EWC)     [layout-position drift]
    C   -> Candidate C (doccl_c, Routed Prompts) [scenario-dependent per-modality]
    FALLBACK                                      [fail to reject H0 / no data]

Decision rule (RUNBOOK / CLAUDE.md):
  - fusion-dominant forgetting          -> A
  - 2D layout-position drift dominates  -> B
  - scenario-dependent per-modality     -> C
  - no clear pattern (fail to reject H0)-> characterization-only fallback

Dominant Fisher group names come from doccl/eval/fisher.py param_groups:
  text_word_embed, layout_2d_pos_embed, image_patch_embed, text_attn, ffn,
  classifier, other.

Mapping group -> candidate:
  layout_2d_pos_embed         -> B   (layout-position drift)
  ffn / text_attn / other     -> A   (fusion / cross-modal mixing in the trunk)
  image_patch_embed           -> C   (visual modality dominates -> routed prompts)
  text_word_embed             -> C   (single-modality dominance -> routed prompts)
  classifier                  -> A   (head-level; treat as fusion/trunk default)

Exit code 0 always; the decision is on stdout (last line = the token).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

GROUP_TO_CANDIDATE = {
    "layout_2d_pos_embed": "B",
    "ffn": "A",
    "text_attn": "A",
    "other": "A",
    "classifier": "A",
    "image_patch_embed": "C",
    "text_word_embed": "C",
}


def decide(summary_path: Path) -> tuple[str, str]:
    if not summary_path.exists():
        return "FALLBACK", f"findings_summary.md not found at {summary_path}"

    text = summary_path.read_text()

    # Per-component verdict line: "Verdict: **reject H0**." or "fail to reject H0"
    comp_section = text.split("### Per-component", 1)
    if len(comp_section) < 2:
        return "FALLBACK", "no per-component section in summary"
    comp = comp_section[1]

    reject = "reject H0" in comp and "fail to reject H0" not in comp.split("Verdict")[1] \
        if "Verdict" in comp else False

    m = re.search(r"Dominant component:\s*\*\*([a-z0-9_]+)\*\*", comp)
    dominant = m.group(1) if m else None

    if not reject or dominant is None:
        return "FALLBACK", (
            f"fail to reject H0 (no clear dominant component); dominant={dominant!r}"
        )

    candidate = GROUP_TO_CANDIDATE.get(dominant)
    if candidate is None:
        return "FALLBACK", f"unmapped dominant group {dominant!r}"
    return candidate, f"dominant={dominant} -> Candidate {candidate}"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/pilot/findings_summary.md")
    decision, reason = decide(path)
    print(f"[gate_a] {reason}", file=sys.stderr)
    print(decision)


if __name__ == "__main__":
    main()
