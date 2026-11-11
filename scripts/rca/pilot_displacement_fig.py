"""Supplement figure "Where forgetting lives": doccl.pilot.analyze.plot_displacement_bars on the
document-pilot conditions, dead runs excluded by the same rule as pilot_depth_profile.py
(final AA < 1.0 or all-zero retention matrix). JSON only; writes one PDF.

    uv run python scripts/rca/pilot_displacement_fig.py [out.pdf]
"""

import sys
from pathlib import Path

from pilot_depth_profile import dead_reason

from doccl.pilot.analyze import load_pilot_results, plot_displacement_bars, to_long_dataframe

DOC_CONDITIONS = (
    "c4_full",
    "c1_text",
    "c3_no_image",
    "c2_no_text",
    "cb_bert",
    "cl_lilt",
    "cr_bros",
)

out = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_cvpr/figures/pilot_displacement_bars.pdf")
runs = [r for r in load_pilot_results(Path("results/pilot")) if r["condition"] in DOC_CONDITIONS]
live = [r for r in runs if not dead_reason(r)]
for r in runs:
    if dead_reason(r):
        print(f"excluded {r['condition']} seed {r['seed']}: {dead_reason(r)}")
df = to_long_dataframe(live)
sub = df[(df["metric"] == "displacement_depth") & (df["boundary"] == "0_to_1")]
print("live seeds per condition:", sub.groupby("condition")["seed"].nunique().to_dict())
plot_displacement_bars(df, out)
