# Method Decision Template (W4 deliverable)

> Fill this in at end of Week 4 after the pilot study. Commit to `develop`,
> share link with advisor before the method-decision meeting.

## Pilot summary

**Pilot completed on:** _DATE_  
**Conditions completed:** _4 / 4_  
**Seeds completed per condition:** _3 / 3_  
**Total runs:** _12_  
**Failed/incomplete runs:** _0_

### CL metrics across conditions (mean ± std over 3 seeds)

| Condition | AA | BWT | AF | FWT |
|---|---|---|---|---|
| C1 (text only) | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ |
| C2 (image+layout) | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ |
| C3 (text+layout) | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ |
| C4 (full) | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ | _XX.XX ± X.XX_ |

## Findings

### Finding 1: _\<short title\>_

_e.g._ "C4 (full multimodal) shows ~2× the Fisher drop in fusion layers compared to
C2 and C3 across all task boundaries (p < 0.01, Mann-Whitney U)."

**Evidence:**
- Figure: `results/pilot/figures/fisher_bars.pdf`
- Statistical test: _Mann-Whitney U_ on per-condition Fisher drops in fusion group, _p = X.XX_
- Effect size: _d = X.XX_

### Finding 2: _\<short title\>_

_..._

### Finding 3: _\<short title\>_

_..._

## Hypothesis evaluation

| Hypothesis | Supported? | Evidence |
|---|---|---|
| H_a (fusion forgetting + layout matters) | _yes/no/partial_ | _ref findings_ |
| H_b (uniform forgetting + position drift) | _yes/no/partial_ | _ref findings_ |
| H_c (scenario-dependent patterns) | _yes/no/partial_ | _ref findings_ |
| H₀ (no clear pattern, fallback) | _yes/no_ | _ref findings_ |

## Decision

**Selected method:** _Candidate A / B / C / FALLBACK to characterization-only_

**Rationale (1-2 paragraphs):**

_Explain in plain language why this candidate fits the pilot findings best.
What would the method specifically protect, replay, or route?_

## Implementation plan (W9-11)

### Components to implement

- [ ] _Component 1_ (W9)
- [ ] _Component 2_ (W9-10)
- [ ] _Component 3_ (W10)

### Hyperparameter sweeps

- _Param X_: _values to try_
- _Param Y_: _values to try_

### Ablations planned

1. _Ablation 1: remove component X_
2. _Ablation 2: vary the most-novel hyperparameter_
3. _Ablation 3: comparison to closest baseline_

## Risk: what if selected method underperforms?

(Mandatory section per Decision B in `CLAUDE.md`.)

If at GATE 5 (Week 11) selected method does not beat best baseline by 2+ F1
on 2/3 scenarios, we pivot to:

_e.g._ "Characterization-only paper" — Section 4 (pilot) becomes headline,
Section 5 (method) shrunk to 1 page describing why the natural method
candidate was insufficient.

## Advisor sign-off

- [ ] Advisor reviewed findings memo (`docs/pilot_findings.md`)
- [ ] Advisor reviewed selected candidate sketch
- [ ] Advisor approved decision

**Meeting date:** _DATE_  
**Meeting notes:** _link to notes file or short summary_
