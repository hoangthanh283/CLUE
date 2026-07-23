# Experiment Analysis and Research Handoff

## Forgetting in Continual Multimodal Document Encoders

**Reporting date:** 2026-07-23  
**Project:** CLUE — continual learning for document information extraction  
**Paper frame:** diagnostic + falsification, AAAI 2027  
**Status:** current evidence consolidated; generalization grid remains the critical path

---

## Document Control

| Field | Value |
|---|---|
| Purpose | Preserve the complete research state, evidence, decisions, and next actions for a new session |
| Evidence window | Experiments and analyses completed through 2026-07-23 |
| Primary scenario | DIL: FUNSD → CORD → SROIE |
| Additional scenarios | CIL-CORD and mixed continual learning |
| Main backbone | LayoutLMv3 |
| Generalization backbones | LiLT, BROS, and BERT text-only |
| Main metrics | Average Accuracy (AA), Backward Transfer (BWT), task trajectories, per-class F1 |
| Mechanistic probes | Output marginals, confusion flow, parameter displacement, CKA, modality ablations |
| Tracking | `STATE.md` and `ROADMAP.md`; `bd` remains write-blocked |
| Repository checkpoint | Branch `doccl`; latest observed pushed commit `5ca8a9c` |

### Evidence labels used in this document

- **Established:** supported by repeated seeds, convergent analyses, or direct controlled comparisons.
- **Provisional:** mechanistically strong but currently instrumented at one seed/backbone.
- **Post-hoc:** discovered after the original analysis plan; useful but not pre-registered.
- **Negative result:** a tested branch that failed its decision criterion and should not be repeated without new evidence.
- **Pending:** required evidence is incomplete or still running.

---

## Executive Decision

The project should remain a **diagnostic + falsification paper**, not pivot back to a
new-method or DIL-SOTA paper.

The current evidence supports the following argument:

1. Catastrophic forgetting is primarily expressed at the shared classifier/readout and in
   late functional representations.
2. In failing methods, old-task predictions undergo a **readout-marginal snap** toward the
   latest task's label distribution. Under-exercised classes such as `KEY` and `HEADER`
   extinguish, while the broadly shared `VALUE` class can appear to recover.
3. Protecting the apparent locus does not remove forgetting. It redirects displacement into
   other parts of the model.
4. Cheap readout corrections, marginal penalties, logit adjustments, read-side blending,
   nullspace projection, and other buffer-free constructions do not recover erased classes.
5. Replay works when it preserves the joint document structure that the task requires:
   feature, position, and label consistency within real documents.
6. CoLaR is the constructive control: compressed per-document latent replay preserves that
   consistency and reaches near-joint performance with materially lower storage than raw replay.
7. The remaining high-value question is whether the mechanism and conclusions generalize
   across seeds, scenarios, and backbones—not whether another small DIL tuning gain can be found.

The best next action is therefore to finish the pre-specified generalization grid and extend
the load-bearing RCA probes beyond LayoutLMv3 seed 42.

---

## Purpose

This document records:

- the research question and locked paper framing;
- all load-bearing empirical results obtained so far;
- the current root-cause analysis;
- hypotheses supported, amended, or refuted;
- negative experiments and branches that are closed;
- known limitations and evidence strength;
- implementation and operational state;
- the exact work that should resume in the next session.

It is a continuation artifact, not a polished paper section. Claims are intentionally bounded
to the evidence currently available.

---

## Research Context

### Core question

Why do continual-learning baselines fail so severely in multimodal document encoders, and is
the failure specifically caused by multimodal pathway interference?

### Locked paper framing

This is a diagnostic and falsification study. Its contribution is not a newly proposed
continual-learning method. The paper explains:

- where forgetting is expressed;
- how the locus changes when it is protected;
- what mechanism best explains the observed class collapse;
- which plausible remedies fail;
- what information successful replay must preserve.

### Current claim set

1. **Readout localization.** In failing methods, classifier-head displacement is much larger
   than trunk displacement, and class-level extinction is visible directly in the readout.
2. **Migration under protection.** Freezing or pinning the head redirects damage into the
   trunk rather than eliminating forgetting.
3. **Buffer-free families fail in the tested design space.** Merge, subspace transfer,
   parametric slots, input-anchored memory, and feature-Gaussian replay have all failed their
   intended retention objective.
4. **Whole-document consistency is necessary for effective replay.** Marginal summaries do
   not preserve the interactions between latent feature, position, and label.
5. **The immediate failure mode is a readout-marginal snap.** On old documents, failing
   models predict a label distribution close to the newest task's marginal rather than the old
   task's own marginal.
6. **Multimodality is not the sole cause.** A text-only BERT control shows the same
   first-boundary retention fraction and class extinction pattern.

Claims 1, 2, 3, and the replay-consistency result have convergent support. The fine-grained
RCA mechanism in claim 5 is still provisional because its full instrumentation is currently
LayoutLMv3 seed 42. Claim 6 is a post-hoc extension across three backbones and three seeds;
BROS remains pending.

---

## Experiment Summary

### Population and setting

The experiments concern token-classification document encoders under continual learning:

- **DIL:** FUNSD → CORD → SROIE.
- **CIL-CORD:** five sequential CORD class partitions.
- **Mixed:** mixed task/domain sequence used by the project grid.

The main model is LayoutLMv3. LiLT and BROS represent alternative multimodal encoders. BERT
is the text-only negative control for the hypothesis that multimodal fusion itself causes the
collapse.

### Major experiment families

| Family | Purpose | Current result |
|---|---|---|
| Baseline mining | Locate when and for which classes forgetting occurs | One-boundary collapse; `KEY`/`HEADER` extinction in failing methods |
| Tier B/C RCA | Test modality, depth, head, and marginal hypotheses | Head/label interference and marginal snap supported; modality and late-depth mechanisms refuted |
| Kill tests | Determine whether the diagnosed symptom is cheaply invertible | All tested cheap corrections failed |
| Replay consistency | Identify what successful memory must preserve | Whole-document consistency is necessary |
| CoLaR | Constructive compressed-replay control | 87.58 AA at 60.4 MB for headline seed-42 setting |
| CoLaR conservation | Test selection/reweighting improvements | No reproducible selection or reweighting lever |
| Cross-backbone control | Test whether failure requires multimodality | Text-only BERT reproduces collapse |
| Full generalization grid | Establish seeds × scenarios × backbones | Pending; current critical path |

---

## Design Notes

### Main DIL task sequence

| Position | Dataset | Relevant schema fact |
|---:|---|---|
| 0 | FUNSD | Contains `HEADER`, `KEY`, `VALUE`, and `O` |
| 1 | CORD | Contains no `KEY` or `HEADER`; approximately 99.8% entity `VALUE`, 0.2% `O` |
| 2 | SROIE | Contains no `HEADER`; approximately 84.8% `O`, 11.5% `VALUE`, 3.7% `KEY` |

This schema asymmetry is central. Classes absent from the current task receive no positive
exercise, while a shared label such as `VALUE` remains active. The changing `O` prevalence
also explains why token/entity metrics can move differently.

### Methods used as diagnostic contrasts

- **Naive:** exposes unprotected sequential learning.
- **LwF:** exposes distillation without real old documents.
- **EWC:** exposes parameter-importance protection and locus migration.
- **ER / ER-CFlat / DER++:** replay controls with real past-task examples.
- **Joint:** non-continual upper reference.
- **CoLaR:** compressed per-document latent replay control.
- **Buffer-free candidate families:** negative controls for whether summaries or parameter
  constraints can replace real document replay.

### Analysis layers

1. Task-level AA/BWT and full accuracy matrices.
2. Per-class F1 trajectories.
3. Boundary-localized forgetting.
4. Old-task prediction confusion flow.
5. Gold and predicted output marginals.
6. Modality ablations and normalized retention.
7. Parameter displacement by module and depth.
8. CKA and functional representation movement.
9. Controlled replay-content interventions.

---

## Hypotheses and Verdicts

| ID | Hypothesis | Verdict | Evidence level |
|---|---|---|---|
| H1 | Forgetting is caused by modality-asymmetric drift | **Refuted as the main mechanism** | Provisional RCA plus cross-backbone post-hoc control |
| H2 | Shared-head/label interference drives class extinction | **Supported** | Strong, convergent |
| H3 | Late multimodal integration layers are the dominant parameter-drift locus | **Refuted** | Provisional RCA |
| H4 | Recency/logit bias causes forgetting | **Supported after amendment to H4′** | Provisional RCA |
| H4′ | Old-input readout marginals snap to the newest task's label marginal | **Selected root-cause description** | Provisional, seed 42/LayoutLMv3 |
| QA | Receipt overlap explains SROIE/FUNSD asymmetry | **Not supported** | Exploratory |
| R1 | Marginal replay statistics are sufficient | **Refuted** | Controlled replay experiments |
| R2 | Whole-document feature-position-label consistency is necessary | **Supported** | Controlled replay experiments |
| M1 | Multimodality is necessary for the collapse | **Refuted** | Post-hoc, 3 backbones × 3 seeds |
| C1 | CoLaR selection or loss reweighting can close the remaining DIL gap | **Not supported** | Multi-seed conservation plus seed-42 isolation tests |

### H4′ in one sentence

After a new task, a failing model's shared classifier maps old-task representations into an
output distribution resembling the newest task's label marginal; absent or weakly exercised
old classes lose effectively all usable probability mass.

---

## Success Criteria Used During the Program

The individual pre-registrations contain exact test-specific bars. The common decision logic
was:

- a proposed correction must materially recover old-task performance, not merely improve the
  newest task;
- acquisition must remain valid;
- extinct `KEY` and `HEADER` classes must recover, rather than a shared `VALUE` class creating
  a misleading aggregate gain;
- improvements close to run-to-run noise or that only redistribute performance between FUNSD
  and SROIE are not sufficient;
- a new branch should not be expanded to three seeds if the seed-42 isolation test is null or
  harmful;
- the generalization claim requires repeated seeds and multiple architectures/scenarios.

---

## Pre-Launch and Measurement Validation

### Validated

- The analysis reads result artifacts from disk rather than depending on online W&B.
- Results include full matrices and per-class F1 where supported.
- The RCA mined **223 per-class runs** and **308 accuracy matrices**.
- ER is not label-degenerate: `KEY` F1 is approximately 87.2 for ER and 89.3 for ER-CFlat.
  Therefore class survival under replay is not a metric artifact.
- The classifier is a plain linear head by design. The previously observed MLP-head
  all-`O` collapse is a known fixed bug and is not part of the current mechanism.
- The CIL analytic-head implementation now disables itself after classifier expansion;
  the resulting `projection_mode=off` run is correctly interpreted as a naive-like control,
  not a valid nullspace-retention test.

### Historical instrumentation gap

Before Tier B instrumentation, **0 of 196 TensorBoard directories** contained the needed
displacement scalars. Parameter-locus claims therefore depend on the purpose-built RCA runs,
not retrospective scalar mining.

### Runtime validity constraints

- Local GPU: RTX 2060, 6 GB.
- Required safety limits: VRAM below 5 GB and host RAM below 14 GB.
- Heavy datasets use batch size 1–2, gradient checkpointing, and zero data-loader workers.
- `fp16` and gradient accumulation configuration fields are not wired into all training loops;
  batch size, checkpointing, and `training.amp` are the reliable controls.
- Earlier low-probability logging under fp16 could floor extinct probabilities; the relevant
  logit-dump and KL underflow issues were fixed before interpreting the final kill tests.

---

## Sample and Exposure Summary

| Analysis | Seeds | Backbones | Scenarios | Interpretation scope |
|---|---:|---|---|---|
| Baseline task/per-class mining | Multiple | Primarily LayoutLMv3 | DIL, CIL-CORD, mixed | Broad descriptive evidence |
| Full Tier B/C RCA instrumentation | 1 | LayoutLMv3 | DIL | Mechanism provisional |
| CoLaR headline | 1 | LayoutLMv3 | DIL | Constructive control |
| ER-CFlat reference | 3 | LayoutLMv3 | DIL | Stable baseline reference |
| Joint reference | 3 | LayoutLMv3 | DIL | Stable upper reference |
| CoLaR conservation | 3 | LayoutLMv3 | DIL | Selection/reweighting negative |
| Cross-backbone naive control | 3 each | LayoutLMv3, LiLT, BERT | DIL | Post-hoc architecture control |
| CIL-CORD architecture comparison | Available slices | LayoutLMv3, BERT | CIL-CORD | Post-hoc supporting control |
| Full B+ grid | Target: 3 | LayoutLMv3, LiLT, BROS, BERT | DIL, CIL-CORD, mixed | Pending generalization |

No causal or significance claim should be inferred from a single-seed comparison. Seed-42
variants are screening experiments unless explicitly identified as multi-seed.

---

## Primary Outcomes

### 1. Baseline forgetting is a one-boundary collapse

For naive DIL, most loss occurs immediately after the first task transition:

- approximate first-boundary drop: **−75.6 points**;
- later apparent recovery: **+7.9 points**.

The later increase is not restoration of extinct old classes. It is largely `VALUE` overlap as
the current-task label marginal rotates. Penalty methods damp the trajectory, replay methods
remain comparatively flat, and prompt-style methods show continued leakage.

### 2. Class extinction is selective and severe

In naive, LwF, and the tested buffer-free chain:

- old `KEY` F1 falls to zero;
- old `HEADER` F1 falls to zero where that class exists;
- `VALUE` can remain around 47 F1 because it is exercised across tasks.

At the final boundary, naive and LwF send all valid old `KEY`/`HEADER` off-diagonal confusion
mass to `VALUE`:

- `KEY`/`HEADER` → `VALUE` share: **1.00**.

Replay methods such as ER and DER++ retain `KEY`, showing that the class is learnable and that
its extinction is specific to continual retention.

### 3. `O` collapse is hidden by entity-only evaluation

For naive and LwF, old-task `O` accuracy reaches **0** at the final state. Entity-level
`seqeval` does not expose this directly. The final diagnosis must therefore retain both entity
F1 and token-level marginal/confusion analyses.

### 4. Failing methods match the latest-task marginal

Cosine similarity between predicted label marginals on old inputs and candidate gold
marginals:

| Method / boundary | Similarity to latest-task marginal | Similarity to old-task marginal | Old-task `O` accuracy |
|---|---:|---:|---:|
| Naive, boundary 1, FUNSD | 0.997 | 0.599 | 0.96 |
| Naive, boundary 2, SROIE | 0.937 | 0.105 | 0.00 |
| LwF, boundary 1 | 0.998 | 0.616 | — |
| LwF, boundary 2, SROIE | 0.925 | 0.105 | — |
| EWC, boundary 2, SROIE | 0.986 | 0.129 | 0.02 |
| ER, boundary 2, SROIE | 0.096 | 1.000 | 0.99 |
| DER++, boundary 2, SROIE | 0.093 | 1.000 | — |
| CoLaR, boundary 2, SROIE | 0.166 | 0.997 | — |

This is the strongest direct evidence for H4′. Failing methods align with the newest label
distribution; replay methods remain aligned with the old task's gold distribution.

### 5. Head displacement dominates in ordinary failing methods

Classifier-head displacement relative to trunk displacement:

| Method | Boundary-level ratios | Interpretation |
|---|---:|---|
| Naive | 8.3×, 16.5× | Strong head localization |
| LwF | 24.2×, 497.9× | Extreme head localization |
| EWC | 0.9–1.5× | Damage migrates when the ordinary locus is protected |

The EWC result is not evidence against the localization finding. It is evidence for a
redistribution law: protecting one locus changes where the model absorbs the required update.

### 6. Raw modality drift does not explain the failure

The original multimodal-correlation idea does not survive controlled analysis:

- raw pathway differences are confounded by different acquisition floors;
- normalized naive retention differs by only **4.7 percentage points** across modality
  ablations;
- text embeddings show the smallest movement in **5 of 6** checks and are never the largest;
- the image-layout pathway drops least;
- LayerNorm movement is **2.4–20.9×** the largest modality-embedding movement and
  **262–10,055×** text-embedding movement.

The collapse is therefore not well described as one modality pathway drifting more than
another.

### 7. Parameter movement is not late-layer concentrated

Normalized displacement is front-loaded:

| Method | Early layers | Late layers |
|---|---:|---:|
| Naive | 0.70 / 0.64 | 0.03 / 0.03 |
| LwF | 0.69 / 0.56 | 0.05 / 0.09 |

This refutes the proposed late-multimodal-integration parameter-drift mechanism. It does not
contradict late functional representation changes: parameter displacement and functional
readout effects are different measurements.

---

## Replay-Consistency Result

### Controlled comparison

| Replay representation | AA | SROIE final | Memory |
|---|---:|---:|---:|
| Whole real latent documents, 5 documents | 87.3 | — | 16 MB |
| Spectral synthetic rank-`r` summary | 41.9 | 3.2 | 0.45 MB |
| Full-dimensional Gaussian summary | 39.4 | 3.0 | 0.39 MB |
| Real-centroid coreset, 4 carriers | 36.7 | 2.9 | 0.48 MB |
| Real centroids, 50 carriers + 16/class | 39.7 | 3.1 | 4.05 MB |

The sharpest control compares:

- **4 whole real documents:** AA **63.8**;
- **4 decoupled real-centroid carriers:** AA **36.7**;
- difference: **+27.1 AA** for preserving whole-document consistency.

### Interpretation

The decisive information is not merely a better estimate of feature or class marginals.
Replay must preserve the joint relationship among:

- token/document features;
- spatial or sequence positions;
- labels within the same document.

This explains why synthetic marginal memories fail while a very small number of coherent real
documents can work.

---

## Constructive Control: CoLaR

### Headline result

CoLaR stores per-document SVD factors for latent replay.

| Measure | Result |
|---|---:|
| Configuration | `k4/d50/r128` |
| AA | **87.58** |
| BWT | **−1.67** |
| Final task vector | **[89.19, 76.30, 97.25]** |
| Stored memory | **60.4 MB** |
| Raw equivalent | **163.4 MB** |
| Compression | **2.7×** |
| Approximate retained variance per document at rank 128 | **95.2%** |

CoLaR is evidence that forgetting is not inevitable and that compressed replay can preserve
the necessary document structure. It is a constructive control, not the paper's claimed new
method.

### Reference performance and available headroom

| Reference | Seeds | Mean AA | BWT | Mean/final task vector |
|---|---:|---:|---:|---|
| ER-CFlat | 3 | **88.418369** | **−1.792** | [86.91, 80.94, 97.41] |
| Joint | 3 | **88.731174** | — | [87.23, 81.82, 97.15] |

The gap from ER-CFlat to joint is only **0.313 AA**. This is approximately the scale where
small DIL tuning changes become difficult to distinguish from seed variation and redistribution
between tasks.

### Conservation and reweighting experiments

| Variant | Seeds | AA | BWT | Final task vector | Verdict |
|---|---:|---:|---:|---|---|
| CoLaR base | 1 headline; 3-seed conservation available | 87.58 headline | −1.67 | [89.19, 76.30, 97.25] | Constructive control |
| Weighted SVD | 1 | 87.0801 | — | [85.90, 78.13, 97.21] | Negative |
| Class + task balanced replay (`p50`) | 1 | 85.889972 | −4.496337 | [86.09, 73.99, 97.59] | Harmful |
| Near task-only balancing (`p1`) | 1 | 87.635141 | −0.625046 | [88.83, 76.72, 97.36] | +0.055 AA; immaterial |
| CoLaR balanced selection, conservation | 3 | approximately 87.8 | — | — | No gain over base |
| K-center selection, conservation | 3 | approximately 87.0 | — | — | Worse than base |

Weighted SVD trades FUNSD for SROIE and lowers AA by about 0.50. Full class/task balancing is
clearly harmful. The isolated task-balancing change improves seed-42 AA by only 0.055 and
remains 0.783 below the three-seed ER-CFlat mean.

**Decision:** do not expand these tuning branches. They show redistribution, not a new
retention mechanism.

---

## Kill-Test Results

### 1. Evaluation-time marginal corrections

Retrained naive smoke reference:

- AA: **39.78**.

Corrections:

| Correction | AA | Old-task AA / key observation | Verdict |
|---|---:|---|---|
| Marginal matching | 38.58 | old AA approximately 11 | Failed |
| Prior-ratio correction | 39.70 | old AA approximately 11 | Failed |
| Per-document EM | 39.60 | old AA approximately 11 | Failed |

Old-class probabilities are already numerically extinguished:

- old `KEY`: approximately **4×10⁻⁶ to 7×10⁻⁶**;
- old `HEADER`: below **7×10⁻⁷**;
- practical saturation threshold checked below **10⁻³**.

The needed information is not recoverable by simple post-hoc prior correction.

### 2. Training-time marginal KL

- `λ=1` AA: **39.14**;
- old `KEY`/`HEADER`: **0**;
- verdict: provisional null.

### 3. Training-time logit adjustment

- `τ=1` AA: **36.9**;
- acquisition guard: passes;
- survival guard: fails.

### 4. Frozen-trunk naive

- AA: **28.3**;
- acquisition: invalid/crippled;
- old-class extinction: reproduced.

Although this run cannot serve as a fair competitive baseline, it rules against the claim
that trunk drift alone is required for extinction.

### 5. Read-side nearest-neighbor blending

| Setting | AA | Final vector | Interpretation |
|---|---:|---|---|
| CoLaR base | 87.58 | [89.19, 76.30, 97.25] | Reference |
| kNN blend, `λ=0.3` | 87.64 | [89.2, 76.5, 97.2] | +0.04; no material signal |
| kNN blend, `λ=1` | 46.24 | — | Destructive |

### 6. Read-side metaplasticity

| Setting | AA | Verdict |
|---|---:|---|
| Control, `m=1` | 85.85 | Below base |
| Stronger, `m=3` | 79.48 | Harmful |

### Consolidated kill-test conclusion

The readout-marginal snap is a useful diagnosis of **what happens**, but it is not an
invertible calibration error. By the end of training, absent classes have lost essentially all
probability mass. Real past-task gradients are required to keep those readout directions
exercised.

Do not build MbPA or continue read-side blending without qualitatively new evidence.

---

## Cross-Backbone Forgetting Analysis

### DIL naive, three seeds per backbone

| Backbone | Task-0 acquisition | After task 1 | Retained | Absolute drop | Later rebound | Old-task final | AA | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LayoutLMv3 | 88.963 | 4.004 | 4.50% | 84.959 | +18.33 | 13.12 | 41.31 | −73.08 |
| LiLT | 80.172 | 3.179 | 3.96% | 76.993 | +15.41 | 11.28 | 40.14 | −70.26 |
| BERT text-only | 61.173 | 2.754 | 4.50% | 58.419 | +13.39 | 10.66 | 39.42 | −59.89 |

Across the nine backbone × seed runs:

- acquisition versus absolute first-boundary drop Pearson correlation:
  **`r = 0.995096`**;
- mean retained fraction: **4.324%**;
- retained-fraction range: **1.543%–6.786%**.

In all three seeds for all three tested backbones:

- final FUNSD `HEADER` F1 = **0**;
- final FUNSD `KEY` F1 = **0**;
- final SROIE `KEY` F1 = **0**.

### Interpretation

The apparent correlation between multimodal capacity and absolute forgetting is largely a
ceiling/floor effect:

- stronger encoders acquire more task-0 performance;
- all tested encoders retain roughly the same tiny fraction after the first transition;
- therefore stronger encoders have more absolute points available to lose.

Multimodality is neither necessary nor sufficient for the collapse. The shared readout reset
is architecture-general across the tested controls. This conclusion is **post-hoc** and must
be labeled as such in the paper until BROS and the full grid are complete.

### Additional scenario checks

#### CIL-CORD naive

| Backbone | Naive AA | Naive BWT | Joint AA |
|---|---:|---:|---:|
| LayoutLMv3 | 18.98 | −93.72 | 33.32 |
| BERT | 19.26 | −93.52 | 32.41 |

The near-identical collapse in multimodal LayoutLMv3 and text-only BERT further weakens a
multimodality-specific explanation.

#### Mixed scenario

| Backbone | Naive AA | Naive BWT | Joint |
|---|---:|---:|---:|
| LayoutLMv3 | 33.85 | −66.89 | 67.51 |
| BERT | 23.37 | −61.28 | seed-42 AA 54.99 |

The BERT joint multi-seed aggregation was not complete at the time of this handoff. Do not
replace the seed-42 value with an inferred mean.

---

## Nullspace Analytic Control

The CIL implementation disables the analytic head after classifier expansion. With
`projection_mode=off`, no valid retention mechanism remains, so this run is a naive-like
overwrite control.

### Final result

- AA: **18.98**;
- BWT: **−92.85**.

### Task trajectory

| Training state | Evaluation vector |
|---|---|
| After task 0 | [95.16] |
| After task 1 | [0, 91.41] |
| After task 2 | [0, 0, 94.12] |
| After task 3 | [0, 0, 0, 90.71] |
| After task 4 | [0, 0, 0, 0, 94.90] |

This is pure latest-task overwrite. It is a decisive no-go for the current implementation and
must not be described as evidence that a functioning analytic nullspace head failed; the
retention mechanism was disabled by the required CIL expansion guard.

---

## Guardrail Metrics and Checks

### Scientific guardrails

| Guardrail | Result |
|---|---|
| Current-task acquisition remains valid | Passed for most kill tests; failed for frozen-trunk naive |
| Old classes recover rather than only `VALUE` | Failed for cheap correction methods |
| Improvement exceeds seed/noise scale | Failed for task-only CoLaR balancing and kNN blend |
| Storage accounting is explicit | Passed for CoLaR; 60.4 MB vs 163.4 MB raw |
| Negative controls retain learnability | Passed; replay controls preserve `KEY` |
| Multimodal-specific claim has text-only control | Passed post-hoc for BERT; BROS pending |
| Generalization beyond seed 42 | Partial; mechanism instrumentation still pending |

### Engineering guardrails

- No existing configuration files were modified; new methods use new config options.
- The live package is `doccl/`; the dead `src/` tree must remain ignored.
- CoLaR balancing changes preserve the original tensor store and reported byte accounting.
- Replay sampling is exactly task-stratified in the new isolation implementation.
- Relevant focused tests, Ruff, and Black checks passed.
- A full fast no-GPU suite passed with **383 tests** for the CoLaR balancing work.
- Repository-wide lint/format checks still expose pre-existing debt:
  **270 Ruff findings** and **48 Black-format files** were observed. These are not attributable
  to the focused experiment changes and should not be bulk-fixed during research runs.

---

## Segment Results

### By method family

| Family | Characteristic trajectory | Mechanistic reading |
|---|---|---|
| Naive | Immediate one-boundary collapse | Shared head follows current task |
| LwF | Same class extinction despite distillation | No real old-document gradients |
| EWC | Damped/redistributed damage, poor later acquisition | Protected locus pushes change elsewhere |
| Replay | Flat old-task retention | Old head directions remain exercised |
| Prompt/buffer-free candidates | Ongoing leak or collapse | Stored summaries omit required joint structure |
| CoLaR | Near-joint retention | Compressed whole-document consistency is sufficient |

### By class

| Class | Behavior | Explanation |
|---|---|---|
| `HEADER` | Extinguishes when absent from later tasks | No positive exercise; absorbed into `VALUE` |
| `KEY` | Extinguishes in failing methods | Under-exercised and task-asymmetric |
| `VALUE` | Survives or appears to recover | Shared across datasets and dominates CORD |
| `O` | Can collapse completely on old tasks | Current label marginal and entity-only metric masking |

### By backbone

The retained fraction after the first boundary is remarkably stable despite large differences
in acquisition. This favors an architecture-general shared-head explanation over a
multimodal-pathway explanation.

---

## Data Quality and Limitations

1. **Mechanism instrumentation is n=1.** The strongest marginal, confusion, displacement, and
   CKA evidence is LayoutLMv3 seed 42. It is internally convergent but must be generalized.
2. **Cross-backbone analysis is post-hoc.** The BERT negative control is compelling, but the
   test was not part of the original pre-registration.
3. **BROS is pending.** Do not claim all four planned backbones support the result.
4. **Scenario coverage is incomplete.** DIL has the deepest analysis. CIL-CORD and mixed
   provide supporting evidence but need the full grid.
5. **Single-seed variant deltas are screening evidence.** CoLaR-CB, weighted SVD, kNN, and
   metaplasticity values should not be presented as population estimates.
6. **EWC has acquisition confounding.** Its later marginal snap coexists with poor SROIE
   acquisition (41.9 versus 82.5 for the reference comparison), so it is best used as a
   migration example rather than a clean performance comparison.
7. **Metric choice matters.** Entity-only F1 can hide total `O` collapse and apparent
   `VALUE`-driven recovery.
8. **Memory comparisons require representation context.** A smaller marginal summary is not a
   fair substitute if it destroys the joint document structure required by the task.
9. **Local resource contention affected wall-clock time.** It does not invalidate completed
   metrics, but concurrent GPU and dataset jobs can cause stalls or OOM pressure.
10. **The joint upper reference is not an absolute oracle.** It is the non-continual training
    reference under the current protocol, not a bound on all possible models.

---

## Root-Cause Interpretation

### Selected causal account

The best current explanation is:

1. The task sequence changes the available label support sharply.
2. New-task optimization drives the shared linear classifier toward the current task's label
   marginal.
3. Classes absent from the current task receive no positive gradient and their logits become
   numerically negligible.
4. Old inputs are then decoded according to the newest task's class geometry, most often as
   `VALUE` or according to the newest `O` prevalence.
5. Readout-local protection can reduce direct head movement, but the optimization burden moves
   into the trunk.
6. Post-hoc correction fails because the old class signal has already been erased from the
   learned readout probabilities.
7. Real replay succeeds because it continually supplies correctly coupled old features,
   positions, and labels, grounding old readout directions during optimization.

### What is causal and what is symptomatic

- The **marginal snap** is a precise observable failure mode.
- The **class-asymmetric task sequence** supplies the pressure.
- The **shared readout** is the ordinary visible locus.
- The inability of cheap corrections to work shows the snap is not merely a reversible prior
  mismatch.
- Replay's success identifies the missing ingredient: old, jointly consistent supervised
  evidence during training.

### What the data do not support

- a single drifting visual/layout pathway as the cause;
- dominant late-layer parameter drift;
- simple logit calibration as a remedy;
- marginal Gaussian or centroid memories as substitutes for documents;
- a useful read-side kNN signal;
- further DIL-only selection/reweighting tuning as a high-value research direction.

---

## Decision Log

| Date/phase | Decision | Evidence |
|---|---|---|
| Paper reframe | Lock diagnostic + falsification framing | Multiple method families failed; RCA and constructive control form coherent contribution |
| RCA selection | Use H4′ readout-marginal snap as provisional root cause | Marginal cosine, confusion flow, extinction, replay contrast |
| Modality hypothesis | Reject multimodality-specific mechanism | Normalized ablations, displacement, BERT text-only control |
| Cheap correction chain | Close | Marginal, prior-ratio, EM, KL, logit-adjust, frozen-trunk tests all fail |
| Read-side retrieval | Do not build MbPA | `λ=0.3` gives only +0.04 AA; `λ=1` is destructive |
| CoLaR role | Keep as constructive control | 87.58 AA at 60.4 MB while preserving whole-document consistency |
| Weighted SVD | Stop | 87.0801 AA; task redistribution only |
| Class-balanced CoLaR | Stop | 85.89 AA; clearly harmful |
| Task-balanced CoLaR | Do not expand | +0.055 AA at one seed; below SOTA and noise-scale |
| Nullspace analytic CIL | Close current branch | Mechanism disabled after expansion; control becomes pure overwrite |
| DIL SOTA chase | Stop | ER-CFlat-to-joint headroom only 0.313 AA |
| Critical path | Finish B+ generalization grid | Needed to upgrade provisional RCA to paper-grade general claim |

---

## Post-Test Actions

### Required next-session work

1. **Check repository and job state before launching anything.**
   - Read `STATE.md`, `ROADMAP.md`, and running process/GPU status.
   - Do not start a second dataset-building job.
   - Respect VRAM <5 GB and RAM <14 GB.

2. **Finish the B+ generalization grid.**
   - Three seeds.
   - Three scenarios: DIL, CIL-CORD, mixed.
   - Four backbones: LayoutLMv3, LiLT, BROS, BERT.
   - Local BERT slice and rented LiLT/BROS execution were the active operational split.

3. **Recompute the cross-backbone table after all artifacts arrive.**
   - First-boundary retained fraction.
   - Acquisition versus absolute drop correlation.
   - Final `KEY`/`HEADER` extinction.
   - AA/BWT by scenario and method.
   - Mark the existing analysis as post-hoc.

4. **Generalize the load-bearing RCA probes.**
   Priority measurements:
   - old-input predicted marginal versus old/new gold marginals;
   - `KEY`/`HEADER` extinction and `O` accuracy;
   - confusion flow into `VALUE`;
   - classifier-to-trunk displacement ratio;
   - selected CKA/functional probes only where they adjudicate a live hypothesis.

5. **Update paper and thesis narrative.**
   - Lead with diagnosis and falsification.
   - Include Finding 3b, the +27.1 AA whole-document consistency control.
   - Present CoLaR as constructive control, not the headline method.
   - Add the text-only architecture control.
   - Preserve the n=1 caveat for full RCA instrumentation.
   - Correct all stale LexSlot prose to describe its actual 200-exemplar buffer setting.

6. **Regenerate analysis artifacts after the grid.**

   ```bash
   uv run python scripts/analyze_results.py --source local
   uv run python scripts/ingest_to_thesis.py
   cd thesis && latexmk -xelatex main.tex
   ```

7. **Update continuation files and push.**
   - Record new results in `STATE.md` and ordered work in `ROADMAP.md`.
   - Use `bd` only if its schema-migration write block has been resolved.
   - Run the required quality gates for code changes.
   - Pull with rebase, push, and verify the branch is up to date.

### Branches explicitly closed

Do not repeat or tune these without a new mechanistic reason:

- weighted-SVD CoLaR;
- CoLaR class balancing;
- CoLaR task balancing;
- CoLaR kNN read-side blending;
- metaplastic read-side control;
- marginal matching/prior-ratio/per-document EM;
- current marginal-KL and logit-adjust formulations;
- MbPA;
- current CIL nullspace-analytic branch;
- further DIL-only SOTA tuning.

### Optional work after the critical path

Only if the generalization grid creates a clear need:

- weight-geometry probes to distinguish directional reset from scalar movement;
- a broader marginal-KL sweep, clearly labeled unregistered;
- int8 CoLaR factors, estimated around 30 MB at rank 128;
- additional CoLaR seeds/scenarios where needed for the constructive-control claim.

---

## Session Resume Checklist

Use this order at the start of the next session:

1. Read this file.
2. Read `STATE.md` and `ROADMAP.md` for changes after 2026-07-23.
3. Run `git status --short --branch` and inspect recent commits.
4. Check active training processes, GPU memory, host RAM, and free disk.
5. Inventory `.done` markers and result directories; rely on the grid's resume behavior.
6. Continue the incomplete BERT/LiLT/BROS slices rather than duplicating completed jobs.
7. When the grid is complete, regenerate the summary CSVs before interpreting results.
8. Upgrade or weaken claims based on the completed multi-seed evidence.
9. Update this handoff or create a dated successor.

---

## Appendix A — Metric Definitions

### Average Accuracy

For final continual-learning state \(T\) and tasks \(1,\ldots,T\):

\[
\mathrm{AA} = \frac{1}{T}\sum_{i=1}^{T} a_{T,i},
\]

where \(a_{T,i}\) is performance on task \(i\) after training through task \(T\).

### Backward Transfer

\[
\mathrm{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}(a_{T,i} - a_{i,i}).
\]

More negative BWT indicates stronger forgetting relative to performance when each task was
first learned.

### First-boundary retained fraction

\[
\mathrm{Retention}_{0\rightarrow1}
= \frac{a_{1,0}}{a_{0,0}}\times 100\%.
\]

This normalization is essential for backbone comparisons because acquisition floors differ.

### Absolute first-boundary drop

\[
\Delta_{0\rightarrow1} = a_{0,0} - a_{1,0}.
\]

Absolute drop is strongly correlated with acquisition and should not be interpreted alone as
evidence that a stronger or multimodal backbone forgets more intrinsically.

### Marginal-snap similarity

Cosine similarity is computed between:

- the predicted label-frequency vector on old-task inputs after a boundary; and
- either the old task's gold marginal or the latest task's gold marginal.

A failing run is snap-like when similarity to the latest marginal is much larger than
similarity to the old marginal.

### Head/trunk displacement ratio

The normalized classifier-head displacement divided by normalized trunk displacement.
Ratios much greater than one indicate ordinary head localization; ratios near one under a
protection method can indicate migration rather than elimination.

---

## Appendix B — Evidence and Artifact Map

### Primary narrative documents

- `docs/FINDINGS_ANALYSIS_PAPER_2026-07.md` — full paper argument spine.
- `docs/RCA_FORGETTING_BASELINES_2026-07.md` — Tier A/B/C root-cause analysis.
- `docs/RCA_KILLTESTS_PREREG_2026-07.md` — pre-registered kill tests and criteria.
- `docs/THESIS_METHOD_CHAPTER_REFRAME_2026-07.md` — required thesis reframing.
- `STATE.md` — current operational and result snapshot.
- `ROADMAP.md` — ordered remaining work.

### Main machine-readable RCA artifacts

- `results/rca/a1_per_class_ledger.csv`
- `results/rca/a1_class_survival.csv`
- `results/rca/a2_trajectories.csv`
- `results/rca/c_confusion_flow.csv`
- `results/pivot_AA.csv`

### Key result directories

- `results/dil_colar_cb_seed42_k4_d50_r128_p50/`
- `results/cil_cord_nullspace_analytic_seed42_off/`
- backbone/scenario result directories under `results/`

Use the result JSON and matrix files as the source of truth if prose and generated tables ever
disagree.

---

## Appendix C — Implementation Checkpoint

Relevant pushed commits observed before this handoff:

| Commit | Purpose |
|---|---|
| `b505fd5` | Add CoLaR weighted SVD |
| `1e044e2` | Add RCA-balanced CoLaR |
| `7107749` | Stratify CoLaR replay |
| `71a231c` | Add nullspace-analytic method with hierarchical lexical memory |
| `02271dd` | Auto-disable analytic head for CIL scenarios |
| `5ca8a9c` | Record BERT-slice operational state and sweep-up chain |

The CoLaR implementation work includes:

- inverse-frequency class-balanced replay CE;
- task-balanced replay-loss scaling by `task_id`;
- exact task-stratified replay sampling;
- unchanged tensor-store and memory-byte accounting.

No existing file needs to be deleted to continue this research.

---

## Final Research Takeaway

The useful result is not that multimodal continual learning forgets badly; that was already
visible in aggregate metrics. The contribution is the chain of falsification:

- collapse is class-asymmetric and immediate;
- the ordinary visible locus is the shared readout;
- the output marginal on old documents snaps to the latest task;
- protecting that locus moves the damage;
- cheap marginal/readout remedies cannot recover extinguished classes;
- marginal memory summaries fail;
- coherent old-document supervision succeeds, even when compressed.

The next session should test the breadth of this chain, not search for another small DIL
increment.
