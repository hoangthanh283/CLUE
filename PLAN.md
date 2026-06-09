# PLAN.md — DocCL Master's Thesis & AAAI 2027 Submission

**Last updated:** May 6, 2026
**Author:** Thanh Hoang
**Repository:** doccl/
**Paper type:** Diagnostic + Remedy (insight-driven, method emergent from pilot study)

---

## Executive Summary

This section is the high-level narrative for anyone (advisor, Claude Code, future-self) opening this file fresh. Detailed weekly tasks come after.

### The Big Picture

```
TODAY                       DEADLINE                  RESULT
May 6, 2026                 Aug 8, 2026               Dec 2026
    │                           │                         │
    │◄────── 14 weeks ─────────►│                         │
    │                           │                         │
    │   READ + PILOT + BUILD    │   REVIEW PHASE          │
    └───────────────────────────┴─────────────────────────┘
              Phase 1-5              Rebuttal Oct 2026
```

**Submission target:** One paper to AAAI 2027 (deadline ~Aug 8, 2026, conference Feb 2027).

**Paper structure (diagnostic + remedy):**

```
characterize → hypothesize → exploit → validate
```

The paper:
1. **Characterizes** how catastrophic forgetting manifests in multimodal document encoders (LayoutLMv3) compared to unimodal/modality-ablated baselines
2. **Identifies** which architectural components forget most
3. **Proposes** a method whose design is *justified by pilot findings* (selected from 3 pre-sketched candidates)
4. **Validates** on FCS-CL benchmark (FUNSD/CORD/SROIE) across 3 scenarios

**Paper format:** AAAI standard (7 pages content + 2 pages references).

### Why This Approach (Phrasing 3)

We chose **insight-driven framing** over pure method paper because:
1. **Scientific value:** characterization findings are publishable independent of method
2. **Risk mitigation:** if method fails, pilot study + benchmark is still publishable (Zhai 2023 style)
3. **Stronger contribution:** "we found X about LayoutLMv3 forgetting" is harder to attack than "our method works"
4. **NeurIPS extension natural:** characterization findings drive NeurIPS theoretical analysis

### The Five Phases (14 Weeks)

| Phase | Weeks | Dates | Focus | Key Output |
|---|---|---|---|---|
| 1 — Infrastructure + Reading | 1-2 | May 6-19 | Repo, datasets, scenarios, pre-read 3 papers, sketch 3 candidates | End-to-end pipeline + GATE 1 |
| 2 — Pilot Study | 3-4 | May 20 - Jun 2 | 4 conditions, CKA + Fisher analysis, method selection | Characterization findings + GATE 2 (method choice) |
| 3 — Core Baselines | 5-6 | Jun 3-16 | EWC, LwF, ER, DER++ | 6 methods × 3 scenarios + GATE 3 |
| 4 — Advanced Baselines | 7-8 | Jun 17-30 | L2P, DualPrompt, CODA-P, O-LoRA | 10 methods × 3 scenarios + GATE 4 |
| 5 — Selected Method + Writing | 9-14 | Jul 1 - Aug 11 | Implement selected candidate, run experiments, write paper, submit | Submitted paper + GATE 5 |

### The Three Novel Contributions

1. **First per-component forgetting characterization** of LayoutLMv3 (and modality-ablated variants) using CKA + Fisher methodology adapted from Ramasesh 2021 — extends Zhai 2023 from MLLMs to document encoders.

2. **Method selected from 3 candidates based on pilot findings:**
   - Candidate A: LAPP + H-LoRA (if fusion forgetting + layout matters)
   - Candidate B: Layout-Protected Regularization (if uniform + position drift)
   - Candidate C: Modality-Routed Prompts (if scenario-dependent)

3. **FCS-CL Benchmark** — First public CL benchmark on FUNSD/CORD/SROIE with reproducible splits, 3 scenarios, 11-method baseline grid, fixed OCR.

### The Eleven Methods (Final List)

| # | Method | Family | Week | Reference |
|---|---|---|---|---|
| 1 | Naive | Lower bound | W2 | — |
| 2 | Joint | Upper bound | W2 | — |
| 3 | EWC | Regularization | W5 | Kirkpatrick PNAS 2017 |
| 4 | LwF | Regularization | W5 | Li & Hoiem ECCV 2016 |
| 5 | ER | Replay | W6 | Rolnick NeurIPS 2019 |
| 6 | DER++ | Replay | W6 | Buzzega NeurIPS 2020 |
| 7 | L2P | Prompt | W7 | Wang CVPR 2022 |
| 8 | DualPrompt | Prompt | W8 | Wang ECCV 2022 |
| 9 | CODA-Prompt | Prompt | W8 | Smith CVPR 2023 |
| 10 | O-LoRA | LoRA | W8 | Wang EMNLP-F 2023 |
| 11 | **Selected (ours)** | **Hybrid** | **W10-11** | **— (this paper)** |

### The Five Critical Gates

```
Week 2 ──► GATE 1: Are scenarios + harness ready?
               Test: Joint − Naive gap ≥ 10 F1 on ≥ 2/3 scenarios
                     CKA + Fisher implementations validated
               PASS → enter pilot study
               FAIL → fix infrastructure

Week 4 ──► GATE 2: Method Selection (CRITICAL DECISION POINT)
               Pilot study complete. Select 1 of 4 paths:
                 - Pattern matches H_a (fusion focus) → Candidate A (LAPP + H-LoRA)
                 - Pattern matches H_b (uniform + drift) → Candidate B (Layout-Protected EWC)
                 - Pattern matches H_c (scenario-dep) → Candidate C (Modality-Routed)
                 - No clear pattern → Pivot to characterization-only paper
               
               This decision LOCKS method until end of W11.

Week 6 ──► GATE 3: Do simple baselines work correctly?
               Test: Best baseline ≥ Naive + 5 F1 on ≥ 2 scenarios
               PASS → continue Phase 4
               FAIL → debug against author code

Week 8 ──► GATE 4: Do advanced baselines work correctly?
               Test: ≥ 1 prompt method and O-LoRA reach ER-level
               PASS → continue Phase 5
               FAIL → debug prompt forward pass / LoRA orthogonality

Week 11 ──► GATE 5: Method Performance Decision
               Branch A: Selected method beats best baseline ≥ 2 F1 on ≥ 2 scenarios
                         → Method paper (full strength)
               Branch B: Selected method ties best baseline (within 2 F1)
                         → Method paper with "competitive efficiency" framing
               Branch C: Selected method underperforms > 2 F1
                         → Pivot to characterization-only paper (FALLBACK)
               
               ALL THREE BRANCHES → publishable at AAAI.
```

### What Happens After Submission

```
Aug 9, 2026   : Submit paper + anonymized code
Sep-Oct 2026  : Initial review phase
Oct 2026      : Rebuttal week
Oct-Nov 2026  : Thesis writing (paper content → thesis Ch 4-5)
Dec 2026      : AAAI 2027 notification + thesis defense
              ↓
Dec 2026 → May 2027: Build NeurIPS extension
May 2027      : Submit NeurIPS 2027 (full version)
```

### Compute Budget Summary

```
Total expected: ~340 GPU-hours on Vast.ai RTX 4090
At interruptible price ~$0.35/h: ~$120
Plus on-demand for final runs + buffer: ~$200-250 total
Within $250-300 budget set initially
```

### Risk Posture

The plan is designed to **always produce a submittable paper**, even if multiple things go wrong:
- If pilot reveals no clear pattern → characterization-only paper (still publishable)
- If selected method fails → pivot to characterization + benchmark paper (Branch C)
- If advanced baselines fail → submit with 6 baselines (still rigorous)
- If compute runs out → drop seed count to 2 for non-essential ablations
- If advisor unavailable W13 → backup reviewer = senior PhD lab mate

The only true blocker would be infrastructure failure in W1-2. Hence those weeks are heavily front-loaded with smoke tests and verification gates.

---

## North Star

| Milestone | Date | Status |
|---|---|---|
| Project kick-off | May 6, 2026 | ✅ Done |
| Reading list complete | May 13, 2026 | ⏳ |
| Pilot study complete + method selected | Jun 2, 2026 | ⏳ |
| All baselines complete | Jun 30, 2026 | ⏳ |
| Selected method complete | Jul 21, 2026 | ⏳ |
| AAAI 2027 submission | ~Aug 8, 2026 | ⏳ |
| Thesis writing | Oct-Nov 2026 | — |
| Thesis defense | Dec 2026 | — |
| AAAI 2027 notification | Dec 2026 | — |
| NeurIPS 2027 submission | ~May 2027 | — |

**Defense constraint:** thesis Chapters 4-5 are based on AAAI submission. Even if AAAI rejects, the work stands as thesis content.

---

## Goals & Non-Goals

### Goals (this plan delivers)
1. **Pilot study findings** characterizing per-component forgetting in LayoutLMv3 vs. baselines
2. **Selected method** justified by pilot (one of 3 candidates)
3. **Working AAAI 2027 paper** with 11 methods × 3 scenarios × 3 seeds, statistical tests, ablations
4. **FCS-CL benchmark** as reproducible artifact
5. **Generalization study** on LiLT + BROS (parallel with writing)
6. **Thesis Chapters 4-5** based on AAAI submission

### Non-Goals (explicitly deferred to NeurIPS extension)
- DSR (Document-aware Structural Replay)
- TIL (Task-Incremental Learning) scenario
- LayoutLMv3-large experiments
- Multilingual CL on XFUND/LayoutXLM
- Theoretical forgetting bounds
- DocVQA, RVL-CDIP datasets
- Production deployment study (latency, memory profiling)
- Pre-locked method choice (method emerges from pilot)
- Entity Linking (mentioned briefly as future work)
- JSON generation framing (architecturally incompatible with chosen backbones)

---

## Compute Budget (Detailed)

| Phase | Runs | GPU-hours | Cost (interruptible $0.35/h) |
|---|---|---|---|
| Setup + smoke (W1-2) | — | 10 | $4 |
| Pilot study (W3-4) | 4 conditions × 3 seeds × 3 tasks = 36 | 30 | $11 |
| Core baselines (W5-6) | 6 methods × 3 scenarios × 3 seeds = 54 | 54 | $19 |
| Advanced baselines (W7-8) | 4 methods × 3 scenarios × 3 seeds = 36 | 43 | $15 |
| Selected method + ablations (W9-11) | ~57 | 74 | $26 |
| Generalization (W12-13, parallel writing) | 6 methods × 3 scenarios × 3 seeds × 2 backbones = 108 | 130 | $46 |
| Final reruns (W13) | 30 | 30 | $11 |
| **Total** | **~321** | **~371** | **~$132** |
| Buffer (failed runs, extras) | | | $100 |
| **Total expected** | | | **$200-250** |

Within $250-300 budget set initially.

**Vast.ai instance specs:**
- GPU: RTX 4090 24GB
- Disk: ≥ 50GB (datasets + checkpoints)
- Verified host
- Image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`
- Use **interruptible** for exploration; **on-demand** for final result runs.

---

## Phase 1 — Infrastructure + Reading + Candidates (Weeks 1-2: May 6-19)

### Week 1: May 6-12 — Foundation

**Goal:** Repo working, single-task FUNSD baseline, 3 papers read, candidates sketched.

#### Day 1-2 (May 6-7) ✅ Done as of skeleton commit
- [x] Project skeleton committed
- [x] `ContinualMethod` interface defined
- [x] `LayoutLMv3Wrapper` with extension points
- [x] `CLMetricsTracker` with AA/BWT/FWT
- [x] Hydra config structure
- [x] Smoke tests
- [x] CLAUDE.md + PLAN.md committed

#### Day 3 (May 8) — **READING + sketching candidates**
- [ ] **Start reading: Ramasesh et al. ICLR 2021 (~2 hours)**
  - Abstract + Section 3 (Methodology) + Section 4 (Findings)
  - Take notes on per-layer forgetting methodology
- [ ] Sketch **Candidate A (LAPP + H-LoRA)** on paper:
  - Architecture diagram
  - Loss formulation
  - Implementation outline (2 pages)

#### Day 4 (May 9) — **READING + sketching**
- [ ] **Read: Zhai et al. 2023 (~2 hours)**
  - Abstract + Section 3 (EMT) + Section 4 (Findings)
  - Note: which metrics adaptable to LayoutLMv3?
- [ ] Sketch **Candidate B (Layout-Protected EWC)** on paper:
  - Modified EWC formulation
  - Per-parameter-group importance weights
  - Implementation outline (1.5 pages)

#### Day 5 (May 10) — **READING + sketching**
- [ ] **Read: Kornblith et al. ICML 2019 (~2 hours)**
  - Abstract + Section 2 (CKA definition) + Section 3 (properties)
  - Critical: linear CKA formula for implementation
- [ ] Sketch **Candidate C (Modality-Routed Prompts)** on paper:
  - Three sub-pools architecture
  - Router network design
  - Implementation outline (2 pages)

#### Day 6-7 (May 11-12) — Setup + first run
- [ ] Setup conda env locally on RTX 2060 machine
- [ ] Run `pytest tests/test_smoke.py -v -m "not slow"` — all pass
- [ ] First single-task run: `python scripts/train.py method=naive scenario=single_funsd seed=42`
  - **Acceptance criterion:** F1 on test set ≥ 88 after 10 epochs
- [ ] Setup Vast.ai account, verify GPU works
- [ ] Document instance setup in `scripts/setup_vastai.sh`

**Verification questions to answer (before W2):**
- What does CKA measure? How is it different from cosine similarity between weights?
- How does Ramasesh measure forgetting per-layer?
- What metrics does Zhai use for MLLMs? Adaptable to LayoutLMv3?

**Week 1 deliverables:**
- ✅ Skeleton repo with all infrastructure
- ✅ Single-task FUNSD F1 ≥ 88 reproducibly
- ✅ Vast.ai workflow documented
- ✅ 3 papers read with notes
- ✅ 3 candidate methods sketched on paper

**Week 1 risks:**
- HF dataset download fails → mirror to local disk
- LayoutLMv3 OOM on RTX 2060 → use batch_size=2, gradient_accumulation_steps=4
- W&B sync issues → use `wandb.init(mode="offline")` and sync later

---

### Week 2: May 13-19 — All datasets + scenarios + CKA/Fisher implementation

**Goal:** All 3 datasets + 3 scenarios + pilot infrastructure (CKA, Fisher, modality ablation) ready.

#### Day 1-3 (May 13-15) — Datasets
- [ ] Implement `CORDDataset` with 30 fine-grained classes (61 BIO tags)
- [ ] Implement `SROIEDataset` (4 fields, 9 BIO tags)
- [ ] Sanity check single-task: CORD F1 ≥ 95, SROIE F1 ≥ 94 with Joint training

#### Day 4 (May 16) — Scenarios
- [ ] Implement `build_cil_cord(num_sessions=5)`:
  - 30 classes / 5 sessions = 6 classes per session
  - Document partitioning in `docs/cil_cord_split.md`
- [ ] Implement `build_dil()`:
  - Common 4-class schema: HEADER / KEY / VALUE / OTHER
  - Mapping in `docs/dil_schema_mapping.md`
- [ ] Implement `build_mixed()` (6 sessions interleaving)

#### Day 5 (May 17) — Pilot infrastructure
- [ ] Implement `linear_cka(X, Y)` in `doccl/eval/cka.py`
  - Formula: $\text{CKA}(X, Y) = \frac{||Y^T X||_F^2}{||X^T X||_F \cdot ||Y^T Y||_F}$
  - Unit test: CKA(X, X) = 1, CKA(X, random) ≈ 0
- [ ] Implement `fisher_information_per_group()` in `doccl/eval/fisher.py`
  - Empirical Fisher via squared gradient norm
  - Group-wise aggregation by parameter name patterns

#### Day 6 (May 18) — Modality ablation infrastructure
- [ ] Implement `forward_with_modality_mask()` in `LayoutLMv3Wrapper`:
  - `text=False`: zero out word embeddings
  - `image=False`: zero out patch embeddings
  - `layout=False`: zero out 2D position embeddings (rare, sanity only)
- [ ] Test that ablated forward pass produces sensible output (not NaN, not collapsed)

#### Day 7 (May 19) — First CL run + Gate 1
- [ ] Run Naive sequential on `cil_cord` and `dil` and `mixed`
- [ ] Run Joint on all 3 scenarios
- [ ] Compute Joint − Naive gap

**🚨 GATE 1 (May 19): Infrastructure Ready**

| Check | Pass criterion |
|---|---|
| 3 datasets load | FUNSD/CORD/SROIE return dict with input_ids, bbox, pixel_values, labels |
| 3 scenarios build | CIL/DIL/Mixed produce valid task sequences |
| CKA implementation | Unit tests pass |
| Fisher implementation | Unit tests pass |
| Modality ablation | Forward pass works for all combinations |
| Joint − Naive gap | ≥ 10 F1 on ≥ 2/3 scenarios |
| Reading complete | 3 papers read, verification questions answered |
| Candidates sketched | A, B, C have ≥ 1.5 pages each |

**If GATE 1 fails:**
- Gap < 10 on majority: scenarios too easy, redesign (more sessions, harder schema)
- CKA/Fisher buggy: debug carefully — these are pilot critical
- Modality ablation collapses: investigate which modality is dominant

---

## Phase 2 — Pilot Study (Weeks 3-4: May 20 - Jun 2)

**This is the diagnostic core of the paper. Method selection happens at end of W4.**

### Week 3: May 20-26 — Pilot Experiments

**Goal:** Run all 4 conditions × 3 seeds × 3 tasks = 36 runs. Collect CKA and Fisher data.

#### Day 1 (May 20) — Pilot setup
- [ ] Implement `pilot/conditions.py`:
  - C1: BERT-base wrapper (use `bert-base-uncased` + custom pooling for token classification)
  - C2: LayoutLMv3 with text masked (`forward_with_modality_mask(text=False)`)
  - C3: LayoutLMv3 with image masked (`forward_with_modality_mask(image=False)`)
  - C4: LayoutLMv3 full
- [ ] Implement `pilot/run_pilot.py`:
  - Train each condition on `[FUNSD, CORD, SROIE]` sequentially
  - At each task boundary, save checkpoint
  - Compute CKA between consecutive checkpoints per layer
  - Compute Fisher per parameter group at each task

#### Day 2-4 (May 21-23) — Run pilot grid
- [ ] Submit Vast.ai grid: 4 conditions × 3 seeds = 12 sequential runs
  - Each run: 3 tasks × ~30min = ~1.5h
  - Total: ~18 GPU-hours
- [ ] Monitor W&B; identify any condition that diverges (NaN, collapse)

#### Day 5-6 (May 24-25) — Data collection
- [ ] Aggregate CKA values: dimensions (condition, seed, task_boundary, layer_id, layer_group)
- [ ] Aggregate Fisher values: dimensions (condition, seed, task, parameter_group)
- [ ] Aggregate accuracy values: dimensions (condition, seed, task, evaluated_task)
- [ ] Save to `results/pilot/raw_data.parquet`

#### Day 7 (May 26) — First analysis
- [ ] Generate exploratory plots:
  - CKA heatmap per condition (rows = layers, cols = task boundaries)
  - Fisher bar chart per condition (groups: text-attn, vision-attn, fusion, layout-emb, classifier)
  - Forgetting matrix per condition
- [ ] Save plots to `results/pilot/exploratory/`

**Week 3 deliverable:** Raw pilot data + exploratory plots ready for analysis.

### Week 4: May 27 - Jun 2 — Pilot Analysis + Method Selection

**Goal:** Statistical analysis, identify pattern (or absence), select method candidate.

#### Day 1-2 (May 27-28) — Statistical analysis
- [ ] For each metric (CKA drop, Fisher drop, accuracy drop):
  - Compute mean ± std across 3 seeds
  - Compare C4 (full) vs. C1/C2/C3 baselines
  - Pairwise statistical tests (Mann-Whitney U, Bonferroni-corrected)
- [ ] Identify **patterns**:
  - Pattern P1: Per-component variance — which component drops most?
  - Pattern P2: Modality dependence — does C2/C3 forget less than C4?
  - Pattern P3: Scenario dependence — does pattern change across CIL/DIL/Mixed?

#### Day 3 (May 29) — Generate pilot figures (paper-ready)
- [ ] Figure 4.1: CKA-based forgetting heatmap (4 conditions side by side)
- [ ] Figure 4.2: Fisher information per parameter group (bar chart with error bars)
- [ ] Figure 4.3: Cross-condition comparison (highlights what's unique to C4)
- [ ] Save to `results/pilot/figures/`

#### Day 4 (May 30) — Write pilot findings memo
- [ ] Draft `docs/pilot_findings.md`:
  - Setup recap
  - 3-5 key findings with evidence
  - Implications for method design
- [ ] Send to advisor for review

#### Day 5 (May 31) — **Advisor meeting + Method selection**
- [ ] Present pilot findings to advisor
- [ ] Discuss method selection based on findings
- [ ] **DECISION:** select 1 of 4 paths
  - Pattern matches H_a (fusion focus) → **Candidate A (LAPP + H-LoRA)**
  - Pattern matches H_b (uniform + drift) → **Candidate B (Layout-Protected EWC)**
  - Pattern matches H_c (scenario-dep) → **Candidate C (Modality-Routed Prompts)**
  - No clear pattern → **Pivot to characterization-only paper**

#### Day 6-7 (Jun 1-2) — Document decision + plan adjustments
- [ ] Update CLAUDE.md decision log with method selection
- [ ] Update PLAN.md Phase 5 with selected method specifics
- [ ] Refine selected candidate's implementation outline (1 day buffer)
- [ ] Prepare W5-W6 baseline implementation plan

**🚨 GATE 2 (Jun 2): Method Selection — CRITICAL DECISION**

This decision **locks the method until end of W11**. Pass criteria:

- Pilot data collected for all 4 conditions × 3 seeds = 12 successful runs
- Statistical analysis complete with significance tests
- Pattern identified (or absence justified) with advisor concurrence
- Selected method (or fallback) documented in `docs/method_decision.md`

**If pilot data insufficient:**
- Re-run failed seeds (1-2 days max)
- If still incomplete → defer decision to W5 (compresses Phase 3 by 1 week)

**If no clear pattern (fallback to characterization-only):**
- Continue with full baseline grid (Phases 3-4) for benchmark contribution
- W9-11 dedicated to expanded pilot analysis instead of method
- Paper structure shifts: Section 4 becomes 3 pages, Section 5 (method) removed

---

## Phase 3 — Core Baselines (Weeks 5-6: Jun 3-16)

**Goal:** 6 baselines (Naive, Joint, EWC, LwF, ER, DER++) running on all 3 scenarios.

### Week 5: Jun 3-9 — Regularization methods

- [ ] Implement `EWC` in `doccl/methods/ewc.py`:
  - Compute Fisher matrix in `after_task()` using ~200 samples
  - Quadratic penalty: $\frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$
  - Sweep: $\lambda \in \{100, 1000, 10000\}$
- [ ] Implement `LwF` in `doccl/methods/lwf.py`:
  - Cache teacher in `after_task()`
  - KD loss: $\alpha \cdot \text{KL}(\sigma(z_s/T) || \sigma(z_t/T))$
  - Sweep: $\alpha \in \{0.5, 1.0, 2.0\}$, T=2
- [ ] Tests in `tests/test_methods.py`
- [ ] Smoke runs on CIL-CORD with seed=42

**Deliverable:** EWC > Naive and LwF > Naive on CIL-CORD demonstrably.

### Week 6: Jun 10-16 — Replay methods

- [ ] Implement `ER` in `doccl/methods/er.py`:
  - Reservoir sampling buffer
  - Buffer stores `(input_ids, bbox, pixel_values, labels)`
  - Alternate batches from current task + buffer
  - Sweep: $|M| \in \{200, 500\}$
- [ ] Implement `DER++` in `doccl/methods/der.py`:
  - Buffer also stores logits
  - Loss = $\text{CE}(\text{current}) + \alpha \cdot \text{MSE}(\text{logits}_{\text{cached}}) + \beta \cdot \text{CE}(\text{replay})$
  - Sweep: $\alpha, \beta \in \{0.5, 1.0\}$
- [ ] Run grid: 6 methods × 3 scenarios × 3 seeds = 54 runs
  - Estimated: 54 × 1.0h = ~54 GPU-hours
- [ ] Generate Table 1 v1 (baselines only)
- [ ] Generate Figure 5.1: forgetting curves

**🚨 GATE 3 (Jun 16): Baseline Validation**

| Check | Pass criterion |
|---|---|
| 6 methods × 3 scenarios complete | 54 runs without errors |
| Best baseline (likely DER++) | ≥ Naive + 5 F1 on ≥ 2 scenarios |
| Std across 3 seeds | ≤ 2 F1 |
| Statistical test | Best vs Naive p < 0.05 (paired t-test, Bonferroni) |

**If fail:** debug against author code (Mammoth repo for DER++, original EWC code).

---

## Phase 4 — Advanced Baselines (Weeks 7-8: Jun 17-30)

**Goal:** 4 advanced baselines (L2P, DualPrompt, CODA-P, O-LoRA) — paper's prompt-CL and LoRA-CL representatives.

### Week 7: Jun 17-23 — Prompt foundation: L2P

- [ ] Resolve `_forward_with_prompts` in `LayoutLMv3Wrapper`:
  - Get text embeddings via `embeddings.word_embeddings(input_ids)`
  - Prepend prompts; adjust attention_mask, position_ids, bbox
  - Test forward pass produces shape `(B, L+P, num_labels)`
- [ ] Implement `L2P`:
  - Frozen LayoutLMv3 backbone
  - Prompt pool: N=10, length L_p=5
  - Query: CLS embedding
  - Top-K=5 selection
  - Loss: CE + λ·key_pull
- [ ] Run L2P on CIL-CORD first to verify

### Week 8: Jun 24-30 — DualPrompt + CODA-Prompt + O-LoRA

- [ ] Implement `DualPrompt`:
  - G-Prompt (layers [0,1]) + E-Prompt (layers [2,3,4] via prefix-tuning)
- [ ] Implement `CODA-Prompt`:
  - Decomposed: $P = \sum_i \alpha_i \cdot P_i$ with attention weights
  - Orthogonality reg on prompts AND keys AND attention matrices
- [ ] Implement `O-LoRA`:
  - LoRA on Q/K/V of all 12 attention layers, rank 8
  - Per-task: orthogonality $A_t^T A_{<t} = 0$
  - Inference: sum all task LoRAs (merging)
- [ ] Run grid: 4 methods × 3 scenarios × 3 seeds = 36 runs
  - Estimated: 36 × 1.2h = ~43 GPU-hours
- [ ] Update Table 1 with all 10 baselines

**🚨 GATE 4 (Jun 30): Advanced Baseline Health**

| Check | Pass criterion |
|---|---|
| 4 advanced × 3 scenarios complete | 36 runs without errors |
| ≥ 1 prompt method | ER-level performance |
| O-LoRA | ER-level performance |
| Total grid | 10 methods × 3 scenarios × 3 seeds = 90 runs done |

**If fail:** prompt below ER suggests bug in `_forward_with_prompts`. Reference [JH-LEE-KR/dualprompt-pytorch](https://github.com/JH-LEE-KR/dualprompt-pytorch).

---

## Phase 5 — Selected Method + Writing (Weeks 9-14: Jul 1 - Aug 11)

**Goal:** Implement selected candidate (chosen Week 4), evaluate, write paper, submit.

### Week 9: Jul 1-7 — Selected method implementation (Part 1)

The exact tasks depend on which candidate was selected at Week 4. General template:

- [ ] Refine selected candidate's design from W1-2 sketch
- [ ] Implement core component(s) (most novel piece first)
- [ ] Unit tests for new components
- [ ] Smoke run on CIL-CORD seed=42

**If Candidate A (LAPP + H-LoRA):**
- Day 1-3: Implement LAPP (extends L2P with layout signature)
- Day 4-7: Implement H-LoRA (3 banks with per-bank orthogonality)

**If Candidate B (Layout-Protected EWC):**
- Day 1-2: Identify layout parameters (2D position embeddings module names)
- Day 3-5: Implement selective EWC with per-group importance weights
- Day 6-7: Optional layout distillation

**If Candidate C (Modality-Routed Prompts):**
- Day 1-3: Implement 3 prompt sub-pools (text/visual/layout)
- Day 4-5: Implement router network
- Day 6-7: Joint training procedure

### Week 10: Jul 8-14 — Selected method implementation (Part 2) + Ablations

- [ ] Combine components if multi-part method
- [ ] Run main results: selected × 3 scenarios × 3 seeds = 9 runs
- [ ] Component ablations (each piece alone)
- [ ] Hyperparameter ablations (rank, prompt size, lambda values)
- [ ] Total ablations: ~30-40 runs

### Week 11: Jul 15-21 — Final ablations + GATE 5

- [ ] Complete all ablations
- [ ] Final results table
- [ ] Verify statistical significance vs. best baseline

**🚨 GATE 5 (Jul 21): Method Performance Decision**

| Branch | Criterion | Paper framing |
|---|---|---|
| A (Win) | Selected ≥ best baseline + 2 F1 on ≥ 2 scenarios | "Method paper, selected method advances SOTA on FCS-CL" |
| B (Tie) | Selected within 2 F1 of best baseline | "Method paper, selected method matches SOTA with [param efficiency / privacy / speed advantage]" |
| C (Loss) | Selected underperforms by > 2 F1 | "**Pivot to characterization-only paper.** Pilot study (Section 4) is headline contribution. FCS-CL benchmark (Section 6) secondary. Method section shrunk or removed." |

All three branches → publishable at AAAI. Communicate decision to advisor by Jul 22 morning.

### Week 12: Jul 22-28 — Draft v1

- Day 1: Lock narrative based on GATE 5 decision
- Day 2-3: Method section (or expand pilot if Branch C)
- Day 4-5: Experiments section setup, Tables/Figures
- Day 6-7: Related work + Problem formulation

**Parallel: Generalization study starts on Vast.ai overnight runs**
- LiLT + BROS × 6 selected methods × 3 scenarios × 3 seeds = 108 runs
- Estimated: ~130 GPU-hours over 2 weeks

### Week 13: Jul 29 - Aug 4 — Draft v2 + advisor review

- Day 1-2: Final tables/figures (including generalization results)
- Day 3 (Jul 31): **Send to advisor**
- Day 4-5: Polish Intro + Abstract
- Day 6-7: Address advisor feedback

### Week 14: Aug 5-11 — Final polish + submit

- Day 1-2: Last advisor-requested ablations
- Day 3: Reproducibility checklist
- Day 4: Final read-through (printed)
- Day 5 (Aug 9): **SUBMIT** (1 day buffer)
- Day 6-7: Reserved for emergencies

---

## Paper Structure (Diagnostic + Remedy)

```
1. Introduction (1 page)
   - Document AI in production needs CL
   - Existing CL methods designed for unimodal models
   - What's unique about multimodal document encoders?
   - Contributions: characterization + method emergent + benchmark

2. Related Work (1 page)
   - Document AI (LayoutLM family)
   - CL methods (replay, prompt, LoRA)
   - Mechanistic CF analysis (Ramasesh, Zhai, Kornblith)
   - Gap: no characterization for document encoders

3. Problem Formulation (0.5 page)
   - Formal CL setup
   - Sequence labeling task definition
   - Three scenarios (CIL, DIL, Mixed)

4. Per-Component Forgetting Analysis (1.5 pages) ← CORE CONTRIBUTION 1
   4.1 Experimental setup (4 conditions: BERT, LayoutLM-no-text, LayoutLM-no-image, LayoutLM-full)
   4.2 Metrics: CKA, Fisher, accuracy
   4.3 Findings (emergent from data)
   4.4 Implications for CL method design

5. DocCL Method (1.5 pages) ← CORE CONTRIBUTION 2 (or shrunk if Branch C)
   - Selected candidate, justified by Section 4 findings
   - Architecture + algorithm

6. Experiments (1.5 pages) ← CORE CONTRIBUTION 3
   - FCS-CL benchmark
   - Main results vs 10 baselines
   - Ablations linking back to pilot findings
   - Generalization to LiLT + BROS

7. Conclusion (0.25 page)
   + Future work: NeurIPS extension (DSR, multilingual, theory)

References (2 pages)
Supplementary (unlimited)
```

---

## Risk Management

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Pilot reveals no clear pattern | Med | High | Fallback: characterization-only paper |
| Selected method fails Week 11 | Med | Med | GATE 5 Branch C → characterization-only |
| AAAI deadline earlier than expected | Med | High | Verify CFP daily; 2-day buffer in W14 |
| Compute insufficient | Low | High | Reduce seeds 3→2 for non-essential ablations |
| Advisor unavailable W4 (method decision) | Med | Critical | Schedule NOW; backup = senior PhD |
| Advisor unavailable W13 (paper review) | Med | High | Schedule NOW; backup reviewer |
| CKA/Fisher implementation buggy | Med | High | Unit tests Week 2; test against reference |
| Modality ablation collapses | Low | Med | Test in W2; if fails, drop C2 (ViT path) |
| Reading list insufficient | Low | Med | Add Pope 2023 if needed |
| Generalization runs preempted | High | Low | Use on-demand for finals; checkpoint epochs |
| AAAI rebuttal during defense prep | Med | Med | Rebuttal Oct, Defense Dec — 2 month buffer |

### Decision Trees

**If Week 4 pilot reveals no clear pattern:**
1. Verify with advisor
2. Consider extending pilot (1 week extra) with 2 more conditions
3. If still no pattern → commit to characterization-only paper, restructure plan

**If Week 11 method underperforms:**
1. Consider hybrid (selected method + best baseline ensemble)
2. If still underperforms → Branch C (characterization-only)
3. Restructure paper Section 4-5 (expand 4, shrink/remove 5)

**If pre-locked candidate doesn't fit pilot finding:**
- Allow Week 9-10 buffer to refine OR switch to second-best candidate
- Document deviation in `docs/method_decision.md`

---

## Weekly Check-in Routine

Every Friday (4 PM Hanoi time):

1. **Self-review (30 min):**
   - Tasks done this week vs. PLAN.md
   - Tasks slipped → why?
   - Update PLAN.md status checkboxes

2. **Advisor sync (15-30 min, if available):**
   - Show W&B dashboard
   - Discuss any blocker
   - Get green light for next week

3. **Update tracking:**
   - GPU-hours used vs. budget
   - Number of runs completed
   - Updated risk matrix

4. **Commit `git push`:**
   - Push code progress to `develop`
   - Push PLAN.md updates to `main`

---

## Communication Templates

### To advisor (weekly):
```
Subject: [DocCL W{N}] Progress + question on {topic}

Hi Prof,

Week {N} status:
- Done: {bullet list}
- In progress: {bullet list}
- Blocked: {if any}

Quick question: {1-2 questions}

W&B dashboard: {link}

Best,
Thanh
```

### To advisor (Week 4 method-decision meeting request):
```
Subject: [DocCL] Pilot findings ready — method selection meeting

Hi Prof,

Pilot study complete. Findings memo attached.

Key result: {1 sentence summary of pattern (or absence)}

Three candidate methods sketched in Week 1. Based on findings,
I lean toward Candidate {A/B/C/fallback} because {1 sentence}.

Need 30-min meeting this week to decide.

Available: {3 time slots}

Pilot findings memo: docs/pilot_findings.md
Pilot figures: results/pilot/figures/

Best,
Thanh
```

### To advisor (Week 13 paper review request):
```
Subject: [DocCL] Draft v2 ready for review — AAAI 2027 submission

Hi Prof,

Draft v2 attached for your review. Key changes from v1:
- {3-5 bullet points}

Submission deadline: August 9, 2026
Suggested review meeting: {date/time options}

Repo: {link}
Paper PDF: attached

Best,
Thanh
```

---

## Appendix A — File Conventions

### Code style
- Black formatter (line length 100)
- Ruff linter
- Type hints on all public functions
- Docstrings: Google style
- All comments and docstrings in English

### Commit conventions
- Format: `[scope] short description`
- Scopes: `infra`, `data`, `methods`, `models`, `eval`, `pilot`, `docs`, `tests`
- Examples:
  - `[methods] add EWC with Fisher matrix`
  - `[pilot] add CKA metric implementation`
  - `[infra] migrate to Hydra 1.3`
  - `[docs] update PLAN.md week 5 progress`

### Branch naming
- Feature: `feat/<scope>` (e.g., `feat/pilot-study`, `feat/cka-metric`)
- Fixes: `fix/<short-description>`
- Documentation: `docs/<topic>`

---

## Appendix B — Key References Quick Lookup

**Backbone:**
- Huang et al., LayoutLMv3, ACM MM 2022, [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)

**Pilot study foundations (REQUIRED reading before W2):**
- Ramasesh et al., "Anatomy of CF," ICLR 2021, [arXiv:2007.07400](https://arxiv.org/abs/2007.07400)
- Zhai et al., "Investigating CF in MLLMs," 2023, [arXiv:2309.10313](https://arxiv.org/abs/2309.10313)
- Kornblith et al., "Similarity of NN Reps," ICML 2019, [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

**CL methods:**
- Kirkpatrick et al., EWC, PNAS 2017
- Li & Hoiem, LwF, ECCV 2016
- Rolnick et al., ER, NeurIPS 2019
- Buzzega et al., DER++, NeurIPS 2020, [arXiv:2004.07211](https://arxiv.org/abs/2004.07211)
- Wang et al., L2P, CVPR 2022
- Wang et al., DualPrompt, ECCV 2022
- Smith et al., CODA-Prompt, CVPR 2023
- Wang et al., O-LoRA, EMNLP-F 2023, [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)

**Metrics:**
- Lopez-Paz & Ranzato, GEM (defines AA, BWT, FWT), NeurIPS 2017

**Document AI CL precedents (sparse):**
- Minouei et al., "CL for Table Detection," Applied Sciences 2022
- Wójcik et al., "Domain-Agnostic NA," arXiv:2307.05399, 2023
- Kumar et al., ProtoNER, DAS 2024

---

## Appendix C — Hyperparameter Defaults

```yaml
# Default training
optimizer: AdamW
lr: 5e-5
weight_decay: 0.01
warmup_ratio: 0.1
batch_size: 8
gradient_accumulation_steps: 1
max_grad_norm: 1.0
fp16: true
epochs: 10  # per task

# CL-specific
seeds: [42, 123, 7]  # 3 seeds for main results
buffer_size: 200  # for replay
ewc_lambda: 1000  # default; sweep {100, 1000, 10000}
lwf_alpha: 1.0  # default; sweep {0.5, 1.0, 2.0}
lwf_temperature: 2.0
prompt_pool_size: 10
prompt_length: 5
top_k: 5
lora_rank: 8  # default; sweep {4, 8, 16}
lora_alpha: 16
lora_dropout: 0.1

# Pilot-specific
cka_n_samples: 500  # samples used to compute CKA
fisher_n_samples: 200  # samples used to compute Fisher
```

---

## Appendix D — Submission Checklist (Aug 9, 2026)

- [ ] Paper PDF (7 pages content + ≤2 pages references)
- [ ] Anonymized PDF (no author names, no identifying acks)
- [ ] AAAI 2027 LaTeX template used correctly
- [ ] All figures readable in B&W print
- [ ] All tables fit on page (no overflow)
- [ ] Abstract ≤ 250 words
- [ ] Code release link (anonymized GitHub or zip)
- [ ] Reproducibility Checklist completed
- [ ] Supplementary < page limit
- [ ] All references have full bibliographic info
- [ ] No copyrighted text quoted
- [ ] Statistical tests reported with p-values
- [ ] All claims grounded in results section
- [ ] Future work mentions NeurIPS extension (DSR + multilingual + theory)
- [ ] Pilot study figures clear and self-contained
- [ ] Method (or characterization) story aligned with pilot findings

---

**END OF PLAN.md**

This document is updated weekly. Last review: May 6, 2026.
