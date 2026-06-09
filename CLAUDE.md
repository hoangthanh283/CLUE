# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DocCL** is a Master's thesis project on **Continual Learning for Information Extraction and Document Understanding**. The project produces:

1. **Master's thesis** (defense December 2026, Hanoi University of Science and Technology)
2. **AAAI 2027 publication** (deadline ~August 2026) — diagnostic + remedy paper
3. **NeurIPS 2027 publication** (deadline ~May 2027, post-thesis) — full extension

**Crucial distinction from typical CL papers:** This is an **insight-driven paper**, not a pure method paper. Structure:

```
characterize → hypothesize → exploit → validate
```

Specifically, the paper:
- **Characterizes** how catastrophic forgetting manifests in multimodal document encoders (LayoutLMv3) compared to unimodal baselines (BERT, modality-ablated variants)
- **Identifies** which architectural components forget most (fusion vs. text vs. visual streams)
- **Proposes** a method whose design is **justified by characterization findings**, not predetermined
- **Validates** on FCS-CL benchmark (FUNSD/CORD/SROIE)

This framing places the paper in the **diagnostic + remedy** family (e.g., Ramasesh ICLR 2021, Zhai 2023). Method emerges from data, not from author preference.

## Author Context

- **Researcher:** Thanh Hoang
- **Institution:** Hanoi University of Science and Technology (HUST)
- **Compute:** 1× RTX 4090 (Vast.ai cloud) + 1× RTX 2060 (offline, debug only)
- **Compute budget:** ~$250 for AAAI phase (~340 GPU-hours expected including secondary backbones)
- **Logging:** W&B account active (project: `doccl-aaai2027`)
- **Advisor:** Available for review (critical at Week 4 method-decision point and Week 13 paper review)

## Strategic Framing

### Two-Tier Conference Strategy

| Aspect | AAAI 2027 (Lite) | NeurIPS 2027 (Full) |
|---|---|---|
| Pilot study | Per-component forgetting on 4 conditions | + theoretical analysis, more conditions |
| Method | 1 method (selected from 3 candidates post-pilot) | Refined + 2 additional components |
| Scenarios | CIL, DIL, Mixed (3) | + TIL, Class+Domain, Cross-lingual (5) |
| Baselines | 10 methods | 15+ methods |
| Backbones | LayoutLMv3-base + LiLT + BROS | + LayoutLMv3-large, LayoutXLM, Donut |
| Datasets | FUNSD, CORD, SROIE | + DocVQA, RVL-CDIP, XFUND |

When writing AAAI paper, **explicitly mention extensions as future work** to legitimize the NeurIPS extension.

### Scope Discipline

The thesis-time scope is intentionally **cut**: drop DSR (Document-aware Structural Replay) and TIL scenario from AAAI paper. These are reserved for the NeurIPS extension. **Do not let scope creep back in.**

If asked to implement DSR or TIL during AAAI phase: politely push back — these are post-thesis NeurIPS extensions. The infrastructure should *support* them (extension points exist) but implementations stay deferred.

## Problem Formulation (Locked)

### Primary Task: Sequence Labeling for Key Information Extraction

```
Input:  Document D = (W, B, I)
        W = (w_1, ..., w_N)   sequence of tokens (post-OCR + WordPiece)
        B = (b_1, ..., b_N)   sequence of bounding boxes, b_i ∈ [0, 1000]^4
        I ∈ R^(3 × 224 × 224) page image

Output: Y = (y_1, ..., y_N)   sequence of BIO labels
        y_i ∈ L_dataset       dataset-specific BIO tag set

Loss:   Token-level cross-entropy with -100 for ignored positions

Metric: Entity-level F1 (seqeval, BIO-aware span matching)
```

**Output spaces per dataset:**
- FUNSD: 7 BIO tags (3 entity types + O)
- CORD: 61 BIO tags (30 fine-grained classes + O) — fine-grained, not super-classes
- SROIE: 9 BIO tags (4 fields + O)

**Why this framing (defensible to reviewers):**
1. Aligns with dominant VRDU literature (LayoutLM family, LiLT, BROS)
2. Unified output format → CL methods apply straightforwardly
3. Compatible with all 3 chosen encoder-only backbones
4. Semantic shift of "O" tag is central to CIL

### Three CL Scenarios

- **CIL-CORD:** 30 fine-grained classes split into 5 sessions × 6 classes
- **DIL:** FUNSD → SROIE → CORD-superclass with unified 4-class schema (HEADER/KEY/VALUE/OTHER)
- **Mixed:** 6 sessions interleaving class-IL and domain shifts

**Explicitly NOT in AAAI scope:** TIL, entity linking, JSON generation, document classification.

## Pilot Study Design (CORE OF THE PAPER)

The **diagnostic core**. Conducted Week 3-4. Findings drive method selection in Week 4.

**Setup:** Train sequentially on FUNSD → CORD → SROIE (3 tasks, naive sequential, no CL method) for each condition. Measure forgetting at task boundaries.

**Four conditions (apples-to-apples):**

| Condition | Model | Input streams | Purpose |
|---|---|---|---|
| C1 | BERT-base | Text only | External text-only baseline |
| C2 | LayoutLMv3-no-text | Image + layout (text masked) | Isolate visual+layout-only forgetting |
| C3 | LayoutLMv3-no-image | Text + layout (image masked) | Isolate text+layout-only forgetting |
| C4 | LayoutLMv3-full | Text + image + layout | Main subject of analysis |

C2 and C3 are **modality-ablated variants** of LayoutLMv3 (same architecture, modalities zeroed at input).

**Metrics:**
- **Per-layer CKA** (Centered Kernel Alignment) between checkpoints (Kornblith 2019)
- **Per-layer-group Fisher Information** drop
- **Final accuracy** on each seen task (standard forgetting metric)

**Hypothesis testing:**
- **H0 (null):** Forgetting uniform across components
- **H1 (alt):** Forgetting concentrates in specific components (pattern emerges from data)

**Decision rule (Week 4):** If H0 cannot be rejected → pivot to fallback (characterization-only paper).

## Three Candidate Methods (Sketched in Week 1-2)

Three methods are pre-sketched on paper. **One is selected at end of Week 4 based on pilot findings.** This is the safety net for the "method emergent" approach.

### Candidate A: LAPP + H-LoRA
**Triggered if pilot reveals:** Fusion layers forget most + layout-sensitivity matters

```
Layout-Aware Prompt Pool (LAPP):
  q(x) = CLS_emb + W·φ(boxes)
  φ(boxes) = 4×4 grid histogram of box centers (16-dim)
  W = trainable Linear(16, 768)

Hierarchical LoRA (H-LoRA):
  Bank 1 (Text): Q/K/V of text-attention layers
  Bank 2 (Visual): Patch projection + visual self-attn
  Bank 3 (Fusion): Cross-modal fusion projection ← key bank
  Per-bank orthogonality constraint (relaxed vs. global O-LoRA)
```

### Candidate B: Layout-Protected Regularization
**Triggered if pilot reveals:** Uniform forgetting BUT 2D position embeddings drift fastest

```
Selective EWC variant:
  L_total = L_CE + λ_high · F_layout · (θ_layout - θ*_layout)²
                 + λ_low · F_other · (θ_other - θ*_other)²
  
  where F_layout = Fisher for 2D position embeddings (high importance weight)
  
Optional: layout embedding distillation from previous task model
```

### Candidate C: Modality-Routed Prompts
**Triggered if pilot reveals:** Per-modality forgetting patterns scenario-dependent

```
Three prompt sub-pools per modality:
  P_text, P_visual, P_layout

Router selects which sub-pool to activate based on input features:
  r(x) = softmax(MLP([CLS_emb, layout_sig, visual_feature]))

Active sub-pool prepended to corresponding stream.
```

### Selection Criteria (used at end of Week 4)

| Pilot Finding | Pick |
|---|---|
| Fusion forgetting dominant + layout matters | Candidate A |
| Uniform forgetting + position embeddings drift | Candidate B |
| Scenario-dependent patterns | Candidate C |
| **No clear pattern (uniform across all metrics)** | **Pivot to characterization-only paper** |

## Backbones (final list)

**Primary:** LayoutLMv3-base (133M params, 12 layers, hidden=768)
- `microsoft/layoutlmv3-base` from HuggingFace
- Run **full grid** (10 baselines + selected method × 3 scenarios × 3 seeds)

**Secondary (generalization study):** LiLT-base + BROS-base
- `nielsr/lilt-xlm-roberta-base` and `naver-clova-ix/bros-base-uncased`
- Run **subset** (Naive + Joint + best-replay + best-prompt + best-LoRA + DocCL = 6 methods)
- Conducted **parallel with paper writing in W12-13** (overlap compute with writing time)

## Methods (final list for AAAI)

**Lower/upper bounds:**
- `Naive` — sequential FT (lower bound)
- `Joint` — multi-task on all data (oracle upper bound)

**Regularization (Week 5):**
- `EWC` — Kirkpatrick et al., PNAS 2017
- `LwF` — Li & Hoiem, ECCV 2016

**Replay (Week 6):**
- `ER` — Rolnick et al., NeurIPS 2019
- `DER++` — Buzzega et al., NeurIPS 2020

**Prompt-based (Weeks 7-8):**
- `L2P` — Wang et al., CVPR 2022
- `DualPrompt` — Wang et al., ECCV 2022
- `CODA-Prompt` — Smith et al., CVPR 2023

**LoRA-based (Week 8):**
- `O-LoRA` — Wang et al., EMNLP-F 2023

**Proposed (Week 10-11):** Selected post-pilot from {Candidate A, B, C}.

**Total:** 10 baselines + 1 selected method = 11 methods on primary backbone.

## Reading List (Pre-Pilot, REQUIRED before Week 2)

Required reading to design pilot study correctly. Total time: ~6 hours.

1. **Ramasesh, Dyer, Raghu (ICLR 2021)** — *"Anatomy of Catastrophic Forgetting"*
   - arXiv: 2007.07400
   - Read: Abstract + Section 3 (Methodology) + Section 4 (Findings)
   - Skip: Long appendices

2. **Zhai et al. (2023)** — *"Investigating CF in Multimodal LLMs"*
   - arXiv: 2309.10313
   - Read: Abstract + Section 3 (EMT Framework) + Section 4 (Findings)
   - Skip: Specific MLLM model details

3. **Kornblith et al. (ICML 2019)** — *"Similarity of NN Representations Revisited"*
   - arXiv: 1905.00414
   - Read: Abstract + Section 2 (CKA definition) + Section 3 (properties)
   - Skip: Long experimental sections

**Verification questions** (answer before starting pilot):
- What does CKA measure? How is it different from cosine similarity between weights?
- How does Ramasesh measure forgetting per-layer?
- What metrics does Zhai use for MLLMs? Adaptable to LayoutLMv3?

## Repository Structure

```
doccl/
├── data/                    # Dataset loaders + scenario builders
│   ├── datasets.py          # FUNSDDataset, CORDDataset, SROIEDataset
│   └── scenarios.py         # build_cil_cord, build_dil, build_mixed + registry
├── methods/                 # CL method implementations
│   ├── base.py              # ContinualMethod ABC, TaskInfo, TaskState
│   ├── naive.py             # NaiveFineTune, JointMultiTask
│   ├── ewc.py               # (Week 5) EWC
│   ├── lwf.py               # (Week 5) LwF
│   ├── er.py                # (Week 6) ER
│   ├── der.py               # (Week 6) DER++
│   ├── l2p.py               # (Week 7) L2P
│   ├── dualprompt.py        # (Week 8) DualPrompt
│   ├── coda_prompt.py       # (Week 8) CODA-Prompt
│   ├── o_lora.py            # (Week 8) Orthogonal LoRA
│   └── doccl.py             # (Weeks 10-11) Selected candidate
├── models/
│   ├── layoutlm_wrapper.py  # LayoutLMv3 wrapper with extension points
│   ├── lilt_wrapper.py      # (Week 12) LiLT for generalization
│   ├── bros_wrapper.py      # (Week 12) BROS for generalization
│   └── modality_ablation.py # (Week 3) Pilot conditions C2, C3
├── eval/
│   ├── metrics.py           # AA, BWT, FWT, AF + token F1 (seqeval)
│   ├── cka.py               # (Week 2) CKA implementation
│   └── fisher.py            # (Week 2) Fisher info per-layer-group
├── pilot/                   # Pilot study scripts
│   ├── conditions.py        # C1-C4 setup
│   ├── analyze.py           # Forgetting characterization
│   └── visualize.py         # Generate Figure 4.X for paper
└── utils/

configs/
├── default.yaml
├── scenario/
├── method/
├── model/                   # layoutlmv3_base, lilt_base, bros_base
├── pilot/                   # pilot_c1, pilot_c2, pilot_c3, pilot_c4
└── training/

scripts/
├── train.py                 # Main entry (Hydra)
├── run_pilot.sh             # Pilot study grid (W3-4)
├── run_grid.sh              # Main baseline grid
├── run_generalization.sh    # Secondary backbones (W12-13)
└── analyze_results.py
```

## Critical Code Conventions

### `ContinualMethod` Interface

```python
def before_task(task, train_loader)  # extend classifier head, allocate LoRA bank
def train_task(task, train_loader)   # main training loop (REQUIRED)
def after_task(task, train_loader)   # update buffer, compute Fisher, freeze weights
def evaluate(task, eval_loaders)     # eval on all seen tasks (REQUIRED)
```

### LayoutLMv3 Extension Points

- `expand_classifier(new_labels)` — for class-incremental learning
- `get_layout_signature(boxes, grid_size=4)` — layout descriptor
- `_injected_prompts: dict[int, Tensor]` — for L2P/LAPP
- `_lora_banks: dict[str, Module]` — for O-LoRA / H-LoRA
- **NEW:** `forward_with_modality_mask(text=True, image=True, layout=True)` — for pilot conditions

### CKA and Fisher (NEW for pilot study)

```python
# In doccl/eval/cka.py
def linear_cka(X: Tensor, Y: Tensor) -> float:
    """Linear Centered Kernel Alignment between two activation matrices.
    X, Y of shape (N_samples, D_features). Returns scalar in [0, 1].
    """

# In doccl/eval/fisher.py
def fisher_information_per_group(
    model, dataloader, param_groups: dict[str, list[Parameter]]
) -> dict[str, float]:
    """Empirical Fisher per parameter group."""
```

### Configuration

Use **Hydra** with composable configs:
```bash
python scripts/train.py method=ewc scenario=cil_cord seed=42 method.lambda_=1000
```

### Logging

W&B project: `doccl-aaai2027`. Tag schema: `{phase}/{scenario}/{method}/seed{seed}`.

Pilot phase: `pilot/{condition}/seed{seed}` (e.g., `pilot/c4_full/seed42`).

### Reproducibility

3 seeds for main results: `[42, 123, 7]`. Set `torch.backends.cudnn.deterministic = True`.

## Development Workflow

```bash
# Setup
conda create -n doccl python=3.10 -y && conda activate doccl
pip install -r requirements.txt && pip install -e .

# Smoke tests
pytest tests/test_smoke.py -v -m "not slow"

# Sanity-check
python scripts/train.py method=naive scenario=single_funsd seed=42

# Pilot study (Week 3-4)
bash scripts/run_pilot.sh

# Main CL run
python scripts/train.py method=naive scenario=cil_funsd seed=42

# Grid run (after Week 6)
bash scripts/run_grid.sh

# Generalization study (Week 12-13)
bash scripts/run_generalization.sh

# Aggregate
python scripts/analyze_results.py --project doccl-aaai2027 --output results/table1.tex
```

### Branch Strategy

- `main` — published states only
- `develop` — integration
- `feat/<scope>` — feature branches (`feat/pilot-study`, `feat/cka-metric`)

## Communication Style with Claude Code

1. **Stay in scope.** Defer DSR, TIL, multilingual to NeurIPS phase.
2. **Always read SKILL.md when present** before creating typed files.
3. **Reproduce before extending.** Make ER work before DER++.
4. **No shortcuts on metrics.** AA/BWT/FWT per Lopez-Paz & Ranzato 2017. CKA per Kornblith 2019.
5. **Statistical rigor.** 3 seeds + paired t-test or Wilcoxon for main results.
6. **Pilot findings drive method design.** Do NOT pre-decide method beyond 3 candidates.
7. **Commit format:** `[scope] short description` (e.g., `[pilot] add CKA metric`)
8. **Vietnamese OK in commits**, but code/docstrings/configs/paper are English.

## What Claude Code Should NEVER Do

1. Reproduce copyrighted text verbatim — paraphrase always.
2. Modify the `ContinualMethod` interface without explicit discussion.
3. Change the OCR pipeline mid-experiment.
4. Run paper-result experiments locally on RTX 2060.
5. Push to `main` directly.
6. Reduce seed count below 3 for main results.
7. **Lock the proposed method before pilot completes (Week 4).**
8. **Collapse the pilot study to save time.** It's the foundation contribution.

## Quick Reference: Expected Performance

Single-task ceilings (Joint, LayoutLMv3-base):
- FUNSD: F1 ≈ 90-92
- CORD: F1 ≈ 96-97
- SROIE: F1 ≈ 95-96

CL gap (Joint - Naive) target: **≥ 10 F1** on ≥ 2/3 scenarios.
Selected method target: **≥ best non-self baseline + 2 F1** on ≥ 2 scenarios.
Pilot study target: **clear pattern emerges** distinguishing C4 from C1/C2/C3.

## Fallback Strategy

If Week 11 reveals selected method does not outperform best baseline → pivot to **characterization-only**:
- Pilot study (Section 4) becomes headline contribution
- FCS-CL benchmark (Section 6) becomes secondary
- Method section shrunk or removed
- Comparable to Zhai 2023 in spirit
- Still publishable at AAAI as benchmark + analysis paper

This fallback exists because Decision B (no pre-locked method) accepts higher implementation risk. Pilot study's value is independent of method success.

## Decision Log (Locked May 6, 2026)

| Decision | Choice | Rationale |
|---|---|---|
| Paper framing | Insight-driven (Phrasing 3) | Scientific value + method |
| Evidence level | Level 2 + 3 (lit + pilot) | Sweet spot for AAAI |
| Hypothesis | Open, emerge from data | Honest research |
| Memory constraint | Soft preference | Don't exclude replay arbitrarily |
| Backbones | LayoutLMv3 + LiLT + BROS | 1 primary + 2 generalization |
| Generalization timing | Parallel W12-13 | Overlap compute with writing |
| Pilot conditions | BERT + 3 LayoutLM variants | Apples-to-apples |
| Candidate methods | 3 sketched W1-2 | Safety net for W4 decision |
| Pre-reading | 3 papers before W2 | ~6h for pilot quality |
| Fallback strategy | Characterization-only paper | If method fails W11 |
| CORD label space | 30 fine-grained (61 BIO) | Match SOTA |
| TIL scenario | Deferred to NeurIPS | AAAI scope cut |
| Entity Linking | Future work mention | AAAI scope cut |

## Reference Material

**Backbone:** Huang et al., LayoutLMv3, ACM MM 2022, arXiv:2204.08387

**Pilot study foundations (REQUIRED reading):**
- Ramasesh et al., "Anatomy of CF," ICLR 2021, arXiv:2007.07400
- Zhai et al., "Investigating CF in MLLMs," 2023, arXiv:2309.10313
- Kornblith et al., "NN Representations," ICML 2019, arXiv:1905.00414

**CL methods:**
- Kirkpatrick et al., EWC, PNAS 2017
- Li & Hoiem, LwF, ECCV 2016
- Rolnick et al., ER, NeurIPS 2019
- Buzzega et al., DER++, NeurIPS 2020, arXiv:2004.07211
- Wang et al., L2P, CVPR 2022
- Wang et al., DualPrompt, ECCV 2022
- Smith et al., CODA-Prompt, CVPR 2023
- Wang et al., O-LoRA, EMNLP-F 2023, arXiv:2310.14152

**Document AI CL precedents (limited):**
- Minouei et al., "CL for Table Detection," Applied Sciences 2022
- Wójcik et al., "Domain-Agnostic NA for CIL," arXiv:2307.05399, 2023
- Kumar et al., ProtoNER, DAS 2024
