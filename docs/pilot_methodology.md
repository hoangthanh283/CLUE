# Pilot Study Methodology

This document is the design contract for the pilot study (Phase 2, Weeks 3-4).

## Goal

Characterize *where* and *how* catastrophic forgetting occurs in LayoutLMv3
when fine-tuned sequentially on three document understanding datasets
(FUNSD → CORD → SROIE).

The result drives method selection for the AAAI submission.

## Setup

### Conditions (4 × 3 = 12 runs)

| ID | Active modalities | Implementation |
|----|----|----|
| C1 | text only | LayoutLMv3 with `ModalityMask.TEXT_ONLY` (image and bbox zeroed) |
| C2 | image + layout | LayoutLMv3 with `ModalityMask.IMAGE_LAYOUT` (text → PAD tokens) |
| C3 | text + layout | LayoutLMv3 with `ModalityMask.TEXT_LAYOUT` (image zeroed) |
| C4 | full | LayoutLMv3 with `ModalityMask.FULL` |

**Why same architecture for C1-C4?** Apples-to-apples comparison. A naive
approach would be "BERT vs LayoutLMv3" — but architectural differences
(self-attention layers, positional encodings, layer counts) confound the
modality contribution. Running the *same* LayoutLMv3 with different inputs
masked at the input layer keeps everything else fixed.

**Caveat:** C1 (TEXT_ONLY) is not a true BERT comparison — it's
"LayoutLMv3 with all non-text inputs zeroed", which still has 2D position
embeddings (just receiving zero positions) and visual patches (just zero pixels).
These additional input pathways may still propagate signal. A true BERT
comparison would require a separate `BERTWrapper` (`doccl/models/bert_wrapper.py`,
not yet implemented). For AAAI submission, we use the same-architecture
TEXT_ONLY mask as the closest apples-to-apples text baseline; if pilot results
look suspicious, we add the true BERT wrapper as a follow-up.

### Task sequence

FUNSD (149 train / 50 test, 7 BIO tags)  
→ CORD (800 train / 100 test, 61 BIO tags, fine-grained)  
→ SROIE (626 train / 347 test, 9 BIO tags)

Sequential, naive fine-tuning (no CL strategy). Classifier head expanded at
each new task to accommodate new labels.

### Seeds

3 seeds: 42, 123, 7. Pilot must be reproducible — without 3 seeds we cannot
distinguish noise from signal in CKA/Fisher patterns.

## Metrics

### CKA (Kornblith et al. ICML 2019)

Linear CKA between activations of the *same model* at different checkpoints
(after task t-1 vs after task t).

Captured at 6 layers (low/mid/high text + embeddings + patch_embed + classifier;
see `doccl/pilot/run_pilot.py::LAYERS_TO_TRACK`).

**Interpretation:** CKA = 1 means representations unchanged; CKA = 0 means
representations completely reorganized. Layers with CKA drops most are the
"forgetting hotspots".

### Fisher information per parameter group

Empirical Fisher (squared gradient norm) computed on each task's data, aggregated
into 8 groups: text_word_embed, layout_2d_pos_embed, image_patch_embed, text_attn,
visual_attn, fusion, ffn, classifier.

**Interpretation:** Group-level Fisher tracks how *important* each component is
to the current task. Big drops between tasks indicate components losing
their representations.

### Forgetting matrix (R[i, j])

Standard CL accuracy matrix. R[i, j] = F1 on task j after training task i.

From this: AA = mean of last row, BWT = mean of (R[T-1, i] - R[i, i]) for i < T-1,
AF = -BWT (per Lopez-Paz & Ranzato NeurIPS 2017).

## Hypotheses

- **H₀:** Forgetting is uniform across components (CKA drops equally everywhere,
  Fisher proportional to baseline scale).
- **H₁:** Forgetting concentrates in specific components.

Sub-hypotheses (not pre-registered, but the patterns we look for):

- **H_a (fusion focus):** Fusion-related parameters (cross-modal attention
  projections) show largest CKA drops in C4 but not in C1/C2/C3 ablations
  → triggers Candidate A (LAPP + H-LoRA).
- **H_b (position drift):** 2D position embeddings show largest Fisher drops
  uniformly across conditions where layout is active (C2, C3, C4)
  → triggers Candidate B (Layout-Protected EWC).
- **H_c (scenario-dependent):** Forgetting pattern changes between FUNSD↔CORD
  vs CORD↔SROIE transitions
  → triggers Candidate C (Modality-Routed Prompts).

## Decision rule

End of Week 4, with advisor:

| Pattern | Action |
|---|---|
| H_a strongly supported | Implement Candidate A |
| H_b strongly supported | Implement Candidate B |
| H_c strongly supported | Implement Candidate C |
| Mixed evidence | Implement most-feasible candidate, plan ablations |
| H₀ cannot be rejected | Pivot to characterization-only paper |

"Strongly supported" = the relevant statistical comparison (Mann-Whitney U
on per-component CKA drop) shows p < 0.05 with Bonferroni correction across
8 parameter groups.

## What's *not* in the pilot

- ViT comparison: deferred. Not directly comparable to LayoutLMv3 since ViT
  doesn't do token classification. The "C2 = LayoutLMv3 with text masked"
  serves as the visual-stream-only proxy.
- Larger backbones (LayoutLMv3-large): deferred to NeurIPS extension.
- Multilingual (LayoutXLM): deferred to NeurIPS extension.
- Hyperparameter sensitivity: pilot uses fixed defaults (lr=5e-5, 10 epochs,
  batch=8). Sensitivity is a separate ablation in the main grid.

## Outputs

- `results/pilot/{condition}_seed{seed}.json` — per-run dump (CKA, Fisher, F1)
- `results/pilot/aggregated.parquet` — long-format dataframe (after analyze.py)
- `results/pilot/figures/cka_heatmap.pdf` — main paper figure 4.1
- `results/pilot/figures/fisher_bars.pdf` — main paper figure 4.2
- `results/pilot/figures/forgetting_matrix.pdf` — main paper figure 4.3
- `results/pilot/findings_summary.md` — auto-generated text summary

## Time budget

- Setup pilot infrastructure: 2 days (W3 days 1-2)
- Run 12 sequential training runs: ~18 GPU-hours (W3 days 3-5, parallel on Vast.ai)
- Analysis + plotting: 1 day (W4 day 1)
- Statistical tests: 1 day (W4 day 2)
- Writing memo: 1 day (W4 day 3)
- Advisor meeting + decision: W4 day 4
- Buffer: W4 days 5-7

Total: ~14 days, fits in Phase 2 envelope.
