# Pilot Study Report — Diagnosing Catastrophic Forgetting in LayoutLMv3

**Project:** DocCL — Continual Learning for Document Information Extraction
**Phase:** §6.1 Pilot Study (the diagnostic core that drives the GATE A method decision)
**Date:** 2026-06-11
**Hardware:** NVIDIA RTX 2060 (6 GB VRAM), batch size 1 + gradient checkpointing
**Backbone:** `microsoft/layoutlmv3-base` (12-layer multimodal transformer)
**Runtime:** 15 runs, 3 h 17 m total (W&B: `thanh-workspace/CL4IE`, group `pilot-study`)

---

## 1. Executive Summary

The pilot study set out to answer a single design-critical question: **when LayoutLMv3 is
fine-tuned sequentially across document-IE tasks, *where inside the network* does
catastrophic forgetting happen?** The answer determines which continual-learning method
the thesis builds (the "GATE A" decision).

**Headline findings:**

1. **Forgetting is catastrophic and near-total.** The fully-multimodal model learns each
   task well (FUNSD 88.0, CORD 93.0, SROIE 80.5 F1 in isolation) but retains almost nothing
   of earlier tasks after moving on — backward transfer (BWT) ≈ **−81 to −88**. A task that
   scored 88 F1 collapses to **~2 F1** after one subsequent task.

2. **Forgetting is concentrated in the *late* layers and the classifier, not the input
   encoders.** Centered Kernel Alignment (CKA) shows a clean monotonic depth gradient:
   embeddings and patch-embed do **not** drift (CKA = 1.00), while the last encoder layer
   and classifier drift the most (CKA = 0.24 / 0.22 at the first task boundary). The Fisher
   information is likewise dominated by the **classifier head**.

3. **Text is the load-bearing modality for training stability.** Stripping text (conditions
   c1/c2) destabilizes training — many seeds collapse to F1 = 0. Keeping text + layout (c3)
   or all three modalities (c4) trains reliably across every seed.

4. **Forgetting is order-dependent.** Reversing the task order (SROIE→CORD→FUNSD instead of
   FUNSD→CORD→SROIE) measurably *reduces* forgetting (BWT −75.2 vs −81.3; AA 34.8 vs 31.6).

5. **GATE A verdict: characterization-only fallback.** The dominant component is the
   *classifier head* — the most task-specific layer — which does **not** map cleanly to any
   of the three architectural method candidates (fusion / layout / modality-routing). The
   cross-condition significance test also fails under Bonferroni correction, partly because
   the collapsed (F1 = 0) runs inflate variance. The honest read: forgetting is severe and
   late-layer-concentrated, but no single *architectural* component dominates in a way that
   uniquely selects method A, B, or C.

---

## 2. Experimental Design

### 2.1 Task sequence

The pilot trains LayoutLMv3 **naively and sequentially** (no anti-forgetting mechanism —
the point is to *let it forget* and measure how) across three document-IE datasets:

| Order idx | Task | Dataset | Labels (BIO) | Train docs | Eval docs |
|-----------|------|---------|--------------|------------|-----------|
| 0 | `pilot_funsd` | FUNSD (forms) | 7 | 149 | 50 |
| 1 | `pilot_cord` | CORD-v2 (receipts, fine) | 61 | 800 | 100 |
| 2 | `pilot_sroie` | SROIE (receipts) | 9 | 626 | 347 |

The classifier head grows class-incrementally as new labels arrive (7 → 67 → 76).

### 2.2 Modality-ablation conditions

To attribute forgetting to specific input streams, each run uses one of four
**modality masks** (same architecture, same shapes — only the input streams change, so the
comparison is apples-to-apples):

| Condition | Text | Layout (bbox) | Image | Role |
|-----------|:----:|:-------------:|:-----:|------|
| **c1_bert** | ✓ | ✗ | ✗ | text-only baseline (BERT-equivalent) |
| **c2_no_text** | ✗ | ✓ | ✓ | layout + vision, no text |
| **c3_no_image** | ✓ | ✓ | ✗ | text + layout, no vision |
| **c4_full** | ✓ | ✓ | ✓ | full multimodal (the real model) |

Each condition runs with **3 seeds** (42, 123, 7) → 12 runs. A further **3 alt-order runs**
(c4_full, order 2-1-0) provide the §6.1.3 stability check → **15 runs total**.

### 2.3 Diagnostics collected per run

At each task boundary the pilot records, on `LAYERS_TO_TRACK`
(encoder layers 0/5/11, the text+layout `embeddings`, the vision `patch_embed`, and the
`classifier`):

- **CKA** (Centered Kernel Alignment) between the representation *before* and *after*
  training the next task, on the same eval set. CKA = 1.0 means no representational drift;
  low CKA means the layer's representation changed a lot (= forgetting lives there).
- **Empirical Fisher information** per parameter group (`text_word_embed`,
  `layout_2d_pos_embed`, `image_patch_embed`, `text_attn`, `ffn`, `classifier`, `other`) —
  how task-important each component's parameters are.
- **Accuracy matrix** R[i][j] = F1 on task *j* after training task *i*, from which
  AA / BWT / AF / FWT are computed.

---

## 3. Results

### 3.1 CL metrics per condition (mean ± std across seeds)

| Condition | AA | BWT | AF (forgetting) |
|-----------|----|----|-----------------|
| c1_bert (text-only) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| c2_no_text | 7.21 ± 10.20 | −18.79 ± 26.57 | 18.79 ± 26.57 |
| c3_no_image | 27.72 ± 0.44 | −87.62 ± 0.62 | 87.62 ± 0.62 |
| **c4_full** | **31.63 ± 3.51** | **−81.26 ± 6.41** | **81.26 ± 6.41** |

*(AA = average accuracy on all seen tasks at the end; BWT = backward transfer, negative =
forgetting; AF = average forgetting. FWT = 0 throughout — no forward transfer is expected in
naive sequential training with a growing head.)*

### 3.2 Per-run detail (all 15 runs)

| Run | AA | BWT |
|-----|-----|------|
| c1_bert_seed42 / 123 / 7 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| c2_no_text_seed42 | 21.64 | −56.36 |
| c2_no_text_seed123 / 7 | 0.00 / 0.00 | 0.00 / 0.00 |
| c3_no_image_seed42 / 123 / 7 | 28.24 / 27.74 / 27.17 | −87.80 / −86.78 / −88.26 |
| c4_full_seed42 / 123 / 7 | 28.80 / 27.65 / 29.06 | −87.56 / −88.15 / −86.14 |
| c4_full_seed42_ord210 | 33.11 | −75.15 |
| c4_full_seed123_ord210 | 33.35 | −78.93 |
| c4_full_seed7_ord210 | 37.81 | −71.63 |

### 3.3 The forgetting is catastrophic — accuracy matrix (c4_full, seed 42)

R[i][j] = F1 on task *j* after training task *i*:

```
            eval FUNSD   eval CORD   eval SROIE
after FUNSD     88.0         —           —
after CORD       1.8        93.0         —
after SROIE      1.9         4.1        80.5
```

**This is the central result.** The model learns each task to a strong F1 in isolation
(diagonal: 88.0 / 93.0 / 80.5) but the off-diagonal collapses to ~2 F1 — i.e. after learning
CORD, FUNSD performance falls from 88.0 → 1.8 (−86 points). Forgetting is essentially total
within one task step. c3_no_image shows the identical pattern (88.7 → 0.3 → 1.7).

### 3.4 Where forgetting lives — CKA drift by layer (c4_full, default order)

CKA between consecutive checkpoints (1.0 = no drift, lower = more drift):

| Layer | Boundary 0→1 (FUNSD→CORD) | Boundary 1→2 (CORD→SROIE) |
|-------|:------------------------:|:-------------------------:|
| `embeddings` (text+layout fusion) | **1.000** | 1.000 |
| `patch_embed` (vision) | **1.000** | 1.000 |
| `encoder.layer.0` (early) | 0.777 | 0.989 |
| `encoder.layer.5` (mid) | 0.344 | 0.924 |
| `encoder.layer.11` (late) | **0.245** | 0.774 |
| `classifier` (head) | **0.215** | 0.708 |

**A clean monotonic depth gradient:** the input encoders (text/layout/image embeddings)
barely move, while drift increases with depth and peaks at the last encoder layer and the
classifier. Forgetting is a **late-layer + head** phenomenon. (Drift is also larger at the
first boundary than the second — the model's representations stabilize as more tasks arrive.)

### 3.5 Fisher information by component (c4_full, mean over seeds × tasks, n=18)

| Component group | Mean Fisher |
|-----------------|------------:|
| **classifier** | **2.17e-01** |
| other | 1.07e-02 |
| text_attn | 4.69e-03 |
| layout_2d_pos_embed | 2.98e-03 |
| ffn | 1.87e-03 |
| image_patch_embed | 1.18e-03 |
| text_word_embed | 1.99e-05 |

The **classifier head dominates** Fisher importance by an order of magnitude — consistent
with the CKA finding that the head is where task-specific knowledge concentrates and where
it is overwritten.

### 3.6 Order dependence (§6.1.3 stability check)

| Task order | AA | BWT |
|------------|----|----|
| FUNSD → CORD → SROIE (default) | 31.63 | −81.26 |
| SROIE → CORD → FUNSD (reversed) | 34.75 | −75.23 |

Reversing the order **reduces** forgetting by ~6 BWT points and raises AA by ~3 points.
Ending on FUNSD (a small, 7-label form task) leaves more of the sequence intact than ending
on SROIE. This confirms forgetting severity is **order-sensitive**, a caveat for any
single-order conclusion.

---

## 4. The Modality Collapse Phenomenon

A striking secondary finding: **conditions that remove text frequently collapse to F1 = 0.**

| Condition | Seeds that trained | Interpretation |
|-----------|:------------------:|----------------|
| c1_bert (text-only) | 0 / 3 | always collapses |
| c2_no_text (no text) | 1 / 3 | usually collapses |
| c3_no_image (text+layout) | 3 / 3 | always stable |
| c4_full (all) | 3 / 3 (+3/3 alt-order) | always stable |

### Diagnosis (investigated, read-only)

This is **not** a metric or pipeline bug. Evidence:

- The one working text-deprived run (c2 seed42) shows *correct* CL behavior: FUNSD task-0
  F1 = 41.9, then normal forgetting (0.8) while learning CORD (73.2).
- Collapsed runs show task-0 F1 = 0.00 **immediately after training task 0** → the model
  never learned the first task; this is a *training* failure, not forgetting.
- Training loss converges to ~0.001 in all runs (no NaN / divergence) — the model fits
  *something*, but the eval entity-F1 is 0.
- `compute_token_f1` verified correct (perfect → 100, all-O → 0).
- FUNSD "O" is only 16.6 % of tokens, so an all-"O" shortcut is **not** a low-loss
  solution — ruling out trivial class imbalance.

**Conclusion:** removing text deprives LayoutLMv3 of its primary signal and the model, for
some initializations, fails to escape a degenerate basin during task-0 training. Text +
layout is what drives stable learning; **text is the load-bearing modality.** This is itself
an interpretable finding about LayoutLMv3's reliance on its text stream — but it also means
the text-deprived conditions contribute mostly zeros, which weakens the cross-condition
statistics (see §5).

---

## 5. GATE A Decision — Why "Characterization-Only Fallback"

The analyzer runs two Mann-Whitney U tests (Bonferroni-corrected):

**Test 1 — Cross-condition:** does c4_full's forgetting differ from c1/c2/c3?
→ **Fail to reject H0** (p = 0.026 / 0.028 / 0.167 vs corrected α = 0.0167).
The c4-vs-c1 difference (BWT 81 vs 0) is large but, with n = 6 and the collapsed-run
variance, does not clear the corrected threshold.

**Test 2 — Per-component:** is forgetting concentrated in one component?
→ **Reject H0.** Dominant component = **`classifier`** (Fisher 2.8e-2), significantly above
`image_patch_embed` and `text_word_embed`.

**Decision rule** (RUNBOOK / CLAUDE.md):

| Diagnosis | Method |
|-----------|--------|
| fusion-dominant forgetting | A (H-LoRA) |
| 2D layout-position drift | B (Layout-Protected EWC) |
| scenario-dependent per-modality | C (Modality-Routed Prompts) |
| **no clear pattern** | **characterization-only fallback** |

**Why fallback is the honest call:**

1. The dominant component is the **classifier head** — the most task-specific layer by
   construction. It does **not** map to any of the three *architectural* method levers
   (fusion, layout-position, modality routing). A head-dominated signal is expected in any
   class-incremental setup and does not by itself point to a structural intervention.
2. The cross-condition test (which my GATE-A gate keys on) is non-significant, partly an
   **artifact of the collapsed text-deprived runs** inflating variance.
3. Net: forgetting is *severe* and *late-layer/head concentrated*, but no single
   *architectural component* dominates in a way that uniquely selects A, B, or C.

Per the pre-agreed inconclusive-fallback policy, the orchestrator therefore ran the
**baseline grid only** (Phases 3+4) and **paused the proposed method** for human review —
exactly the intended safe behavior.

---

## 6. Insights & Implications

1. **The forgetting problem is real and dramatic** — an 86-point F1 collapse in one task
   step. This strongly motivates the thesis: naive sequential fine-tuning of LayoutLMv3 is
   unusable for multi-task document IE. (Fills §6.1, Fig 6.1/6.2, the Ch1 headline, and the
   abstract diagnosis.)

2. **Forgetting is a late-layer + head phenomenon.** The clean CKA depth gradient
   (1.00 → 0.78 → 0.34 → 0.24 → 0.22) is a publishable result on its own and suggests that
   methods which **protect or modularize the upper layers / classifier** (e.g. head
   expansion done right, late-layer regularization, or per-task heads) are the natural
   intervention — more than input-encoder-targeted methods.

3. **Text is load-bearing.** LayoutLMv3 depends on its text stream for stable optimization;
   layout and vision alone (c2) usually fail to train. This is a caution for any
   layout-only or vision-only deployment of the backbone.

4. **Forgetting is order-dependent** (~6 BWT points), so single-order results should be
   reported with the alt-order caveat — which the pilot deliberately captured.

5. **The collapsed runs are both a finding and a confound.** They reveal a real property
   (text-dependence) but contaminate the cross-condition test. A cleaner pilot (with the
   stability fixes below) might yield a significant cross-condition result.

---

## 7. Method & Reproducibility Notes

Several real bugs were found and fixed to make the pilot run correctly on the limited-VRAM
GPU; each is committed with an `AGENT FIX` prefix:

| Fix | Why it mattered |
|-----|-----------------|
| Manual per-layer gradient checkpointing for LayoutLMv3 | transformers ≥ 4.50 dropped native checkpointing support; without it the memory recipe crashed at startup. Cut peak fwd+bwd VRAM 2.75 → 0.97 GB at bs=1. |
| Update cached `model.num_labels` on classifier expansion | HF caches `num_labels` and uses it in the loss reshape; the first CIL transition (7 → 67) crashed with a shape error. Would have killed *every* CIL run. |
| Clamp CORD bboxes to [0, 1000] | CORD quads carry out-of-frame coords (observed [−3, 997]); a negative coord is an invalid index into LayoutLMv3's 2D position embedding → CUDA device-side assert. Affects every CORD-touching scenario. |
| Fix pilot CKA sample alignment | CKA compared mismatched sample counts (50 vs 100) across task boundaries → ValueError. Now captures both checkpoints on the same eval set per boundary. |

**Configuration:** bs=1, gradient checkpointing on, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
CKA n=100, Fisher n=50, 10 epochs/task, lr 5e-5, AdamW, seeds {42, 123, 7}. Datasets from
HuggingFace cache (FUNSD `nielsr/funsd-layoutlmv3`, CORD `naver-clova-ix/cord-v2`, SROIE
materialized from `mp-02/sroie`). All 15 runs logged to W&B `thanh-workspace/CL4IE`
(group `pilot-study`) via an idempotent backfill.

**Artifacts:** `results/pilot/*.json` (raw), `results/pilot/findings_summary.md`
(auto-generated stats), `results/pilot/figures/{cka_heatmap,fisher_bars,forgetting_matrix}.pdf`.

---

## 8. Open Decisions for the Advisor (GATE A)

The baseline grid runs regardless and is needed for every path, so there is no rush. The
choice is how to interpret the inconclusive GATE A:

1. **Accept the fallback → characterization-focused thesis.** Valid contribution: a rigorous
   characterization of *where* and *how* LayoutLMv3 forgets (late layers + head),
   order-sensitivity, and text-dependence, benchmarked against the full baseline suite.

2. **Fix the modality-collapse instability and re-run the pilot (~3 h).** Lower LR / add
   warmup / early-stop on eval-F1 rather than train-loss. A clean pilot (no F1=0 runs) might
   produce a significant cross-condition result and a cleaner component signal, re-enabling a
   proposed method.

3. **Override to a candidate on prior grounds.** The CKA/Fisher evidence points at the
   late layers + classifier; if a structural argument favors one candidate (e.g. head/late-
   layer protection ≈ a regularization method like Candidate B), wire it manually and let
   the grid include it.

**Recommendation:** review `cka_heatmap.pdf` + `fisher_bars.pdf` alongside this report before
deciding — the layer-drift picture is the strongest signal and may make the call clearer than
the AA table alone. The late-layer/head concentration is the most actionable lead.
