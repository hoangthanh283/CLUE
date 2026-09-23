# Pre-registration: image-classification scope test of the DocCL claims

Registered 2026-09-23, before any image run. Adjudicate only against this file.

## Why

The paper's Finding 3 ("only replay re-grounds the head; five buffer-free families fail")
and 3b ("a buffer must hold whole documents") are established on token-level document IE.
On pre-trained ViT-B/16 class-incremental image benchmarks the literature reports the
opposite for buffer-free methods: SLCA (slow backbone LR + Gaussian classifier alignment,
ICCV 2023) and RanPAC reach near-joint accuracy without exemplars. Finding 1
(head/late-localised, depth-monotone drift) agrees with Ramasesh et al. 2021 and Davari et al.
2022. This experiment tests the scope of F1/F3/F3b, not a method.

## Setup (fixed)

- Backbone `google/vit-base-patch16-224-in21k`; CLS-token linear head; plain `Linear`.
- Split CIFAR-100: 10 sessions x 10 classes; Split ImageNet-R: 10 x 20. Class order: fixed
  permutation with seed 0 (L2P convention). Head grows by session; no background class.
- Methods: naive, joint, EWC, LwF, ER (200 exemplars), DER++ (200), SLCA
  (`lca` alignment with merge disabled + 1e-4x backbone LR).
- Protocol identical to the document grid: early stop on task val accuracy (patience 2,
  delta 0.1), 100-epoch cap, best-val checkpoint carried; bs 128; AdamW; seeds 42/7/123.
- Diagnostics on naive: per-group Fisher-weighted displacement, per-layer CKA (CLS token and
  mean patch token) at every boundary.
- H3 pair: latent replay at frozen layer 8 with (a) a bank of real CLS activations,
  50 images/class; (b) per-class Gaussian summary of the same bank (`aglr_replay`).

## Hypotheses and decision rules

Effect claimed only if the sign agrees on all three seeds and |mean| > pooled s.d.

| | Hypothesis | Prediction | Support | Kill |
|---|---|---|---|---|
| H1 | Anatomy transfers | Yes | head bucket carries the largest displacement on 3/3 seeds and CKA is monotone in depth | any seed with a non-head dominant bucket |
| H2 | Remedy ordering does not transfer | Yes | SLCA within 3 pp of joint AND within 3 pp of ER on both benchmarks | SLCA below ER by > 10 pp on either benchmark (the doc-IE pattern) |
| H3 | Buffer-content law is IE-specific | Yes | Gaussian summary within 3 pp of real bank | summary below real bank by > 10 pp |

Sanity gate (blocks all adjudication): joint and SLCA on Split CIFAR-100 seed 42 within 2 pp
of the published values (SLCA 91.5, joint ~93). If the gate fails, fix the port first.

## What each outcome means for the paper

- H1 + H2 + H3 all supported: F1 is cross-domain; F3/F3b are boundary-conditioned on
  per-token heads over near-full-rank token features. Paper gains a section "what changes
  when the head is not per-token" and a mechanism question, not a retraction.
- H2 killed: "only replay" is general — stronger claim; requires the full 3-seed grid.
- Mixed: report as measured; no post-hoc hypothesis.

Optional bridge H4 (only if H2 supported): document *classification* (RVL-CDIP, 16 classes,
4 sessions x 4) on LayoutLMv3 with `aglr_replay`; support = works (within 3 pp of ER),
implying the boundary is task structure, not modality.
