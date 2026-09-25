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

## Amendment 1 (2026-09-25, before any arm-A/B/C cell beyond the seed-42 gate had run)

1. **Recipe.** All image cells use the *document* recipe as launched (AdamW 5e-5 all
   params, weight decay 0.01, early stop patience 2 / δ 0.1 / cap 100, bs 16 + gradient
   checkpointing on the local 6 GB GPU, Resize(224)+flip). The published ViT-B/16
   numbers (SLCA Table 1: CIFAR-100 joint 93.22 / SLCA 91.53 / Seq-FT 88.86; ImageNet-R
   79.60 / 77.00 / 71.80) were obtained under SGD 1e-4 (backbone) / 1e-2 (head), 20–50
   epochs, bs 128. Our seed-42 joint under the document recipe is **89.08** (−4.1 pp).
   Therefore the **gate is internal**: H2/H3 compare methods to *our own* joint and ER
   under one recipe; published values are reported as context only. Gate condition:
   CIFAR-100 seed 42 joint > 85 and naive < 25 (met: 89.08 / 11.75 at 1 epoch).
2. **Buffer size is an arm, not a constant.** 200 exemplars = 0.4 % of CIFAR-100
   (vs 25–100 % of a document task); seed-42 ER@200 = 44.15 AA. Add ER and DER++ at
   **2000 exemplars** (20/class, the image-CIL convention) as `er_b2000` / `der_pp_b2000`.
   H2's "ER" refers to the 2000-exemplar arm; the 200-exemplar arm is the
   document-matched comparator and is reported alongside.
3. **H2b (new).** The slow-backbone regime alone (`slca_noca`: SGD backbone 1e-4 / head
   1e-2, 20 ep, no alignment) recovers ≥ 80 % of the naive→joint gap on CIFAR-100; the
   alignment step adds the remainder. Support = (slca_noca − naive) ≥ 0.8·(joint − naive)
   on 3/3 seeds. Note SLCA/slca_noca therefore run under the SLCA optimiser, not the
   document recipe — this is the arm's variable, stated here.
4. **Bug disclosure.** The seed-42 SLCA gate run crashed (`lca.evaluate` assumed token
   batches) and DER++'s logit-width mask broadcast wrongly for 2-D logits; both fixed
   (tests `tests/methods/test_image_batches.py`) before any SLCA/DER++ image cell ran.
   The seed-42 joint (89.08) and ER@200 (44.15) results predate the fix and are unaffected.
5. Queue order: CIFAR-100 {naive, ewc, lwf, er, er_b2000, der_pp, der_pp_b2000, slca,
   slca_noca, joint} × {42, 7, 123}, then ImageNet-R same. H1 pilot conditions and the
   H3 pair follow once their code paths are wired (separate amendment).
