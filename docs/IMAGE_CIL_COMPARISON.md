# Image class-incremental learning — CoLaR / CoLaR++ vs. baselines (2026-10-07)

ViT-B/16 (ImageNet-21k), 10 tasks. Final average accuracy (AA), average incremental accuracy
(Inc-Acc), backward transfer (BWT); mean ± s.d. over seeds 42/7/123 unless `n=1`.
**Ours** = our runs under one recipe (AdamW 5e-5, early stop, bs 16, local RTX 2060).
**Published** = numbers reported by the papers under their own recipe (bs 128, 20–50 epochs,
stronger augmentation) — context only, not directly comparable (our joint is 4 pp below theirs
on CIFAR-100, 2 pp on ImageNet-R).

## 1. Split CIFAR-100 (10 × 10)

| Family | Method | Memory stored | AA | Inc-Acc | BWT | Source |
|---|---|---|---|---|---|---|
| Bounds | Naive fine-tuning | — | 11.7 ± 1.1 | 31.5 | −95.6 | ours |
| | Joint (oracle) | all data | 89.5 ± 0.4 | — | — | ours |
| Regularisation | EWC | — | 13.9 ± 1.5 | 35.3 | −93.1 | ours |
| | LwF | — | 15.7 ± 0.6 | 35.2 | −91.0 | ours |
| Exemplar replay | ER, 200 images | 0.6 MB¹ | 42.3 ± 2.7 | 66.9 | −61.6 | ours |
| | ER, 2 000 images | 6 MB¹ | 76.2 ± 1.2 | 86.8 | −23.8 | ours |
| | DER++, 200 images | 0.6 MB¹ | 43.6 ± 4.0 | 68.2 | −60.6 | ours |
| | DER++, 2 000 images | 6 MB¹ | 76.5 ± 1.5 | 86.3 | −23.6 | ours |
| Buffer-free, PTM | Slow trunk only (SLCA w/o alignment) | — | 38.0 ± 3.9 | 57.8 | −66.7 | ours |
| | **SLCA** | class Gaussians (0.2 MB) | **87.5 ± 0.2** | **92.0** | −7.8 | ours |
| | Gaussian latent replay (AGLR) | 0.7 MB | 10.7 ± 0.8 | 31.9 | −96.5 | ours |
| **Latent replay (ours)** | Raw latent bank, k=4, 500/task | 1 513 MB | 80.0 ± 0.7 | 87.0 | −19.8 | ours |
| | CoLaR++ + head alignment | 1 513 MB | 80.6 (n=1) | 86.8 | −19.3 | ours |
| | CoLaR++ + slow trunk | 1 513 MB | 77.2 (n=1) | 85.5 | −21.8 | ours |
| | CoLaR++ all of CA/SL/WA/BAL | 1 513 MB | 74.6 (n=1) | 83.5 | −25.6 | ours |
| | CoLaR++ r16 + int8 + pool, 1 epoch (exploratory) | **66 MB** | 81.2 (n=1) | 88.1 | **−1.5** | ours |
| Published (context) | Seq. fine-tuning (slow LR) | — | 88.86 | 92.01 | | SLCA Tab. 1 |
| | L2P | prompts | 82.76 | 88.48 | | SLCA Tab. 1 |
| | DualPrompt | prompts | 85.56 | 90.33 | | SLCA Tab. 1 |
| | SLCA | Gaussians | 91.53 | 94.09 | | SLCA Tab. 1 |
| | RanPAC | random projection | 92.2 | | | RanPAC |
| | Joint | | 93.22 | | | SLCA Tab. 1 |

## 2. Split ImageNet-R (10 × 20)

| Family | Method | Memory stored | AA | Inc-Acc | BWT | Source |
|---|---|---|---|---|---|---|
| Bounds | Naive | — | 9.3 ± 0.3 | 26.6 | −87.1 | ours |
| | Joint | all data | 77.6 ± 0.5 | — | — | ours |
| Regularisation | EWC / LwF | — | 9.3 / 12.7 | 26.9 / 35.1 | −87 / −84 | ours |
| Exemplar replay | ER, 200 / 2 000 | 30 / 300 MB¹ | 21.6 / 54.8 | 44.2 / 71.0 | −75 / −38 | ours |
| | DER++, 200 / 2 000 | 30 / 300 MB¹ | 20.2 / 52.8 | 43.9 / 69.4 | −77 / −41 | ours |
| Buffer-free, PTM | Slow trunk only | — | 25.8 ± 0.7 | 44.2 | −67.4 | ours |
| | **SLCA** | Gaussians | **65.0 ± 0.3** | **72.0** | −13.5 | ours |
| **Latent replay (ours)** | CoLaR / CoLaR++ | — | *not run yet* | | | |
| Published (context) | Seq. fine-tuning | — | 71.80 | 76.84 | | SLCA Tab. 1 |
| | L2P / DualPrompt | prompts | 66.49 / 68.50 | 72.83 / 72.59 | | SLCA Tab. 1 |
| | SLCA | Gaussians | 77.00 | 81.17 | | SLCA Tab. 1 |
| | Joint | | 79.60 | | | SLCA Tab. 1 |

¹ ER memory counted as raw uint8 images (CIFAR 32×32×3 = 3 KB; ImageNet-R at 224² = 150 KB).
Our implementation stores the transformed tensors, so its actual RAM use is higher.

## 3. Reading

1. **Where CoLaR stands.** The raw latent bank (80.0) beats every exemplar baseline at the
   matched 500-sample scale and ER/DER++ with 2 000 images (+3.5–4 pp), but trails SLCA
   (−7.5 pp) and joint (−9.5 pp) under the same recipe. On the byte axis it is far worse on
   CIFAR (1.5 GB vs 6 MB) — raw images are tiny there — so CoLaR's case must be made on
   compression (the 66 MB exploratory point) and on ImageNet-R.
2. **What blocks it.** Phase A showed the stored latents are memorised within a few epochs
   (replay loss → 0); head alignment, slow trunk, weight aligning and balanced replay do not
   fix that (80.6 at best). Phase A′ (latent distillation, token-drop replay, epoch cap) is
   running now; the go criterion remains ≥ 87.5.
3. **The strongest competitor is SLCA,** a buffer-free method; under the published recipe the
   PTM methods (RanPAC 92.2, SLCA 91.5) are within 1–2 pp of joint.

## 4. Gaps in the comparison (what a reviewer will ask for)

| Missing | Why it matters | Cost here |
|---|---|---|
| **SimpleCIL** (frozen PTM + class prototypes) | the standard "no training" PTM baseline | minutes; no training |
| **RanPAC** (frozen PTM + random projection + ridge) | strongest published PTM baseline | minutes; no training |
| **L2P / DualPrompt / CODA-Prompt** on ViT | prompt family; numbers exist only as published | needs ViT prompt plumbing (~80 LOC) + 6–9 GPU-h |
| **CoLaR on ImageNet-R** | the benchmark where the memory argument holds | ~3 GPU-h per seed |
| **Published recipe** (bs 128) | any SoTA claim vs. published numbers | ≥ 16 GB GPU |
