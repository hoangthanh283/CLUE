# Full comparison — AA across methods × scenarios × seeds (2026-10-21)

Numbers are AA (average accuracy over all seen tasks). Where available, 3-seed means
± SD are given; otherwise seed-42 only. Pre-training backbone: ViT-B/16 on images,
LayoutLMv3-base on documents. Memory column: replay footprint in MB per final checkpoint
(empty = no buffer).

## Image CIL (ViT-B/16, 10 sessions)

| method | year | memory | CIFAR-100 s42 | 3-seed | ImageNet-R s42 | 3-seed |
|---|---|---|---|---|---|---|
| naive                     | —    | — | 10.6 | 11.7 ± 1.1 | 9.6  | 9.3 ± 0.3 |
| joint (oracle)            | —    | all | 89.1 | 89.5 ± 0.4 | 77.4 | 77.6 ± 0.5 |
| EWC                       | 2017 | — | 14.7 | 13.9 ± 1.5 | 9.4  | 9.3 ± 0.2 |
| LwF                       | 2017 | — | 15.1 | 15.7 ± 0.6 | 12.5 | 12.7 ± 1.4 |
| ER @200/cls               | 2019 | 4 MB  | 44.1 | 42.3 ± 2.7 | 20.9 | 21.6 ± 1.3 |
| ER @2000                  | 2019 | 40 MB | 75.6 | 76.2 ± 1.2 | 54.0 | 54.8 ± 0.7 |
| DER++ @200                | 2020 | 4 MB  | 48.2 | 43.6 ± 4.0 | 22.7 | 20.2 ± 2.5 |
| DER++ @2000               | 2020 | 40 MB | 77.0 | 76.5 ± 1.5 | 51.5 | 52.8 ± 1.1 |
| SimpleCIL (frozen)        | 2023 | — | 81.7 | 81.7 | 48.2 | 48.2 |
| RanPAC (frozen)           | 2023 | — | 85.4 | 85.3 ± 0.1 | 58.9 | 58.5 ± 0.3 |
| SLCA-noCA                 | 2023 | 40 MB | 38.4 | 38.0 ± 3.9 | 25.1 | 25.8 ± 0.7 |
| **SLCA**                  | 2023 | 40 MB | **87.4** | **87.5 ± 0.2** | **65.3** | **65.0 ± 0.3** |
| CoLaR raw bank d500 | 2026 | 1.5 GB | 80.5 | 80.0 ± 0.6 | 56.6 | 55.1 ± 1.3 |
| CoLaR r=128 d500    | 2026 | 60 MB  | — | — | 53.8 | 55.3 ± 1.4 |
| **Pass 2b ridge ceiling** | 2026 |  | **88.3** |  | **72.3** |  |

**Reading:** on image CIL, the Pass 2b ridge probe says the adapted CoLaR trunk can
reach 88.3 (CIFAR) and 72.3 (ImageNet-R) with an analytic head — above SLCA by 0.8 pp
on CIFAR and by 7.3 pp on ImageNet-R. CoLaR's current plastic head wastes 9.5 pp and
16.1 pp respectively.

## Document CL (LayoutLMv3-base)

| method | year | memory | DIL s42 | 3-seed | CIL-CORD s42 | 3-seed | Mixed s42 | 3-seed |
|---|---|---|---|---|---|---|---|---|
| naive              | —    | — | 40.1 | 41.3 ± 1.5 | 18.8 | 19.0 | 34.1 | 33.9 |
| joint (oracle)     | —    | all | 89.7 | 88.7 ± 1.1 | 33.4 | 33.3 | 67.5 | 67.5 |
| EWC                | 2017 | — | 40.2 | 41.2 ± 1.1 | 1.1  | 0.5  | 16.8 | 22.0 |
| LwF                | 2017 | — | 42.5 | 43.7 ± 1.1 | 19.0 | 18.8 | 35.1 | 35.2 |
| ER                 | 2019 | 163 MB | 86.2 | 87.9 ± 1.4 | 16.9 | 16.0 | 64.3 | 60.3 |
| DER++              | 2020 | 163 MB | 88.2 | 88.0 ± 0.9 | 17.9 | 18.2 | 60.9 | 60.8 |
| CPFD               | 2023 | — | — | — | 2.7  | 1.4 (n=2) | 6.0  | 2.0 (n=2) |
| IS3                | 2024 | — | — | — | 5.0  | 4.4 (n=2) | 28.8 | 30.1 (n=2) |
| LexSlot (buffer-free) | 2026 | 0 | 40.5 | 38.7 | 19.2 | 18.9 | 33.1 | 28.7 |
| **CoLaR k4/d50/r128** | 2026 | 60 MB | **86.4** | **86.8 ± 0.6** | 21.7 | 20.4 ± 1.1 | 57.5 | 58.75 ± 0.9 |
| CoLaR-R oracle (†) | 2026 | 60 MB | — | — | 26.8 | 26.3 ± 2.9 | 63.5 | 64.4 ± 1.6 |
| CoLaR-R pseudo     | 2026 | 60 MB | — | — | 23.7 | 20.0 ± 3.4 | 59.5 | 59.5 ± 0.0 |
| **CoLaR-R sliced** (R2b) | 2026 | 60 MB | — | — | **27.7** | **25.4 ± 2.0** | 62.0 | 61.3 ± 1.6 |
| CoLaR-R sliced+pseudo (R2d) | 2026 | 60 MB | — | — | 19.5 | 20.7 ± 0.8 | 60.5 | 60.1 ± 0.9 |
| CoLaSlot-shadow (= CoLaR+LexSlot) | 2026 | 60 MB | **88.3** | — | — | — | — | — |
| **Pass 2b ridge ceiling** | 2026 |  | **91.4** |  | **75.2** |  | **71.6** |  |

(†) Oracle uses the original per-document labels at session capture time (ledger). Not
deployable; measures the stale-label cost.

**Reading (documents):** DIL is saturated — plain CoLaR 86.4 vs ridge 91.4, and CoLaSlot
(the hybrid) wins +2 AA with a 2.5× slower training recipe. On CIL-CORD, the sliced-
softmax arm (R2b) is the first and only arm clearing the pre-registered bar (3/3 sign,
+5.0 vs plain). On Mixed, oracle 64.4 clears the bar; sliced near-misses (+2.5 mean, s123
tie). All CoLaR variants' ridge ceilings sit at ~71-75 on CIL-CORD and Mixed, so
head-side gains cap at ~7-15 pp — unlike image side where the lane is clean.

## Where CoLaR / LexSlot "lose to" state of the art, by the probe

| scenario | current best in-house | SOTA ref | ridge ceiling | gap source |
|---|---|---|---|---|
| CIFAR-100 | CoLaR 80.5 | SLCA 87.5 | 88.3 | **80% readout coverage, 20% statistics** |
| ImageNet-R | CoLaR 55.3 | SLCA 65.0 | 72.3 | **readout coverage 14 pp + adapted trunk +15 pp over frozen** |
| CIL-CORD | sliced 25.4 | — (no SLCA for docs) | 75.2 | **plastic head 7.7 pp; features saturated** |
| Mixed | sliced 61.3 | joint 67.5 | 71.6 | plastic head 0.7 pp; AA shortfall is seqeval structure |
| DIL | CoLaR 86.4 | joint 88.7 | 91.4 | plastic head 0.3 pp; saturated |

## Method years & venues (verified from `paper_cvpr/refs.bib`)

| method | year | venue / citation key |
|---|---|---|
| EWC | 2017 | PNAS; `kirkpatrick2017ewc` |
| LwF | 2017 | TPAMI (ECCV 2016 ext.); `li2017lwf` |
| ER | 2019 | NeurIPS; `rolnick2019er` |
| DER++ | 2020 | NeurIPS; `buzzega2020der` |
| L2P | 2022 | CVPR; `wang2022l2p` |
| DualPrompt | 2022 | ECCV; `wang2022dualprompt` |
| CODA-Prompt | 2023 | CVPR; `smith2023coda` |
| CPFD | 2023 | EMNLP; `zhang2023cpfd` |
| SLCA | 2023 | ICCV; `zhang2023slca` |
| SLCA-noCA | 2023 | ICCV ablation of SLCA |
| SimpleCIL | 2023 | arXiv (ICCV workshop extension) |
| RanPAC | 2023 | NeurIPS; `mcdonnell2023ranpac` |
| O-LoRA | 2023 | EMNLP; `wang2023olora` |
| IS3 | 2024 | ACL (continual NER); `liu2023ice` / `is3_acl24` |
| LCA | 2026 | ICLR; `lca2026` |
| CL-LoRA | 2025 | ACL; `he2025cllora` |
| C-Flat / C-Flat++ | 2024 | NeurIPS; `bian2024cflat` |
| **LexSlot / CoLaR / CoLaR-R / CoLaSlot-shadow** | **2026 (this project)** | the DocCL/CLUE repo |
| **Pass 2b ridge ceiling** | **2026 (this project)** | probe measurement, not a method |
