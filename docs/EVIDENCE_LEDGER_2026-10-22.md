# Evidence ledger — every result the CVPR 2027 paper rests on (2026-10-22)

One row per claim. *Status* is the strength of the evidence actually on disk, not the strength of
the prose. Seeds are always 42 / 7 / 123 unless stated. Source paths are relative to `CLUE/`.
This supersedes §1 of `docs/RESULTS_REPORT_2026-10.md` and the F-list of
`docs/RESULTS_SYNTHESIS_CVPR2027_2026-10-14.md` for the question "what do we have evidence for".

Legend: **3S** = three seeds on every cell cited · **1S** = seed-42 diagnostic · **NULL** =
pre-registered negative, reported as such · **PEND** = runs not finished · **—** = not applicable.

## A. Diagnosis (documents)

| # | Claim | Evidence (numbers in the paper) | Status | Source | Paper |
|---|---|---|---|---|---|
| F1 | Forgetting is output-localised (head + late layers) | CKA by depth 1.00→0.78→0.34→0.25→0.16; Fisher-weighted displacement 1–4 orders larger at head; uniform-displacement null rejected p<0.01; BWT −81/−88/−79 across modality masks | 3S × 4 backbones × 4 masks | `results/pilot/*`, `doccl/eval/fisher.py`, `fig/pilot_displacement_bars.pdf` | §4 Finding 1, Fig. 2 |
| F1′ | …and architecture-general (shared *location*, magnitude not claimed) | permutation test LiLT p=0.38, BROS p=0.51; BERT same profile; ViT head share 97–100 % (3/3 seeds) | 3S | `results/pilot/cv_vit_*` | §4 Finding 1′ |
| F2 | Protecting the locus relocates the drift (conservation) | freeze-all 41.1/−0.0; freeze head+late 38.8/−76.8, early CKA 1.0→0.19; frozen locus + memory 38.9/−75.6; uniform consolidation 42.6 vs head/late 84.7, head-only 86.1, late-only 86.2 | freeze arms **1S**; consolidation probe 3S; extended to 3 DIL orders + 10-session CIL (1S) | `results/freeze_*`, `results/consol_*`, `figures/conservation.pdf` | §4 Finding 2, Fig. 3 |

## B. Benchmark (documents)

| # | Claim | Evidence | Status | Source | Paper |
|---|---|---|---|---|---|
| F3 | Replay reaches the joint bound on DIL on every backbone; regularisation/distillation stay at the naive floor | ER/DER++ 87.9/88.0 (LMv3), 85.7/84.4 (LiLT), 87.8/87.6 (BROS), 77.1/77.4 (BERT) vs joint 88.7/83.8/87.5/77.2; EWC/LwF 41.2/43.7 vs naive 41.3 | 3S, 68/72 cells (4 OOM cells disclosed) | `scripts/analyze_results.py` → `tab:grid` | §5, Table 1 |
| F3a | Independent per-dataset bound ≈ joint on DIL (no transfer headroom) | 88.2/85.3/88.8/77.3 vs joint; ≥ joint on LiLT, BROS | 3S | `results/dil_indep*` | §5 |
| F4 | Remedy benefit is backbone-dependent, the phenomenon is not | naive→joint gap on Mixed +33.7/+31.1/+31.4/+31.6 (2.5-pt spread); EWC benefit −11.8/−2.6/+3.5/+5.3 (17-pt sign flip) | 3S | `figures/remedy_benefit.pdf` | §5, Fig. 4 |
| F8 | CIL-CORD is a floor regime; replay is *below* naive | all methods 16–24 vs joint 33; ER 16.0, DER++ 18.2, naive 19.0; EWC 0.5 | 3S × 4 backbones | `tab:grid`, `tab:main` | §5 |

## C. Falsification (documents, DIL/LayoutLMv3)

| # | Claim | Evidence | Status | Source | Paper |
|---|---|---|---|---|---|
| F5 | Five buffer-free families fail for one mechanism (readout-marginal snap) | TIES/Fisher merge 21.6 (no-merge 21.9); LCA 43.9±2.5; HGT/CUBER 39.8 (own naive 38.7), HGT ≤25.3; LexSlot (no buffer) 42.2; Ledger FUNSD 21 / SROIE 23; LexMem v3b 66.0, v5 63.2; marginal cosine 0.997–0.998; KEY/HEADER → 0.0 F1 | LCA 3S; others 1S frozen-backbone operating point († in table) | `results/hgt_ablation/*`, `results/dil_lca_seed*`, `results/lexmem*` | §6, Table 2 |
| F5a | Cheap head-side corrections are registered nulls | 5 kill-tests ≤ −0.1 AA; old-class mass ≈ 1e-6 | NULL, pre-registered (`docs/RCA_KILLTESTS_PREREG_2026-07.md`) | `results/marginal_kl*`, `results/logit_adjust*` | §6 |

## D. What the memory must hold (documents, DIL/LayoutLMv3)

| # | Claim | Evidence | Status | Source | Paper |
|---|---|---|---|---|---|
| F6 | Whole documents retain; every summary of the same store collapses | raw bank 50 docs/task 87.3 (k=4); rank-16 spectral 41.9; per-class Gaussians 39.4; k-means centroids 36.7–39.7 | 1S ladder | `results/latent_replay*`, `results/spectral*`, `results/aglr*`, `results/coreset*` | §7, Fig. 5 |
| F6a | Binding is worth +27 AA (single-variable control) | 4 whole docs 63.8 vs same features on decoupled carriers 36.7 (matched 5-epoch budget) | 1S | `results/gates/binding_*` | §7 |
| F6b | **CoLaR r=128 matches raw-exemplar replay at 60 MB, no raw bytes** | 87.6 (seed 42); **87.8 ± 0.5** (87.58/88.30/87.38) 10-epoch protocol; ER 87.9 @ 200 docs; raw latent 87.3 @ 163 MB (2.7×); r=64 80.6 @ 32 MB | **3S headline** | `results/dil_colar_e10_seed*_k4_d50_r128` (restored from logs; 5-epoch control 86.6±0.3 is *not* the headline) | abstract, §7, Table 3 |
| F6c | Memory-side levers obey a conservation law | balanced 87.1, coverage 86.8, entity-weighted 87.1 (1S); 3S check balanced 87.8 = base, k-centre 87.0 | 3S check | `results/dil_colar_bal*`, `colar_cb*`, `colar_wsvd*` | §7 |
| F6d | Learned codecs / position-axis compression break binding | AE c=64 @ 56 KB 31.5 vs SVD r=16 @ 39 KB 40.9; labelled-tokens-only 79.6; drop visual positions 61.2 | 1S | Amendment 9 probes | supp |
| F7 | Intelligence on top of the memory is null | lexical routing +1.8/+0.4/−0.6 (mean +0.5, sign flip); CoLaSlot d50 +0.19 AA for +77 % runtime, d5 +1.9/−0.66/−10.93; kNN blend inert/destructive; metaplastic 85.9→79.5 | NULL, 3S (routing pre-registered, Amendment 10) | `results/gates/*far*`, `results/dil_colaslot*` | abstract, §7 "Routing" |
| F7a | Public-proxy replay is a bounded positive | PLaR 58.9 (FUNSD 76.1, SROIE ≈7) | 1S | `results/dil_proxy_latent*` | §7 |
| F7b | Budget-adaptive allocation is a frontier, not a gain | +4.0 / −8.0 / +5.7 → +0.6 ± 7.5 | NULL, 3S | `results/dil_colar_adaptive*` | supp |

## E. Label growth (CIL-CORD, Mixed; CoLaR-R programme, R1/R2)

| # | Claim | Evidence (AA, sample s.d.) | Status | Source | Paper |
|---|---|---|---|---|---|
| F10 | Banked labels go stale under label growth; CoLaR sits at the naive floor on CIL-CORD | CoLaR 20.4 ± 1.3 (21.7/19.0/20.5) vs naive 19.0, joint 33.3; Mixed 58.8 ± 1.1 | 3S | `results/cil_cord_colar_seed*_k4_d50_r128`, `results/mixed_colar_seed*_k4_d50_r128` | §7 "live labels", Table 3 |
| F10a | True-label refresh (ledger oracle) is the reachable ceiling | CIL-CORD 26.3 ± 3.6 (+5.1/+3.5/+9.1 vs plain); Mixed 64.4 ± 1.9 | 3S, not deployable | `results/*_colar_r_oracle_seed*` | §7, Table 2 ‡ |
| F10b | **Sliced softmax (R2b) is the one deployable arm that clears its bar on CIL-CORD** | 25.4 ± 2.4 (27.7/22.9/25.5), +6.0/+3.9/+5.0 vs plain, 3/3 sign; Mixed 61.3 ± 1.9, +4.5/+3.1/−0.1 → bar missed (1 tie) | 3S, pre-registered; **gain on CIL-CORD, null on Mixed** | `results/*_colar_r_sliced_seed*` | abstract, contrib. 5, §7, Table 3 "CoLaR-R, sliced" |
| F10c | Teacher-free pseudo-relabel is a null | CIL-CORD 20.0 ± 4.1 (23.7/15.6/20.6) vs bar 23.0; Mixed 59.5 ± 0.0; relabel counts 0 at every boundary on s42/s7, 7/0/0/36 on s123 vs oracle 3231/2792/0/305 | NULL, pre-registered (Gate 1, 2026-10-16/17) | `results/*_colar_r_pseudo_seed*` | §7, Table 2 ‡ |
| F10d | Sliced + pseudo does not add | 20.7 ± 1.0 / 60.1 ± 1.1 | NULL | `results/*_colar_r_sliced_pseudo_seed*` | not in paper (dominated) |

## F. Fixed twelve-row reference set (frozen 2026-10-21)

| # | Claim | Evidence | Status | Source | Paper |
|---|---|---|---|---|---|
| F11 | Document side: CoLaR is the only non-oracle row at the DIL oracle without raw bytes; the continual-NER comparators collapse | DIL: ER 87.9, DER++ 88.0, CoLaR 87.8, LexSlot 87.3, joint 88.7; CODA 32.5, CL-LoRA 41.1, CPFD 6.2 ± 7.3, IS3 38.3 (1 seed), LCA 43.9. CIL-CORD: CL-LoRA 15.2 ± 3.1, LCA 8.2 ± 2.6, CoLaR-R sliced 25.4 (best). Mixed: DER++ 60.8, CoLaR-R sliced 61.3, IS3 30.1, LCA 19.0 | 3S except IS3-DIL (1S); **four null cells** (CODA CIL-CORD; CPFD CIL-CORD, Mixed; IS3 CIL-CORD), each with a diagnosed cause | `paper_cvpr/tab_main_core.tex` ← `scripts/analyze_results.py`; nulls: `results/phase1/*`, supp "Null cells" | Table 3, Limitations, supp §B |
| F9 | Image side: the ranking inverts — a buffer-free Gaussian head beats latent replay; the trunk is intact | SLCA 87.5/65.0 > bank 80.0/55.1 > ER 76.2/54.8 ≈ DER++ 76.5/52.8; RanPAC 85.3/58.5; LCA 77.2 ± 3.2 (CIFAR); SLCA-noCA 38.0/25.8; joint 89.5/77.6; CoLaR r128 55.3 vs bank 55.1 (lossless) | 3S on every finished cell | `paper_cvpr/tab_image_core.tex`, `tab_image_supp.tex` ← `scripts/rca/make_image_table.py` | abstract, contrib. 5, §7 "domain-discriminating", Table 4, supp Table 2 |
| F9a | Ridge decomposition: the loss is readout coverage, not trunk damage | head 78.8/56.2; ridge on stored bank 80.8/58.0; ridge on full data 88.3/72.3; ridge on frozen trunk 85.4/57.3; head gap 9.5/16.1, coverage gap 7.5/14.3, trunk gap 2.9/15.0. Doc CoLaR cells: head gap 0.3 (DIL), 7.7 (CIL-CORD), 0.7 (Mixed), coverage gap ≤ 0.6 | **1S diagnostic**, never a method row | `results/*/probes.json` (Pass 2b), `tab_probe_decomp.tex` | §7, supp Table 3 |
| F9b | Pending image cells | CODA-Prompt and CL-LoRA on both datasets; LCA on ImageNet-R (5 cells) | **PEND** — CODA ImageNet-R s42 running (≈1.9 min/epoch, early stop not binding); local completion > 1 week | `results/logs_vision/*` | Table 4 "pending", Limitations |

## G. What is *not* evidence (and is not claimed)

- Cross-backbone forgetting *magnitude* differences (underpowered at 3 seeds).
- Any CIL-CORD reading below three seeds (two runs differing by 3 memory tokens differ by 2 AA).
- The ridge probe as a method (it re-reads all past training images at the end).
- The LexSlot probe cells (`*_lexslot_seed42/probes.json`): taken on the buffer-free run, not the `_off` run behind the LexSlot row; excluded from the paper.
- Replay-memory bytes in `metrics.json` for the image runs (`replay_memory_bytes` counts the uncompressed bank even for CoLaR r128); the 60 MB figure is the document-side measurement.

## H. Paper state vs. this ledger (2026-10-22 22:00)

Everything in A–F is in `paper_cvpr/main.tex` + `supp.tex` and on Overleaf (project 6acd1a2d…).
Known gaps: (i) five image cells pending (F9b); (ii) the body is **12 pages** against the CVPR
8-page limit — the ledger says which rows carry claim weight (F1, F2, F3, F5, F6b, F10b, F9) and
which can move to the supplement wholesale (F6c/F6d/F7a/F7b, the HGT/CUBER and Ledger ledger rows,
the routing paragraph); (iii) Finding 2 freeze arms and the F6 ladder remain seed-42.
