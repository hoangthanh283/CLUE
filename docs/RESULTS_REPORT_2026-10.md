# DocCL — consolidated results report (2026-10-05)

Purpose: one place with every result obtained so far, in comparable tables, as the basis for
the publication write-up. All numbers trace to `results/*/metrics.json` on disk; aggregates
were regenerated with `scripts/analyze_results.py --source local` (document grid) and
`scripts/image_scope_verdict.py` (image scope test). Metric = average accuracy (AA) at the
end of the sequence (entity-F1 for documents, top-1 for images), mean ± s.d. over seeds
42/7/123 unless marked `*` (single seed). BWT = backward transfer. Protocol everywhere:
train each task to convergence (val early stop, patience 2, cap 100), best-val checkpoint
carried; buffers of 200 documents / 200 or 2 000 images.

---

## 1. Paper claims and their current evidence status

| # | Claim | Evidence | Status |
|---|---|---|---|
| F1 | Forgetting is **output-localised** (head + late layers) and **architecture-general** | §4 doc diagnostics on 4 backbones; §7 image diagnostics on ViT-B/16 (new) | Supported on documents (4 backbones, 3 seeds) **and on images** (3 seeds) |
| F2 | Protecting the locus **relocates** forgetting (conservation) | §4 freeze/migration arms (DIL, LayoutLMv3, seed 42; consolidation probe 3 seeds); §7 slow-trunk arm on images (3 seeds) | Supported; image arm reproduces the head-side extinction |
| F3 | Only **replay re-grounds the head**; five buffer-free families fail on doc-IE | §3 grid (68/72 cells); §5 falsification ledger | Supported **on documents**; **boundary-conditioned on images** (§7): a buffer-free Gaussian head re-grounding (SLCA) reaches joint on CIFAR-100 and beats replay on both image benchmarks |
| F3b | A buffer must hold **whole documents** (consistency law); CoLaR lossless at 2.7× | §6 replay-content ladder (DIL, LayoutLMv3) | Supported on documents; image analogue (H3) **inconclusive** — frozen-trunk latent replay does not retain on ViT at the tested settings, so the pair cannot discriminate |
| — | CA-CoLaR adaptive allocation | §6.3 | **Null** at 3 seeds (+0.6 ± 7.5) |

---

## 2. Document grid — six classical strategies × four backbones × three scenarios

Scenarios: **DIL** FUNSD→SROIE→CORD (fixed 9-tag schema), **CIL-CORD** 5 sessions × 6
classes (growing head, O-tag shift), **Mixed** 6 sessions alternating class/domain shifts.
68/72 cells complete; the 4 missing (LiLT LwF ×3 scenarios, LiLT EWC on CIL-CORD) exceeded
the 6 GB GPU even in fp16.

### 2.1 AA

| Scenario | Backbone | Naive | Joint (oracle) | EWC | LwF | ER | DER++ |
|---|---|---|---|---|---|---|---|
| DIL | LayoutLMv3 | 41.3±1.5 | 88.7±1.1 | 41.2±1.1 | 43.7±1.1 | 87.9±1.4 | 88.0±0.9 |
| DIL | LiLT | 40.1±0.5 | 83.8±1.2 | 48.4±5.9 | — | 85.7±0.5 | 84.4±0.3 |
| DIL | BROS | 40.3±1.8 | 87.5±1.3 | 45.0±4.9 | 35.1±3.1 | 87.8±0.4 | 87.5±1.1 |
| DIL | BERT | 39.4±0.3 | 77.2±0.4 | 46.6±3.0 | 40.5±0.1 | 77.1±0.9 | 77.4±0.6 |
| CIL-CORD | LayoutLMv3 | 19.0±0.2 | 33.3±0.3 | 0.5±0.6 | 18.8±0.1 | 16.0±1.4 | 18.2±0.3 |
| CIL-CORD | LiLT | 19.1±0.3 | 32.4±0.4 | — | — | 17.9±0.5 | 18.0±1.0 |
| CIL-CORD | BROS | 19.2±0.1 | 32.9±0.0 | 17.3±0.8 | 23.1±1.2 | 18.3±1.5 | 17.7±1.1 |
| CIL-CORD | BERT | 19.3±0.1 | 32.4±0.2 | 17.9±0.3 | 24.3±0.3 | 17.1±1.6 | 19.2±1.9 |
| Mixed | LayoutLMv3 | 33.9±0.3 | 67.5±0.8 | 22.0±9.4 | 35.2±1.2 | 60.3±5.3 | 60.8±2.7 |
| Mixed | LiLT | 30.7±0.7 | 61.8±0.6 | 28.1±3.0 | — | 57.0±1.4 | 55.1±2.9 |
| Mixed | BROS | 33.9±0.4 | 65.3±2.1 | 37.4±3.3 | 34.4±0.5 | 61.1±1.4 | 63.0±0.4 |
| Mixed | BERT | 23.4±0.3 | 55.0±0.5 | 28.6±0.9 | 29.4±4.2 | 50.2±0.6 | 50.7±1.3 |

### 2.2 BWT (joint omitted: recomputed as 0 / NaN by construction)

| Scenario | Backbone | Naive | EWC | LwF | ER | DER++ |
|---|---|---|---|---|---|---|
| DIL | LayoutLMv3 | −73.1±2.0 | −2.8±1.1 | −69.1±1.7 | −2.8±2.0 | −3.4±1.8 |
| DIL | LiLT | −70.3±0.6 | −30.8±5.2 | — | −1.1±0.7 | −3.7±0.4 |
| DIL | BROS | −74.2±3.1 | −58.5±7.9 | −29.7±2.0 | −3.2±1.2 | −3.8±1.4 |
| DIL | BERT | −59.9±0.3 | −35.7±1.6 | −23.6±1.9 | −3.4±1.1 | −2.7±0.9 |
| CIL-CORD | LayoutLMv3 | −93.7±0.1 | −54.8±10.5 | −93.7±0.7 | −68.3±3.9 | −86.2±4.0 |
| CIL-CORD | LiLT | −92.5±0.7 | — | — | −73.6±2.7 | −82.7±0.4 |
| CIL-CORD | BROS | −94.1±0.2 | −86.8±2.8 | −56.1±1.8 | −77.1±3.5 | −84.5±2.2 |
| CIL-CORD | BERT | −93.5±0.7 | −89.0±1.0 | −54.5±0.6 | −73.9±1.0 | −80.8±2.6 |
| Mixed | LayoutLMv3 | −66.9±0.4 | −8.4±3.0 | −66.4±1.4 | −23.5±6.9 | −34.8±2.4 |
| Mixed | LiLT | −65.8±0.5 | −32.6±5.6 | — | −23.3±2.0 | −34.9±2.4 |
| Mixed | BROS | −67.1±0.7 | −49.1±4.8 | −30.3±6.3 | −21.5±7.4 | −31.6±0.4 |
| Mixed | BERT | −61.3±1.2 | −41.0±3.0 | −24.6±5.4 | −23.2±1.6 | −27.7±3.0 |

### 2.3 Readings

- **Replay reaches the oracle on DIL on every backbone** (ER/DER++ within ≈1 pp of joint; on
  LiLT above it). Regularisation/distillation stay at the naive floor on LayoutLMv3.
- **The phenomenon is architecture-invariant, the remedy is not.** Joint−naive on Mixed is
  33.7 / 31.1 / 31.4 / 31.6 (2.5-pp spread); EWC−naive on Mixed is −11.8 / −2.6 / +3.5 / +5.3
  (17-pp spread with a sign flip). EWC on LayoutLMv3 trades BWT for plasticity (BWT −2.8 but
  AA at floor): it stops fitting new tasks.
- **CIL-CORD is a floor regime for every method including replay** (AA 16–24 vs oracle 33;
  LayoutLMv3 EWC collapses to 0.5). The moving O-tag boundary is not repaired by a
  200-document buffer carrying the old labelling.
- Mixed: replay closes most of the gap; the residual sits at the class-incremental transitions.

### 2.4 LayoutLMv3 — prompt / LoRA / 2025 baselines and the proposed-then-falsified lines

| Method | Family | DIL AA (BWT) | CIL-CORD AA (BWT) | Mixed AA (BWT) |
|---|---|---|---|---|
| ER + C-Flat++ (2025) | replay + flat minima | 88.4±0.1 (−1.8) | 15.9±1.5 (−73.2) | 62.2±0.7 (−21.8) |
| L2P | prompt pool | 30.4±0.4 (−11.5) | 1.1±1.1 (−15.7) | 11.5±0.1 (−2.8) |
| DualPrompt | prompt pool | 29.9±0.6 (−11.1) | 1.8±1.7 (−15.2) | 11.7±0.1 (−2.6) |
| CODA-Prompt | prompt pool | 32.5±1.2 (−10.5) | 0.0±0.0 (−17.9) | 8.1±1.5 (−5.9) |
| O-LoRA | per-task LoRA | 40.8±0.3 (−56.8) | 4.7±8.2 (−36.0) | 13.4±0.7 (−18.9) |
| CL-LoRA (2025) | dual-adapter LoRA | 41.1±0.3 (−58.8) | 8.9±7.7 (−36.2) | 31.4±0.6 (−61.5) |
| DocCL (diagnosis-targeted Fisher + replay hybrid) | — | 84.7±1.7 (−4.6) | 16.1±0.9 (−79.3) | 57.6±2.4 (−27.1) |
| LexSlot (slot memory **+ 200-exemplar buffer**) | — | 88.4* (−1.1); 3-seed 87.3±1.1 | — | — |

Prompt families cap at ≈30 AA on DIL (the head drifts; routing cannot repair it); LoRA
families sit at the naive floor. Nothing buffer-free leaves the floor.

---

## 3. Diagnostics — where forgetting lives

### 3.1 Documents (naive FUNSD→CORD→SROIE, LayoutLMv3 unless stated; 3 seeds)

| Instrument | Result |
|---|---|
| Per-token CKA, consecutive checkpoints, by depth | embeddings 1.00 → early 0.78 → mid 0.34 → late (L11) 0.25 → head 0.16 |
| Old-task-Fisher-weighted displacement by depth bucket | head 1–4 orders of magnitude above every encoder bucket, on C1–C4 masks, BERT, LiLT, BROS; per-component test rejects uniform, head dominant (p < 0.01) |
| Shared location across backbones (profile permutation vs LayoutLMv3) | LiLT p = 0.38, BROS p = 0.51 → cannot reject same location |
| Modality masks (AA / BWT) | full 31.6/−81.3; text+layout 27.7/−87.6; text-only 27.3/−79.1; BERT 33.1/−65.3; layout+vision 21.6/−56.4 (1 of 3 seeds converged) → not fusion-specific; text is the load-bearing stream |
| Freeze arms (DIL, seed 42) | freeze all: 41.1/−0.0; freeze head+late: 38.8/−76.8 with early/mid CKA 1.00→0.19; no-memory control 38.9/−75.6 → drift migrates |
| Depth-scaled consolidation probe (3 seeds) | uniform 42.6 (cannot fit); head/late 84.7; head-only 86.1; late-only 86.2 |
| Root cause (RCA) | readout marginal snaps to the new task's label marginal (cos 0.997–0.998 for naive/LwF; replay 0.99–1.00 to its own); never-re-exercised classes → exactly 0 F1; 5 registered eval-time/marginal kill-tests recover ≤ −0.1 AA |

### 3.2 Images (naive, ViT-B/16, Split CIFAR-100, 5 epochs/task, 3 seeds — new)

| Condition | AA / BWT | Displacement share at head | CKA L0 / L6 / L11 / head |
|---|---|---|---|
| fast (AdamW 5e-5, all params) | 14.3 / −91.3 | 97.2–97.5 % (input 2 %, early 0.3 %, mid 0.2 %, late ≤ 0.1 %) | 0.895 / 0.88 / 0.42 / 0.39 |
| slow trunk (SGD 1e-4 / head 1e-2) | 28.4 / −77.5 | 100 % | 0.997 / 0.98 / 0.93 / 0.80 |

Head-dominant displacement on 3/3 seeds in both regimes; CKA monotone in depth (the
embedding-layer CKA is undefined for a CLS probe — constant across inputs — and is excluded).
**H1 supported**: the anatomy transfers to a CLS-head image classifier.

---

## 4. What the buffer must contain (DIL, LayoutLMv3; single-seed diagnostics unless stated)

| Store (frozen boundary k) | AA | BWT | Bytes | Note |
|---|---|---|---|---|
| Raw latent replay, 50 docs/task, k=4 | 87.3 | −2.2 | ≈163 MB | converged |
| **CoLaR** per-doc SVD r=128, k=4, 50 docs | 87.6 (3-seed 86.8) | −1.7 | 60 MB | lossless at 2.7× |
| CoLaR r=64 | 80.6 | −12.3 | 32 MB | |
| Latent replay, 5 docs/task, k=8 (3 seeds) | 66.5±3.0 | −30.5 | | small-buffer point |
| Spectral summary r=16 | 41.9 | | ≈0.4 MB | collapses |
| Per-class Gaussians (full dim) | 39.4 | | | collapses |
| Real k-means centroids | 36.7–39.7 | | | collapses (not a fidelity problem) |
| 4 whole docs vs same features on carriers (5 ep) | 63.8 vs 36.7 | | | **+27 AA from whole-document binding** |
| PLaR public-proxy replay (0 private bytes) | 58.9 | | | FUNSD 76 / SROIE ≈7: coverage-limited |
| CoLaR + class-balanced / coverage / entity-weighted (3-seed base 87.8) | 87.1 / 86.8 / 87.1 | | | conservation: FUNSD↔SROIE trade ≈1:1 |
| CA-CoLaR adaptive [18,5,2]/[64,128,64] vs uniform d10/r64 (3 seeds) | 75.2±3.9 vs 73.6±5.0 | | equal bytes | **null** (+0.6…+1.6 ± 8) |

---

## 5. Falsification ledger — buffer-free families on DIL/LayoutLMv3 (naive 41.3, replay ≈88)

| Family | Instance | Best AA | Why it fails (RCA) |
|---|---|---|---|
| Weight merging | TIES/Fisher head merge† | 21.6 (no-merge 21.9) | merged head vectors cancel |
| Merge + realignment | LCA (ICLR'26, ported) | 41.9 | backbone merge halves BWT; Gaussian realignment destroys acquisition |
| Parametric slots | LexSlot, no buffer | 42.2 | isolated slots cannot re-ground a shared head |
| Input-anchored memory | Ledger† | FUNSD 21 / SROIE 23 | own-vs-cumulative gap is training-free |
| Feature-Gaussian replay | LexMem v3b (positive control) | 66.0 | partial; dense-label tasks only |
| | LexMem v5 (+relational) | 63.2; mid-task 13.0 | marginal summary breaks whole-doc binding |
| Prompt pools | L2P / DualPrompt / CODA | 30–33 | head drift, routing cannot fix |
| LoRA | O-LoRA / CL-LoRA | 41 | parameter-space isolation leaves head un-exercised |

† reached at a frozen-backbone / 3-epoch operating point; kept as evidence, not load-bearing.

---

## 6. Image-classification scope test (ViT-B/16 IN-21k; pre-registered `docs/IMAGE_SCOPE_PREREG.md`)

Recipe = the document recipe (AdamW 5e-5, early stop, bs 16 + checkpointing) for the
classical six; SLCA arms use SLCA's optimiser (SGD, backbone 1e-4 / head 1e-2). Buffers: 200
(document-matched) and 2 000 (20/class, image convention).

### 6.1 AA (BWT)

| Method | Split CIFAR-100 (10×10) | Split ImageNet-R (10×20) |
|---|---|---|
| Naive | 11.7±1.1 (−95.6) | 9.3±0.3 (−87.1) |
| Joint (oracle) | **89.5±0.4** | **77.6±0.5** |
| EWC | 13.9±1.5 (−93.1) | 9.3±0.2 (−87.2) |
| LwF | 15.7±0.6 (−91.0) | 12.7±1.4 (−84.4) |
| ER @200 | 42.3±2.7 (−61.6) | 21.6±1.3 (−75.2) |
| ER @2000 | 76.2±1.2 (−23.8) | 54.8±0.7 (−37.9) |
| DER++ @200 | 43.6±4.0 (−60.6) | 20.2±2.5 (−77.2) |
| DER++ @2000 | 76.5±1.5 (−23.6) | 52.8±1.1 (−40.5) |
| Slow trunk only (SLCA w/o alignment) | 38.0±3.9 (−66.7) | 25.8±0.7 (−67.4) |
| Slow trunk, fixed 20 ep (published schedule) | 26.7±0.3 (−79.4) | — |
| **SLCA** (slow trunk + Gaussian head alignment, buffer-free) | **87.5±0.2 (−7.8)** | **65.0±0.3 (−13.5)** |
| Latent replay, real CLS bank, 500 img/task, k=8 | 12.0±7.2 (−72.3) | — |
| Gaussian-summary latent replay (AGLR), k=8 | 10.7±0.8 (−96.5) | — |
| Published (SLCA Tab. 1, bs 128, 20–50 ep): joint / SLCA / Seq-FT | 93.2 / 91.5 / 88.9 | 79.6 / 77.0 / 71.8 |

### 6.2 Pre-registered verdicts (`scripts/image_scope_verdict.py`)

| Hypothesis | Rule | Outcome |
|---|---|---|
| Gate | joint within 10 pp of published, naive < 25 | PASS both |
| H1 anatomy transfers | head-dominant displacement 3/3 seeds + monotone CKA | **SUPPORTED** |
| H2 remedy ordering does not transfer (SLCA ≈ joint ≈ ER) | within 3 pp of joint and of ER@2000 | **SUPPORTED on CIFAR-100** (−2.0 vs joint, +11.3 vs ER); **NOT on ImageNet-R** (−12.6 vs joint, +10.2 vs ER) → mixed, reported as measured |
| H2b slow trunk alone recovers ≥ 80 % of the gap | per-seed fraction | **NOT SUPPORTED** (34 %; final rows [0,7,10,14,32,19,50,66,87,97]: old tasks extinguished at the head) |
| H3 Gaussian summary ≈ real CLS bank | within 3 pp | **INCONCLUSIVE**: both arms at the naive floor (12.0 vs 10.7). Frozen-trunk latent replay at k=8 loses plasticity from task 5 (diagonal 99→70; CE 1.1–2.3 at T9) under this recipe; it is not a working replay on ViT at these settings, so the pair cannot discriminate |

### 6.3 Reading for the paper

1. **The anatomy is cross-domain.** On a CLS-head ViT, ≥97 % of Fisher-weighted displacement
   is at the head and CKA falls monotonically with depth — the same picture as four document
   backbones.
2. **The failure of parameter-space protection is cross-domain.** EWC and LwF are at the
   naive floor on both image benchmarks; the slow-trunk regime (trunk CKA ≥ 0.93) still
   extinguishes old classes at the head — the readout-marginal snap of the document RCA.
3. **What differs is the sufficient re-grounding signal.** On images, re-training the head on
   class-Gaussian summaries of CLS features recovers near-oracle accuracy in-distribution
   (CIFAR-100: 87.5 vs 89.5) and the best buffer-free result under domain shift (ImageNet-R:
   65.0 vs 77.6, +10 over 2 000-exemplar replay). On per-token document IE the identical
   mechanism (LexMem v5 / LCA) stalls at 63–42. The boundary condition is therefore not
   "replay vs buffer-free" but *whether a marginal summary of the head's input preserves what
   the head reads*: a single CLS vector per sample is class-clustered and low-rank per class;
   per-token document features are near-full-rank and label-bound to position (the
   consistency law of §4).
4. **Buffer size matters more on images.** 200 exemplars (0.4 % of CIFAR) gives 42; 2 000
   gives 76; even that is 11 pp below buffer-free SLCA — replay is not the ceiling on images.
5. Under domain shift (ImageNet-R) nothing buffer-free closes the gap (−12.6), consistent
   with a trunk that must also move; this is the image counterpart of the Mixed-scenario
   residual on documents.

Unresolved and disclosed: published Seq-FT under the slow recipe is 88.9; ours is 38.0, and
the fixed-20-epoch arm is lower still (26.7). Not a schedule effect; batch size (128 vs 16,
8× fewer head updates per epoch at lr 1e-2) is the remaining candidate and cannot be tested
on the 6 GB box.

---

## 7. Status of the experimental programme

| Block | Runs | State |
|---|---|---|
| Document generality grid | 68 / 72 | 4 LiLT cells need > 6 GB (rented GPU) |
| Document diagnostics, RCA, kill-tests, replay ladder, CoLaR, PLaR, CA-CoLaR | — | complete |
| Image grid (10 methods × 2 benchmarks × 3 seeds) | 60 / 60 | complete |
| Image calibration arm (`slca_noca_pub`) | 3 / 3 | complete |
| Image H1 pilot diagnostics (fast/slow × 3 seeds) | 6 / 6 | complete |
| Image H3 pair | 6 / 6 run | inconclusive (both arms at floor) |

**Are the image experiments done?** The pre-registered programme is complete. Two optional
follow-ups would strengthen the section, neither is required for the claims above: (a) a
bs-128 SLCA/Seq-FT replication on a ≥ 16 GB GPU to close the 38-vs-89 discrepancy; (b) an H3
re-design in which latent replay actually works on ViT (e.g. full-image ER through the frozen
trunk vs Gaussian summary at the *penultimate* layer), so the summary-vs-bank question is
answerable. The H4 bridge (RVL-CDIP document classification) remains optional.

## 8. Pointers

- Paper draft: `paper/main.tex` (ICLR 2027 format, 9-page main text; §4–§7 written).
- Prereg + amendments: `docs/IMAGE_SCOPE_PREREG.md`; verdicts: `docs/IMAGE_SCOPE_VERDICT.md`.
- Tables: `results/pivot_AA.csv`, `results/table_backbone_*.tex`; image cells via
  `scripts/image_scope_verdict.py`; pilot JSONs `results/pilot/*.json`
  (`doccl-pilot-analyze`, `scripts/build_backbone_figures.py` already know the ViT condition).
