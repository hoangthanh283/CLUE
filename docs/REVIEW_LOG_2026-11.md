# Review log (Reviewer role, append-only)

## R1 (2026-11-11): audit of the rewritten paper

Snapshot audited: `HEAD = e3ca7b9` (P1 consistency pass included; P1 already fixed the "ER, matched
memory" label, "level with", "within three points" and "between naive and CoLaR" wordings, so those
are not re-listed). Tables were regenerated at 07:18; my independent recomputation (own script, raw
`metrics.json`, `statistics.stdev` = ddof 1) matches every cell of the current `vals.tex`,
`tab_2x2.tex` and `tab_main_clean.tex` to 0.05. Body ends on page 7, 0 undefined references in
main and supp.

### Verdict: **BLOCK**

One CRITICAL is a code bug with a 3-line fix. The other is pending data: every headline cell is
still legacy protocol. Five HIGH items are text or label errors that one Paper round can fix.

### CRITICAL

**CR1. Generator fallback breaks hard rule 6 and will soon change headline numbers without
warning.** `scripts/rca/make_clean_tables.py:39-50` (`pick`) and
`scripts/rca/make_retention_fig.py:31-32` take the first source with 3 seeds. Failing that, they
take the *first non-empty* source, and the clean source is always listed first. So a clean cell
with only 1 or 2 seeds beats a legacy cell with 2 seeds, or with 1. The docstring says the
opposite ("falls back to the legacy run until all three seeds exist"). Concretely:
- the v2 queue's next runs `cil_cord_colar_sbr_noreplay_seed42`, `cil_cord_colar_seed42` and
  `cil_cord_cl_lora_seed42` will each replace a legacy 2×2 cell (20.4; 28.8/31.9; 18.7/18.6) with a
  one-seed clean value that has no dagger. Through `\nNoBank`, `\nColar` and `\nClLora` that value
  also lands in the intro, the teaser bars and §3.3. The 2×2 then mixes the two protocols.
- it is already live in the full-label column: CoLaR-SBR prints `88.8` (one clean seed, s42) with
  no dagger, beside a dagger AA from different runs in the same row.

Fix: in `pick`, return a clean source only when `len(vs)==3`. Otherwise return the first
*legacy* non-empty source. For a column with no legacy source (full-label), print per-seed values
with an explicit n, or "--". Apply the same rule in `matrices()`.

**CR2. C5/C4 not yet resolved in the data. All headline cells are legacy protocol.** The Setup
(main.tex:93) describes the clean protocol, but every AA cell in `tab_2x2`, `tab_main_clean`,
Table 1 and the abstract/intro/teaser macros carries † (early stopping on the test split). The
baselines are also fp16 with a 100-epoch cap (ER, DER++, LwF, CPFD, IS3, naive, joint) or a
30-epoch cap (CL-LoRA, CODA), against fp32 at 10 epochs for the SBR/Recall/LexSlot rows. Clean
seeds on disk: one (CoLaR-SBR s42). Fix: the queue, not prose. At observed wall times
(1.5-2.0 h per CIL-CORD run; joint 0.6 h), the CIL-CORD 2×2 cells and references in v2 order are
about 30 h of work and the whole v2 queue about 100 h. That fits before the Nov 17 07:00 freeze
only if nothing fails. Regenerate after each cell completes. Any column still † at the freeze must
be stated as such in the caption.

### HIGH

**H1. False claim: "every method sits at 16–24 AA" under legacy scoring** (main.tex:138). Legacy
grid `results/cil_cord_<m>_seed*/metrics.json`, LayoutLMv3:

| Method | Legacy CIL-CORD AA |
|---|---|
| naive | 19.0 |
| LwF | 18.8 |
| DER++ | 18.2 |
| ER | 16.0 |
| CL-LoRA | 15.2 |
| LCA | 8.2 |
| IS3 | 4.4 |
| CPFD (2 seeds) | 1.4 |
| EWC | **0.5** (also in supp `tab:grid`) |
| CODA-Prompt | **0.0** |

Fix: "naive, LwF, ER and DER++ sit at 16–19 against 33 for joint training, and several
baselines collapse below that".

**H2. The 2×2 isolation row is not a single-variable contrast, yet the text says it is.** The
claims at issue:
- main.tex:151: "the same isolation with a blocked readout reaches 92.8".
- `tab:twobytwo` caption: "per-session low-rank adapters ... CL-LoRA with a shared softmax, SBR-Recall with the blocked readout".
- teaser caption: "per-session isolation is on par with it ... but only when the readout is blocked".

What the two cells actually ran:

| | CL-LoRA cell (`doccl/methods/cl_lora.py`, `configs/method/cl_lora.yaml`) | SBR-Recall cell |
|---|---|---|
| Adapters | O-LoRA orthogonal subspaces, rank 8, Q/K/V only | rank-64 Q/K/V+FFN deltas |
| Inference | all task adapters applied together, no per-session routing | one delta per block |
| Extra loss | LwF-style KD (kd_alpha 1.0) | none |
| Base | not frozen | frozen after session 1 |
| Precision, epoch cap | fp16, 30 | fp32, 10 |

Same sentence, separate literal error: "each session reaches 93 F1 when it is learned". The
CL-LoRA diagonals are s42 94.6/86.5/87.8/86.3/93.3 and s7 92.6/89.5/80.9/84.4/92.9.

Fix:
- either run Recall's own adapters with a single shared softmax block as the true control,
- or relabel the row "low-rank adapters, no memory", drop "the same isolation", and soften the teaser claim;
- and replace "93" with "81 to 95".

**H3. Unscoped universals that remain** (the C3 pattern):
- Abstract: background conflict is "an effect that no memory can repair". The evidence is one
  seed of oracle label refresh (61.2, `gates/.../cil_cord_colar_r_oracle_seed42`) and two fp16
  seeds of loss restriction (42.0). Scope it: "that no memory we tried repairs".
- main.tex:65: "any method that keeps updating a shared network without old-task supervision
  loses the old readout". Tested: EWC, LwF and five families, on DIL only. Scope it as main.tex:132
  already does.
- main.tex:67: "Each remedy is necessary and neither is sufficient". Row 3 of the paper's own 2×2
  contradicts this: isolation replaces the memory and reaches 92.8 with no memory. Fix: "each
  cause needs its own remedy; neither remedy is sufficient alone".

**H4. The full-label F1 companion metric (I2) is promised but essentially absent.** Contribution 1
("we pair it with a mask-free full-label F1") and the Scoring paragraph say it is reported. In
`tab:main` the column is "--" everywhere except CoLaR-SBR = 88.8, which is one clean seed from a
different protocol than its row's AA. Joint and naive, the two numbers that carry the "scoring
artefact" claim, have no full-label value. Fix: fill the column from the clean runs (naive, joint,
CoLaR, CoLaR-SBR, Recall) under the CR1 rule. If they are not ready by the freeze, change
contribution 1 to "we add".

**H5. Wrong collapse footnote, and an inconsistent zero-drop rule.** In `tab_main_clean`, IS3
CIL-CORD carries `c` ("one seed collapsed ... and is excluded"), but nothing was excluded: the
values are 5.60 and 7.67, and no s123 run exists. Cause: `make_clean_tables.py:176`
(`mark = "c" if dz else ""`) marks every drop-zero row whether or not a value was dropped. CPFD,
by contrast, keeps its zeros: CIL s7 0.00, Mixed s42 0.27, and DIL s42 0.00, which is fp32. That
contradicts the "fp16-only collapse" framing for CPFD. Fix: set `c` only when a value was actually
removed, and state the CPFD treatment (kept, since it collapses in fp32 too).

### MEDIUM

- **M1. The dagger hides three different legacy setups.** The `tab:main` caption says only
  "evaluation-split early stopping; fp16 for baselines". In fact:
  - the SBR, no-memory SBR, Recall, LexSlot and DIL-SBR cells are fp32 with a 10-epoch cap (gate runs);
  - the naive, joint, ER, DER++, LwF, CPFD and IS3 cells are fp16 with a 100-epoch cap, and CODA and CL-LoRA fp16 with a 30-epoch cap;
  - the DIL baselines come from the older grid.

  The 2×2 therefore compares fp16 CoLaR (28.8/31.9) with fp32 CoLaR-SBR. An fp32 CoLaR s42
  reading exists (34.06, `gates/cil_protocol_2026-11/cil_cord_colar_seed42_k4_d50_r128`). "SBR-Recall
  on par with joint" compares fp32 Recall with fp16 joint (fp32 joint s42 = 93.05). The C4 budget
  mismatch is still undisclosed in main. Fix: one caption sentence per dagger group (or the fp32
  cells). This disappears once CR2 lands.
- **M2. The cosine is attributed to the wrong methods.** "They fail in one way ... (cosine
  0.997–0.998 against 0.60)" (main.tex:132) is attributed to the five families. The source
  (`STATE.md:1056`, `results/rca/c_marginal_snap.csv`) measured naive and LwF. Fix: attribute
  correctly, or cite a measurement on the five families.
- **M3. The rank-8 Recall ablation quotes the run with memory.** "rank 8 ... gives 80.5" is
  `cil_cord_colar_sbr_recall_seed42` (docs_per_task 50). The memory-free rank-8 run, which matches
  the headline Recall setting, is 82.5 (`..._recall_d0_seed42`; diagonals 64.9/67.7 on sessions 3
  and 4). Fix: quote 82.5, or state the memory.
- **M4. Baseline rows never read clean runs.** `BASE` rows use `src(None, …)`, so the clean CL-LoRA
  and LwF runs the v2 queue produces will reach the 2×2 (`C['cllora']`) but not the `tab:main`
  CL-LoRA or LwF rows. The two tables would then disagree on the same cell. Fix: give the BASE
  rows clean names `cil_cord_<key>`.
- **M5. The supplement setup paragraph is stale** (supp.tex:55). It still says "batch size 8 ...
  bf16 autocast ... early stopping on task val F1, cap 100 (never binding)". That contradicts main
  Setup (batch 2; † = early stopping on the evaluation split; fp16 for the re-scored baselines) and
  the senior-review correction. Fix: rewrite it per dagger group.
- **M6. The modality-mask BWT triple cannot be reproduced as aggregated** ("BWT −81, −88, −79",
  main.tex:110). From `results/pilot/*.json`:
  - full: −89.9/−90.0/−89.5 → −89.8;
  - text+layout: −86.5/−86.8/−88.2 → −87.2;
  - text-only: −79.4/−78.7 plus s123, a dead run (matrix all 0.0) → −79.1 over two seeds.

  `STATE.md:261` already flags −81.3 as an unreconciled aggregation that included the `_ord210`
  runs. The dead text-only s123 is also inside the n=3 of supp `tab:depthprofile` ("text only,
  9505×"). Fix: report −89.8/−87.2/−79.1 (two seeds) and drop s123 from the depth table, or say so.
- **M7. The calibration bound is one fp16 run.** "changes AA by at most 0.6" comes from the
  fp16 rescore of SBR s42 (85.2 → 85.8, `STATE.md:22`). Add "(seed 42)".

### LOW

- L1. Table 1 (`tab:dil`) is typed by hand, which breaks rule 6. Three cells disagree with the
  files by more than 0.05: BROS LwF 35.2 (file 35.15), BROS DER++ 87.6 (87.55), BERT ER s.d. 1.0
  (0.95). Generate it instead.
- L2. supp.tex:112, CPFD on DIL: "reported as 6.2±7.2" should be 6.1±7.3 (0.00/4.25/14.19). "Final
  row zero on every domain for seeds 42 and 7" is false for s7 (final row 3.0/9.7/0.0).
- L3. main.tex:102: the collapse numbers 89.1→3.3, 83.9→2.1 and 88.9→6.6 trace to the pilot stream
  FUNSD→CORD→SROIE (`results/pilot/{c4_full,cl_lilt,cr_bros}`). The next sentence points to Table 1
  (DIL, FUNSD→SROIE→CORD), where the same quantity is 89.0→4.0, 80.2→3.2 and 88.5→8.6. Name the
  stream.
- L4. main.tex:95, "so an error is charged once". A false positive on a receipt that has no
  entity of its class's session is never charged, because that receipt is absent from that
  session's test split. Say "at most once".
- L5. The LexSlot row stores raw token ids (`store_input_ids: true`,
  `configs/method/colar_sbr_slot.yaml`). Disclose it in the Ablations paragraph. (CoLaR and SBR do
  not store them; checked.)
- L6. `macro()` writes `\n<Name>{0}` for empty cells (`\nSbrBros`, `\nRecallFull`, `\nJointFull`).
  None is used now, but any later use would print 0. Write `--` instead.
- L7. Limitations, "trail raw replay by 5–9 points on BERT and LiLT". The means are 5.3 and 8.8,
  but per seed BERT s42 trails by only 0.3 (76.4 vs 76.7). Say "on average".
- L8. supp `tab:ledger`: the CoLaR-R row (oracle 26.3/64.4) uses legacy scoring but is not marked,
  while main cites the restricted-scoring oracle 61.2.
- L9. The full-label hook swallows every exception (train.py:413). A failed evaluation leaves only
  a log warning and a "--" cell. That is acceptable, but grep each clean log for "full-label eval
  skipped".

### Part A: status of the senior review items

| Item | Status | Evidence / what is missing |
|---|---|---|
| C1 (p<0.01 on Fisher importance) | RESOLVED | main:110 drops the test; "weighting favours the head by construction"; supp:197 reports 3/12 rejections on importance and disowns it. |
| C1b (no-power generality test) | PARTLY | supp:197 calls the permutation test uninformative; `tab:depthprofile` reports the head-excluded profile (I reproduced it with `pilot_depth_profile.py`; it matches). No effect sizes. The "same profile on all four backbones" claim is visual only. The text-only row includes a dead seed (M6). |
| C2 (replay in the consolidation probe) | RESOLVED | Removed from main; supp:199 discloses the 200-document buffer and ER 87.9. |
| C3 (universal claims) | PARTLY | Scoped at main:132 and in contribution 2; still unscoped in the abstract ("no memory can repair"), main:65 ("any method") and main:67 ("each remedy is necessary") (H3). Recall on DIL is not run; argued only (main:195). |
| C4 (confounded Table 3) | PARTLY | fp16 disclosed, collapsed zeros footnoted (the IS3 footnote is wrong, H5), ER memory size now in the caption, n<3 printed per seed. Epoch-cap mismatch undisclosed, three legacy setups behind one dagger (M1). Fix pending in queue (CR2). |
| C5 (test-split model selection) | NOT RESOLVED (data) | Setup describes the clean protocol and the dagger correctly. Every headline cell is still legacy (CR2). supp:55 still says "val F1" (M5). |
| C6 (controls, prior art) | PARTLY | (iii) RESOLVED: Kim et al. 2022, EASE, InfLoRA, ICE and SpanKL cited at main:81 (keys present in the .bbl). (i) ER+SBR not run. (ii) No isolation baseline with task inference; the CL-LoRA cell is not one (H2). |
| I1 ("at the oracle") | PARTLY | "Oracle" and "bound" are gone from main and supp; Limitations: "a reference, not a bound". Joint not re-tuned. |
| I2 (restricted-scoring argument + full-label F1) | PARTLY | Argument given at main:95 (needs "at most once", L4). Full-label column empty except one seed (H4). |
| I3 (two Independent numbers) | RESOLVED | The Independent column is gone from main; one value, 88.8, in supp `tab:grid`. |
| I4 (SBR on DIL below CoLaR) | RESOLVED | main:206 states 86.2 vs 87.8 and its only two differences. |
| I5 (Recall cost and scope) | PARTLY | Costs: 54 MB (checked: 7.08M params × 4 × 2 B), about 3.7× (44/12 layer passes); order dependence, Mixed n=1 and the fixed-label limit are stated. Order permutation and Mixed seeds 7/123 not yet run (both in queue). |
| I6 (privacy, storage) | RESOLVED | "no raw pixels or token ids ... no formal privacy claim"; 2.7× is stated against the latent memory. CoLaR/SBR do not store ids (checked); LexSlot does (L5). |
| I7 (framing, method figure, loss) | RESOLVED | New title; teaser shows the 2×2 and an SBR schematic; eq. (1) to (3) give the loss and inference. |
| I8 (sample size, CIs) | PARTLY | Test-set sizes in Limitations; no confidence intervals. |
| Minor: legacy columns | RESOLVED | Moved to supp and labelled. |
| Minor: in-house names | RESOLVED | LexSlot defined inline; HGT, CUBER and LexMem only in the supp ledger. |
| Minor: "registered" | RESOLVED in main | main:132 "pass bar fixed before the run"; supp:105/167 "pre-registered" (acceptable). |
| Minor: "exact retention" | RESOLVED | "every curve is flat". |
| Minor: n=2 mean±s.d. | RESOLVED | Per-seed values printed. |
| Minor: code availability | PARTLY | supp:103 "will ship with the code"; no statement in main. |

### Part B: numbers

**Mismatches with the generated files (>0.05):** none at HEAD. (At 2f0a5e9, DER++ CIL-CORD showed
21.3/18.8; s123 = 21.07 had landed at 03:10, and the 07:18 regeneration fixed it to 20.4±1.4.)
Recomputed cells, per seed (42/7/123):

| Cell | Per seed | Mean (± s.d.) |
|---|---|---|
| CIL naive | 18.97/19.08/18.60 | 18.9±0.3 |
| CIL SBR (gate fp32) | 87.62/86.27/90.15 | 88.0±2.0 |
| CIL Recall (gate fp32) | 92.69/93.37/92.44 | 92.8±0.5 |
| CIL joint (fp16) | 92.84/90.48/92.43 | 91.9±1.3 |
| CIL ER (200 documents) | 26.91/22.05/24.01 | 24.3±2.4 |
| CIL LexSlot | 90.15/89.72/89.23 | 89.7±0.5 |
| CIL DER++ | 21.27/18.76/21.07 | 20.4±1.4 |
| CIL CoLaR | 28.83/31.87 | 2 seeds |
| CIL no-memory SBR | 20.36 | 1 seed |
| CIL CL-LoRA | 18.67/18.59 | 2 seeds |
| Mixed SBR (fp16) | 89.77/89.62 | 2 seeds |
| Mixed joint | 86.94/88.46 | 2 seeds |
| Mixed CoLaR | 75.56/74.55 | 2 seeds |
| Mixed ER | 75.58/73.33 | 2 seeds |
| Mixed LexSlot | 88.48/89.99/90.32 | 89.6±1.0 |
| Mixed Recall + memory | 85.94 | 1 seed |
| DIL naive | 40.06/43.03/40.85 | 41.3±1.5 |
| DIL CoLaR | 87.58/88.30/87.38 | 87.8±0.5 |
| DIL SBR (v2 rule) | 86.76/84.77/87.08 | 86.2±1.3 |
| DIL joint | 89.68/89.03/87.49 | 88.7±1.1 |
| DIL ER | 86.21/88.73/88.62 | 87.9±1.4 |
| DIL LexSlot | 87.24/87.25/86.62 | 87.0±0.4 |

All baseline rows also match: LwF and IS3 after the zero drop, CPFD, CODA-Prompt, LCA, CL-LoRA.
Registry mapping checked:
- no `_bert`/`_bros`/`_lilt` leak (DIL CoLaR uses the exact suffix `_k4_d50_r128`; legacy `""` suffixes are exact);
- the clean `*` globs cannot reach the BROS directories (`…_sbr_bros_seed*_bros`) or the order runs (`cil_cord_o2_*`);
- the Mixed SBR cell is v2-rule code (the rescore ran 11-06, after commit 608461c); an fp32 v2 s42 reading of 89.73 exists in the gate directory and is unused.

Mapping issues are the ones listed above: ER is 200 documents (now captioned), CL-LoRA is not a
matched isolation control (H2), and fp16 and fp32 cells are mixed under one dagger (M1).

**Prose literals (main.tex), traced:**

| Literal | Location | Source | Status |
|---|---|---|---|
| 33 (legacy joint) | 65, 95, 138 | legacy grid 33.43/33.57/32.94 = 33.3 | OK |
| 16–24 | 138 | legacy grid | **false (H1)** |
| 89.1→3.3, 83.9→2.1, 88.9→6.6 | 102 | `results/pilot` | OK; wrong stream implied (L3) |
| CKA 1.00/0.78/0.34/0.25/0.16 | 110 | ledger F1 | traced, not recomputed |
| BWT −81/−88/−79 | 110 | pilot JSONs | **not reproducible as aggregated (M6)** |
| "two to four orders of magnitude" | 110 | `tab:depthprofile` (111× to 9505×) | OK |
| 41.1/0.0, 38.8/−76.8, 0.19 | 112 | ledger F2 (one seed) | traced |
| 21.6/21.9, 43.9, 39.8, 42.2, 66.0, 88 | 132 | supp ledger / ledger F5 | OK |
| cosine 0.997–0.998 / 0.60 | 132 | `STATE.md:1056` (naive and LwF) | **misattributed (M2)** |
| ≤−0.1 AA, mass about 10⁻⁶ | 132 | ledger F5a | traced |
| 2.5, 17, −11.9/−2.6/+3.5/+5.2 | 134 | recomputed from supp `tab:grid` | OK |
| 61.2 (seed 42) | 140 | gate oracle 61.17 | OK |
| 42.0 (two seeds) | 140 | rescore sliced 44.17/39.85 (fp16) | OK |
| "about half" | 140 | 0.46–0.51 depending on which seed-42 runs are used | OK |
| 93 (CL-LoRA per session) | 151 | diagonals 80.9–94.6 | **false (H2)** |
| 87.3, 41.9, 39.4, 36.7–39.7, 63.8, 36.7, 27 | 165 | ledger F6/F6a | traced |
| ViT 87.5/65.0, 80.0/55.1, 88.3/72.3 | 165 | ledger F9/F9a | traced |
| 60 MB, 2.7×, 163 MB, 80.6 at 32 MB | 167 | ledger F6b | traced |
| 44.7, 85.9 (Mixed Recall) | 195 | gate r64b 44.66, r64c 85.94 | OK (r64 = 62.07 is the bug run, ledger F17b) |
| 14.8 | 212 | gate `sbr_frozen` 14.77 | OK |
| ≤0.6 (calibration) | 212 | `STATE.md:22`, one fp16 seed | M7 |
| 80.5 (rank 8) | 212 | run with 50-document memory | **M3** |
| "near 65" | 212 | 61.5/68.0 diagonals | OK |
| 96 MB | 214 | `replay_memory_bytes` 100,672,250 B = 96.0 MiB | OK |
| 54 MB | 214 | 7.08M params × 4 × 2 B = 54.0 MiB | OK |
| 3.7× | 214 | (4 + 8×5)/12 = 3.67 | OK |
| 5–9 | 218 | DIL CoLaR vs ER: BERT 5.3, LiLT 8.8, BROS 0.5 | OK on means (L7) |
| 50–347, 149/50, 800/100/100, 626/347 | Setup | dataset sizes | OK |

**Comparative sentences, checked per seed:**

| Sentence | Per-seed check | Verdict |
|---|---|---|
| Recall "on par with joint" | −0.15/+2.89/+0.01 (fp32 vs fp16) | OK |
| CoLaR-SBR "within a few points of joint" | −5.2/−4.2/−2.3 | OK; "2 to 5 points" would be more precise |
| CoLaR-SBR "reaches" joint on Mixed | +2.8/+1.2 | OK |
| CoLaR "matches raw-exemplar replay" on DIL | +1.4/−0.4/−1.2 (but 150 vs 200 documents) | OK |
| "at or below CoLaR (30.4)" | every baseline seed ≤ 26.9 | OK |
| Table 1 "within one point of joint on average" | means | OK |
| LexSlot "within two points ... narrows the spread" | means and s.d. | OK |
| Retention figure | per-seed shapes all 5×5 | OK |

**Clean-protocol runs** (`results/clean_2026-11/`):

| Run | Settings confirmed in log + `metrics.json` | AA | BWT | Full-label F1 |
|---|---|---|---|---|
| `cil_cord_smoke_naive_e1_seed42` | amp false, val_frac 0.1, epochs 1 (smoke) | 18.05 | −76.09 | 12.55 |
| `cil_cord_colar_sbr_seed42` | amp false, fp16 false, val_frac 0.1, epochs 10 (stopped at 5/5/5/6/6), docs 50, k 4, r 128 | 87.76 | −2.41 | 88.83 |

For the SBR s42 run, the batches per epoch (360/172/247/326/351) equal 90% of the session pools
(800/382/549/723/780) at batch 2. That confirms the held-out split comes from training data. Its
legacy counterpart was 87.62, so held-out early stopping moves AA by +0.14 on this seed.
`cil_cord_colar_sbr_seed7` is still running (no `metrics.json`).

### Part C: code review (`git diff f82c3d1^..HEAD -- scripts/ doccl/ tests/ configs/`)

- Full-label hook (`train.py:391-416`, `scenarios.py`): **correct.**
  - It runs after the session loop, outside `restrict_preds`, so `model._allowed` is None and no restriction is active.
  - The test set is remapped with `snapshots[-1]`, the final head map, so it is right for the permuted orders o2/o3 too.
  - `ColarSBR.evaluate` does not use `tid` for decisions. It overwrites `eval_dump[last]`, but `eval_dump.pt` is saved before the hook runs, so nothing is lost.
  - It is wired for `cil_cord*` only; on Mixed and DIL it is a no-op, as documented. The joint path is covered as well.
  - Remaining risk: exceptions are swallowed (L9).
- `split_val`: **no test leakage.** The validation set is drawn only from the session's training
  set, through `random_split` with `Generator(seed + task_idx)`, so the split is identical across
  methods for a given scenario and seed. Joint uses `seed` on the pooled set, a different but
  legitimate split. Early stopping runs under the same session restriction as evaluation.
- Generators: CR1 (fallback order), H5 (unconditional `c` mark), M4 (BASE rows without clean
  sources), L6 (empty macros = 0). Retention figure: inherits CR1; matrix shapes are uniform
  (5×5), so the averaging is valid.
- `pilot_depth_profile.py`: reproduces the supp table exactly (n=3 per condition). The text-only
  condition includes the dead s123 run (M6).
- Queue drivers v1/v2: the resume check `ls $P/${NAME}*/metrics.json` is prefix-safe for the
  names in use. The BROS run directories carry a `_bros` suffix and cannot collide with the
  LayoutLMv3 globs. The v2 CL-LoRA clean cell uses the config defaults, which inherits H2: it is
  still not a matched control for SBR-Recall.

## R1b (2026-11-11): bibliography audit

Scope: all 52 keys cited in `paper_cvpr/main.tex`, `supp.tex` and `fig/*.tex` at HEAD 94c8068.
`refs.bib` has not changed since baff53b. Keys that are in `refs.bib` but never cited (e.g.
`le2025wavecre`, `zhang2025clcrossvqa`, `ge2025dmole`) are not in the built list and were not
audited.

Method: for each record I checked title, author list and order, venue, year and pages against the
following sources (in this order of preference):
- **DOI record:** the publisher-deposited Crossref metadata, at `https://doi.org/<DOI>` (38 entries);
- **arXiv:** the arXiv API record (author order, plus the venue note in the comments field);
- **PMLR:** the proceedings page (`zhai2023investigating`);
- **ICLR/OpenReview pages:** for `lca2026`, `ramasesh2021anatomy` and `yoon2018den`.

LoRA (ICLR 2022), OGD (AISTATS 2020) and Learning without Forgetting (LwF, ECCV 2016) have their
venue confirmed by arXiv and Google Research pages, not by opening the proceedings page.

**Verdict for the bibliography: no non-existent reference (no CRITICAL).** One entry is WRONG
(venue), four need MINOR corrections, and 47 are OK.

### Entries that need a change

| Key | Status | What is wrong | Correct record | Source |
|---|---|---|---|---|
| `gencner` | **WRONG (venue)** | Cited as an arXiv preprint, but it is published (arXiv comment: "Accepted by IJCNN 2025") | Yang, Yawen and Ma, Fukun and Meng, Shiao and Liu, Aiwei and Wen, Lijie. "GenCNER: A Generative Framework for Continual Named Entity Recognition". 2025 International Joint Conference on Neural Networks (IJCNN), 2025, pp. 1–9, doi:10.1109/IJCNN64981.2025.11227587 (arXiv:2510.11444) | https://doi.org/10.1109/IJCNN64981.2025.11227587 ; https://arxiv.org/abs/2510.11444 |
| `lca2026` | MINOR (author name) | Second author is given as "Vargas, Danilo" | Tran, Tung and Vasconcellos Vargas, Danilo and Than, Khoat. "LCA: Local Classifier Alignment for Continual Learning". ICLR 2026 (arXiv:2603.09888) | https://arxiv.org/abs/2603.09888 ; https://iclr.cc/virtual/2026/poster/10011599 |
| `kumar2023protoner` | MINOR (year, series) | The publisher record dates the volume 2024 and places it in a series; the bib entry has 2023 and no series | Kumar, Ritesh and Goyal, Saurabh and Verma, Ashish and Isahagian, Vatche. "ProtoNER: Few Shot Incremental Learning for Named Entity Recognition Using Prototypical Networks". Business Process Management Workshops (BPM 2023), Lecture Notes in Business Information Processing, Springer, 2024, pp. 70–82, doi:10.1007/978-3-031-50974-2_6 | https://doi.org/10.1007/978-3-031-50974-2_6 ; https://arxiv.org/abs/2310.02372 |
| `gdibench` | MINOR (authors truncated) | The first 10 of 16 authors are correct and in order, then "and others". No venue found: it is still arXiv only | Li, Siqi; Shen, Yufan; Chen, Xiangnan; Chen, Jiayi; Ju, Hengwei; Duan, Haodong; Mao, Song; Zhou, Hongbin; Zhang, Bo; Fu, Bin; Cai, Pinlong; Wen, Licheng; Shi, Botian; Liu, Yong; Cai, Xinyu; Qiao, Yu. arXiv:2505.00063, 2025 | https://arxiv.org/abs/2505.00063 |
| `zhang2023slca` | MINOR (venue string) | booktitle is just "ICCV" | Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19091–19101, doi:10.1109/ICCV51070.2023.01754 | https://doi.org/10.1109/ICCV51070.2023.01754 |

Optional page and volume completions, none of which is wrong as written:
- `wang2024survey`: TPAMI 46(8):5362–5383, doi:10.1109/TPAMI.2024.3367329;
- `hong2022bros`: AAAI 36(10):10767–10775;
- `huang2022layoutlmv3`: 4083–4091;
- `he2025cllora`: 30534–30544;
- `liang2024inflora`: 23638–23647;
- `bian2024cflat`: NeurIPS 37, 7608–7630.

### Entries verified OK (47)

| Group | Keys |
|---|---|
| Regularisation and replay | aljundi2018mas, buzzega2020der, kirkpatrick2017ewc, li2017lwf (ECCV 2016), lopezpaz2017gem, rebuffi2017icarl, rolnick2019er, zenke2017si |
| Optimisation geometry | farajtabar2020ogd, saha2021gpm, bian2024cflat |
| Pre-trained-model, prompt and adapter methods | wang2022l2p, wang2022dualprompt, smith2023coda, wang2023olora, he2025cllora, mcdonnell2023ranpac, zhou2024ease, liang2024inflora |
| Architectural isolation | mallya2018packnet, yoon2018den |
| Surveys and theory | masana2022class (TPAMI 2023, 45(5)), wang2024survey, kim2022theoretical |
| Exemplar-free methods | zhu2021pass, petit2023fetril, goswami2023fecam |
| Latent and generative replay | hayes2020remind, pellegrini2020latent, vandeven2020brain (Nat. Commun. 11:4069) |
| Forgetting analysis and tools | ramasesh2021anatomy, zhai2023investigating (CPAL, PMLR 234:202–227), kornblith2019similarity, mccloskey1989catastrophic, hu2022lora |
| Document encoders | huang2022layoutlmv3, wang2022lilt, hong2022bros |
| Continual NER | zhang2023cpfd, zheng2022cfner (EMNLP 2022, 3602–3615), zhang2023spankl (AAAI 37(11):13993–14001), qiu2024is3 (Findings ACL 2024, 777–791), liu2023ice (Findings ACL 2023, 2241–2257) |
| Continual relation extraction | han2020emar, cui2021rpcre, zhao2022crl |
| Document continual learning | minouei2022table (Applied Sciences 12(18):8969, doi:10.3390/app12188969) |

### Sentence checks: does the cited paper say what the sentence attributes to it?

| # | Sentence (main.tex Related Work) | Verdict | Reason (source) |
|---|---|---|---|
| 1 | Current wording: "ICE trains per-session classifiers on a frozen encoder ... neither replays stored examples" | SUPPORTED | ICE stands for "Individual Classifiers with Frozen Feature Extractor": it trains one classifier per session, in the setting where no old-class samples are stored (ACL Anthology 2023.findings-acl.141). |
| 1b | Earlier wording: "neither trains the trunk" | NOT SUPPORTED for SpanKL | SpanKL fine-tunes a shared bert-large-cased contextual encoder with AdamW (arXiv:2302.12200, Sec. on implementation). The current wording, "per-type span heads with distillation; neither replays stored examples", is SUPPORTED: SpanKL uses a span layer per task, per-type sigmoid binary cross-entropy, and Bernoulli-KL distillation, with no rehearsal. |
| 2 | "Document-level CL is nascent: ... ProtoNER addresses few-shot incremental NER" | SUPPORTED, but undersold | ProtoNER does key-value/NER extraction on visually rich documents with LayoutLMv2/v3/LiLT-type models and adds classes from about 30 samples, using prototypical networks and a hybrid loss, without retaining the original training data (arXiv:2310.02372). It is the closest prior work to this paper's setting (class-incremental KIE on LayoutLM-family encoders). Suggest: "ProtoNER adds new KIE classes to a LayoutLM-family model from few labelled documents with prototypical networks". |
| 3 | "REMIND quantises later-layer features, Latent Replay banks activations at a chosen layer ... **Each stores one summary per token or class** and targets image labels" | **NOT SUPPORTED, HIGH** | REMIND stores, *per training image*, product-quantised codes of a *mid-level* CNN feature map (an m×m×s integer array) and replays them through the plastic upper layers (arXiv:1910.02509, Sec. 3). Latent Replay stores per-pattern "activation volumes" at an intermediate layer (arXiv:1912.01100). Neither keeps one summary per class or token: both keep one compressed or raw entry per example. That is CoLaR's design too, so the stated distinction from the closest prior work is false. What does differ: a document has one label per token bound to a position, and per-document truncated SVD replaces product quantisation. Fix the sentence to say that, and write "mid-level", not "later-layer". |
| 4 | "generative replay samples a per-class model" | NOT SUPPORTED | Brain-inspired replay uses a *single* conditional VAE whose latent prior is a Gaussian mixture with one mode per class, and it replays hidden-level representations (Nat. Commun. 11:4069, PMC7426273). Fix: "samples hidden representations from one class-conditional generator". |
| 5 | "Zhai et al. report the same [forgetting at the layers nearest the output] for multimodal LLMs" | NOT SUPPORTED | Zhai et al. evaluate forgetting with EMT, treating the multimodal LLM as an image classifier, and find degradation and hallucination as fine-tuning proceeds, "even when the image encoder remains frozen". There is no layer-wise localisation (arXiv:2309.10313; PMLR 234). Fix: "and Zhai et al. find forgetting in multimodal-LLM fine-tuning even with the image encoder frozen", and drop "the same". |
| 6 | "Kim et al. decompose class-incremental learning into within-task prediction and task-identity prediction, and tie the latter to OOD detection" | SUPPORTED | Abstract: "decomposed into ... Within-task Prediction (WP) and Task-id Prediction (TP)"; "TP is correlated with OOD detection" (NeurIPS 2022 proceedings). |
| 7 | "recent adapter methods train one module per task and infer which to use at test time (EASE, InfLoRA)" | **NOT SUPPORTED** | EASE trains one adapter per task but concatenates all of them for "joint decision-making across multiple subspaces": there is no task inference (arXiv:2403.12030). InfLoRA reparameterises the pre-trained weights within an interference-free subspace, so a single model runs at test time with no module selection (arXiv:2404.00228). Fix: drop "and infer which to use", or cite methods that do infer task identity. HiDe-Prompt (NeurIPS 2023) and MoE-Adapters (CVPR 2024) are candidates; both are named from memory and must be verified before adding. |
| 8 | "GDI-Bench freezes layers heuristically" | NOT SUPPORTED as worded | GDI-Model's Layer-wise Adaptive Freeze-Tuning (LW-AFT) freezes about 99% of individual *parameters*, not layers. Which ones stay trainable is set by update magnitude after a 10% expert fine-tune, with a per-layer budget (arXiv:2505.00063v2, Sec. 4). It is also a supervised fine-tuning study of a multimodal LLM, not a continual-learning method. Fix: "GDI-Bench mitigates forgetting in multimodal-LLM fine-tuning by freezing most parameters, chosen by their update magnitude". |
| 9 | "continual table detection uses plain ER" | SUPPORTED | Abstract: "We employed an effective technique called experience replay" (doi:10.3390/app12188969). |
| 10 | "Ramasesh et al. localise forgetting in image classifiers to the layers nearest the output" | SUPPORTED | "deeper layers are disproportionately the source of forgetting"; "higher layers refer to layers closest to the output" (arXiv:2007.07400). |
| 11 | "LCA: merge models and realign the classifier" | SUPPORTED | It merges parameter-efficient fine-tuning (PEFT) parameters, then realigns the classifier on samples drawn from per-class Gaussians (arXiv:2603.09888). |
| 12 | "CPFD, CFNER: distillation and pseudo-labels" | SUPPORTED for CPFD; UNCLEAR for CFNER | CPFD uses pooled-feature distillation and confidence-based pseudo-labelling. CFNER's abstract describes causal-effect distillation from Other-class tokens plus curriculum learning for label noise; pseudo-labels are not named in the abstract (arXiv:2210.03980). |
| 13 | "GenCNER: span generation" | SUPPORTED loosely | It casts continual NER as "entity triplet sequence generation" with pseudo-labelling and KD. "Entity-triplet generation" would be more exact. |
| 14 | "EMAR, RP-CRE, CRL: episodic memory with prototypes ... all sentence-level and text-only" | SUPPORTED | Memory replay with prototypes or memory-based nearest-class-mean (NCM) classification. All are relation-extraction work on sentences. |

## R2 (2026-11-11): confirmation audit

Snapshot: `HEAD = 348b00f`, `main.pdf` built 07:55 (after `main.tex` 07:55:09), `supp.pdf` 07:44,
generated tables 07:33. Recomputation: own script, raw `metrics.json`, `statistics.stdev`
(ddof 1). No model, dataset, torch or pytest touched.

### Verdict: **BLOCK** (one HIGH, text only, one Paper round)

CR2 (legacy headline cells) is waived for draft uploads and the abstract clause "an effect that
no memory can repair" is with the user; neither is counted. What blocks is **N1**: the 2×2's
third row, its caption, §4.3 and the teaser describe the CL-LoRA cell as per-session adapters
on a trained base. The code runs **one** LoRA adapter shared by all sessions, on a **frozen**
pre-trained encoder. (R1's H2 table had the same error, "all task adapters applied together;
base not frozen", and the Paper agent copied it. I correct it here.) Every R1 CRITICAL and HIGH
item is otherwise fixed, and all numbers match.

### Part A: status of the R1 and R1b items

| Item | Status | Evidence (current text / code) |
|---|---|---|
| CR1 | CONFIRMED FIXED | `choose()` (make_clean_tables.py:44-54) returns a clean source only when `len(vs) == 3`, else the first non-empty *legacy* source; `pick()` and `matrices()` (make_retention_fig.py:22-34) both call it. Full-label: `pick_clean_only` + `fmt(show_n=True)` prints per-seed values with `^{n=k}` when n<3, mean±s.d. only at n=3. Residual: L-j (unused `\v*Full` macros). |
| CR2 | WAIVED (data) | Every AA cell still †; 2 of 3 clean CoLaR-SBR seeds on disk, s123 running. |
| H1 | CONFIRMED FIXED | main:128 "naive training, LwF, ER and DER++ sit at 16--19 AA against 33 for joint training, and several baselines collapse below that band". Per seed ER s7 = 14.4 (L-e). |
| H2 | PARTLY; see **N1** | "81 to 95" fixed (diagonals 80.9 to 94.6); row relabelled; teaser softened ("a different adapter method"); main:141 adds "rather than isolating the readout alone". But the new caption text is factually wrong about the CL-LoRA cell (N1). |
| H3 (body) | CONFIRMED FIXED | main:65 "on a fixed label space every method we tested that keeps updating a shared network without old-task supervision loses the old readout"; main:67 "Neither remedy is sufficient alone and each cause needs its own remedy". Echo of the OPEN abstract clause at main:65, "Memory ... is blind to the second" (L-o). |
| H4 | CONFIRMED FIXED (wording) | C1 "we add a mask-free full-label F1, reported for clean-protocol runs as they complete"; Scoring "with the number of seeds marked"; cell `88.8^{n=1}`. Stale: clean s7 (91.19) landed at 07:49, after the 07:33 generation (L-m). Joint and naive full-label are still "--". |
| H5 | CONFIRMED FIXED | `drops.append(len(kept) < len(vs))`, so the `c` mark appears only when a value was removed; IS3 is now unmarked (`5.6/7.7`); LwF keeps `c`. Footnote: "LwF only ... CPFD and IS3 keep their zero seeds". |
| M1 | CONFIRMED FIXED | Setup:93 lists the early-stopping, epoch-cap and precision groups and the 2×2's precision split; supp `tab:legacyprotocol`. Residual: L-f (the epoch-cap list does not hold on DIL). |
| M2 | CONFIRMED FIXED | main:122 "Where we measured the mechanism, on naive training and LwF". |
| M3 | CONFIRMED FIXED | main:204 "rank 8 ... gives 82.5 (seed 42) ... stuck at 65 and 68" (`recall_d0_seed42`: 82.50, diag 64.9/67.7). |
| M4 | CONFIRMED FIXED | BASE rows: `src(f"cil_cord_{key}", (RESC, ...))`, so clean names match the queue (`cil_cord_lwf`, `cil_cord_cl_lora`). |
| M5 | PARTLY | supp Hyperparameters fixed ("differ between groups ... `tab:legacyprotocol` lists them"). Still stale: supp `tab:legacy` caption "CoLaR, CoLaR-R and the baselines use 10 training epochs per task with early stopping". The June-grid naive/LwF/ER/DER++/CODA/CL-LoRA/joint rows saved no `epochs` (script default 100/30); only CPFD/IS3/LCA show 10. This contradicts the supplement's own `tab:legacyprotocol`. |
| M6 | CONFIRMED FIXED | main:110 "BWT −89.8 and −87.2 over three seeds, −79.1 over the two live text-only seeds"; `tab:depthprofile` text-only n=2. Re-derived exactly (Part B.5). |
| M7 | CONFIRMED FIXED | main:204 "at most 0.6 (seed 42)". |
| L1 | PARTLY | Main Table 1 is now generated (`tab_dil_grid.tex`) and matches the files. Regression: supp `tab:grid` is still typed by hand with the old values (BROS LwF 35.2, BROS DER++ 87.6, BERT ER ±1.0), so it disagrees with main Table 1 (35.1, 87.5, 0.9) on the same runs (L-a). |
| L2 | CONFIRMED FIXED | supp:115 "reported as 6.1±7.3"; "zero on every domain for seed 42 and 3.0/9.7/0.0 for seed 7". |
| L3 | CONFIRMED FIXED | main:102 "on the stream FUNSD→CORD→SROIE"; "Table 1 gives the DIL picture". |
| L4 | CONFIRMED FIXED | main:95 "charged at most once"; C1 the same. |
| L5 | CONFIRMED FIXED | main:204 "Unlike CoLaR, LexSlot stores the token ids of its documents." |
| L6 | CONFIRMED FIXED | `macro()` writes `--` for empty cells. |
| L7 | CONFIRMED FIXED | main:210 "by 5--9 points on average". |
| L8 | CONFIRMED FIXED | `tab:ledger` ‡ "legacy scoring". |
| L9 | CONFIRMED (monitor) | Both clean logs print "Full-label final F1 (unmasked test set)"; no "skipped". |
| R1b `gencner` (WRONG) | CONFIRMED FIXED | .bbl: "2025 International Joint Conference on Neural Networks (IJCNN), pages 1--9, 2025. arXiv:2510.11444". |
| R1b `lca2026` | CONFIRMED FIXED | "Danilo Vasconcellos~Vargas". |
| R1b `kumar2023protoner` | CONFIRMED FIXED | year 2024; the bib has series and volume 492, and the bst does not print the series. |
| R1b `gdibench` | CONFIRMED FIXED | all 16 authors, arXiv:2505.00063. |
| R1b `zhang2023slca` | CONFIRMED FIXED | "Proceedings of the IEEE/CVF ICCV, pages 19091--19101". |
| R1b #1b SpanKL | CONFIRMED FIXED | "SpanKL uses per-type span heads with distillation; neither replays stored examples". |
| R1b #3 REMIND/Latent Replay (HIGH) | CONFIRMED FIXED | "REMIND stores product-quantised mid-level CNN feature maps for each training image ... Like CoLaR, REMIND and Latent Replay keep one entry per example, but each of their examples carries a single label". Consistent with R1b's facts. |
| R1b #4 generative replay | CONFIRMED FIXED | "samples hidden representations from one class-conditional generator". |
| R1b #5 Zhai et al. | CONFIRMED FIXED | "find that multimodal LLMs forget under fine-tuning even when the image encoder is frozen; our drift analysis agrees with the former". |
| R1b #7 EASE/InfLoRA | CONFIRMED FIXED | "need no task identity at test time, either by combining the features of all modules (EASE) or by folding them into the pre-trained weights (InfLoRA)". Consistent with R1b's facts. |
| R1b #8 GDI-Bench | CONFIRMED FIXED | "freezing most parameters, chosen by their update magnitude". |
| R1b #2 ProtoNER (undersold) | FIXED, but a new claim was added (**N5**) | "ProtoNER, the closest work to ours, adds new key-information classes to a LayoutLM-family model from a few labelled documents with prototypical networks". New: "none addresses the background class under label growth". Checked against the paper; it is not supported. |
| R1b #12 CFNER, #13 GenCNER | CONFIRMED FIXED | "causal-effect distillation with curriculum learning (CFNER) or entity-triplet generation". |

### New findings

#### HIGH

**N1. The CL-LoRA cell of the 2×2 is misdescribed.** The code runs neither per-session adapters
nor a trained base.

Where the claim appears:
- `tab_2x2.tex` row label (generator line 199): "Per-session low-rank adapters, no memory".
- `tab:twobytwo` caption: "rank-8 orthogonal adapters on the attention projections, all applied together at inference, distillation loss, base trained".
- main:141: "The third row replaces the memory by per-session low-rank adapters: CL-LoRA, with a shared softmax, collapses".
- teaser bar-group label "per-session adapters", under the CL-LoRA bar too.

Evidence:
- `doccl/methods/o_lora.py:54-66` builds one `LoraConfig` and calls `get_peft_model` once. No
  `add_adapter` / `set_adapter` exists anywhere in `doccl/` or `scripts/train.py`. The one
  `default` adapter is trained in every session. `_ortho_loss` penalises its current `A`
  against **snapshots of itself** taken at earlier session ends (`after_task`).
- PEFT freezes every non-LoRA parameter except the head (`modules_to_save`). The legacy runs
  confirm it: `results/rescore/cil_cord_cl_lora_seed{42,7}` `trainable_params = 489,277`.
  That is exactly 36 rank-8 Q/K/V sites × 12,288 (= 442,368) plus the 61-way head (46,909);
  `total_params` exceeds naive's by exactly 442,368. So the encoder is the frozen pre-trained
  checkpoint from session 1 onward.
- The paper contradicts itself: Related Work lists CL-LoRA among "methods over a frozen
  backbone".
- The module docstring ("per-task LoRA on Q/K/V") does not match the code.

Fix (text, generator label; Paper owns both):
- Row label: "Low-rank adapters, no memory" (or "Low-rank adaptation of a frozen trunk, no memory").
- Caption, CL-LoRA clause: "CL-LoRA (our port: one rank-8 adapter on Q/K/V of the frozen
  pre-trained encoder, shared by all sessions and trained with distillation and an
  orthogonality penalty against its own earlier states; fp16 or fp32 by seed, 30-epoch cap)
  with a shared softmax".
- main:141: "CL-LoRA, one low-rank adapter shared by all sessions with a shared softmax, collapses".
- Teaser label: "low-rank adapters".

Keep "SBR-Recall (one rank-64 delta per session ...)" as is: it is correct (Part C).

#### MEDIUM

- **N2. CL-LoRA / O-LoRA ports are not per-task (fidelity; could change numbers).** Same code
  as N1. Published O-LoRA/CL-LoRA keep one frozen adapter per task; this port retrains one
  adapter and pushes it away from its own past states. That penalty actively erases the old
  adapter. This affects:
  - every CL-LoRA cell (tab:main CIL/Mixed/DIL; supp `tab:prompt_lora` O-LoRA and CL-LoRA rows; image tables);
  - the supp sentence "LoRA adapters acquire but forget as much as naive training does, because the adapter does not protect the head" (the single overwritten adapter is an equally likely cause).

  Fix:
  - disclose in Setup or supp ("our port keeps a single adapter across sessions");
  - drop the causal clause;
  - long term, a per-task port.

  The queued capacity-matched run inherits this (Part C).
- **N3. The Limitations sentence will become false on the next regeneration.** main:210: "Cells
  marked † ... are not compared against clean-protocol cells in the text."
  - When `cil_cord_colar_sbr_seed123` finishes (running since 07:49), `choose()` switches `\vSbr` to clean.
  - The abstract ("attain `\nSbr` against `\nJoint`"), intro, §4.3, §6 and the 2×2 will then compare a clean cell with † cells.
  - Fix now: "A cell never combines the two protocols; † cells and clean cells are compared only where the text says so."
- **N4. "EWC and LwF stay at the naive level" (tab:dil caption, main:122) fails per backbone and per seed.**
  - EWC over naive: LiLT +8.2 (s42 52.32 vs 40.68), BERT +7.2, BROS +4.7.
  - LwF on BROS: −5.1.
  - Fix: "stay within about 8 points of naive and far below replay".
- **N5. The new ProtoNER claim is not supported.** main:85: "none addresses the background class
  under label growth". ProtoNER (arXiv:2310.02372, Sec. 3.3 Training procedure) discards the
  "Other"-class prototypes before every incremental phase and annotates its incremental samples
  for all n+1 classes. Its results discussion explains learning from few samples by
  sub-clusters that future-class words, "labeled as 'Other' during the base training", form
  under the "Other" class. Fix: "ProtoNER avoids the conflict by annotating its few incremental
  documents for every class; none of these works separates the causes of forgetting".
- **M5 residual** (see Part A): supp `tab:legacy` caption, "the baselines use 10 training epochs".

#### LOW

- L-a: supp `tab:grid` hand-typed values disagree with main Table 1 (35.2 / 35.1, 87.6 / 87.5, ±1.0 / ±0.9). Generate it, or copy the generated values.
- L-b: supp `tab:grid` caption, "on CIL-CORD every method, replay included, sits at or below the naive level". False: LwF BROS 23.1 and BERT 24.3 against naive 19.2 and 19.3.
- L-c: supp IS3 prose uses population s.d.: "4.4±1.7" (ddof 1: 2.2), "30.1±1.0" (ddof 1: 1.2; `tab:legacy` prints 30.10±1.18).
- L-d: memory units are mixed.
  - "60 MB" is decimal (DIL `replay_memory_bytes` 60,403,200 = 57.6 MiB).
  - "96 MB" and "54 MB" are MiB (100,672,250 B; 56.6e6 B).
  - Use one unit.
- L-e: "16--19" per seed: ER s7 = 14.43. Write "14--19" or "on average".
- L-f: Setup epoch caps ("100 for ... CPFD and IS3, 30 for ... CL-LoRA") hold only on CIL-CORD and Mixed. DIL CPFD and IS3 are 10; DIL CL-LoRA has script default 100. Add "on CIL-CORD and Mixed".
- L-g: The modality-mask sentence omits the no-text condition, BWT −66.7 over three seeds (it forgets least). Add it.
- L-h: The prose uses `\nColar` (30.4, two seeds) and `\nNoBank` (20.4, one seed) with no n (intro, §4.2, §6). The tables show seeds; the sentences do not.
- L-i: `pick()` drops `v <= 0.5`, not exact zeros as the docstring and footnote say. It is harmless today (only LwF; its dropped values are 0.00). Use `v != 0.0` or fix the docstring.
- L-j: `\vSbrFull`, `\vRecallFull` and `\vJointFull` print an n<3 mean with no n mark. They are unused; do not use them in prose.
- L-k: The v3 driver has no lock. Relaunching it while a run is in progress (no `metrics.json` yet) starts a second GPU job. Add `flock` the next time a driver is written (not while this one runs).
- L-l: `docs/LEGACY_PROTOCOL_TABLE_2026-11.md:99` says IS3 Mixed s42 was "dropped". The generator keeps it, and the doc's own note at line 129 says so.
- L-m: Generated tables are stale. The clean CoLaR-SBR s7 run landed at 07:49; after regeneration the full-label cell reads `88.8/91.2^{n=2}`.
- L-n: "LwF does not collapse in fp32" rests on one fp32 run (CIL-CORD s42, 18.45). Say "in its fp32 run".
- L-o: main:65 "Memory ... is blind to the second" is the body form of the OPEN abstract clause. Resolve the two together.
- L-p: main:198 "every method with a shared softmax stays at or below CoLaR". Joint training also uses a shared softmax. Write "every continual method".

### Part B: numbers

**B.1, mismatches > 0.05: none.** Every cell of `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex` and
`tab_dil_grid.tex` reproduces from the run files (rescore runs are one directory deeper; CoLaR
directories carry `_k4_d50_r128`). Per seed (42/7/123), new or changed cells only (the rest as
in R1):
- CIL-CORD LwF: 18.45/18.51/0.00 → `18.4/18.5^{†,c}`.
- Mixed LwF: 47.75/0.00 → `47.7^{†,c}`.
- CIL-CORD IS3: 5.60/7.67 (no mark).
- Mixed IS3: 0.00/40.59.
- CPFD: CIL-CORD 2.72/0.00; Mixed 0.27/15.09; DIL 0.00/4.25/14.19.
- CL-LoRA: CIL-CORD 18.67/18.59.
- Table 1 LiLT: naive 40.14±0.47, EWC 48.38±5.87, ER 85.65±0.48, DER++ 84.35±0.31, joint 83.79±1.18; LwF has a run directory with no `metrics.json` → "--".
- Table 1 BROS: LwF 35.15±3.10, DER++ 87.55±1.08.
- Table 1 BERT: ER 77.06±0.95.

No cell lacks a source. The selection rule in code is exactly the stated rule, for AA (`pick`)
and for the retention figure (`matrices`). The full-label column can show mean±s.d. only when
three clean seeds exist.

**B.2, clean-protocol runs** (`results/clean_2026-11/`; log config dump + `metrics.json`):

| Run | amp / fp16 | val_frac | epochs cap (stopped at) | cil_restrict_eval | full_label_f1 | AA | BWT | final row of `matrix.npy` |
|---|---|---|---|---|---|---|---|---|
| `cil_cord_smoke_naive_e1_seed42` | false / false | 0.1 | 1 (smoke) | true | 12.55 | 18.05 | −76.09 | 0/0/0/0/90.26 |
| `cil_cord_colar_sbr_seed42` | false / false | 0.1 | 10 (5/5/5/6/6) | true | 88.83 | 87.76 | −2.41 | 91.41/82.78/88.14/83.06/93.40 |
| `cil_cord_colar_sbr_seed7` | false / false | 0.1 | 10 (8/9/9/6/3) | true | 91.19 | 91.32 | −1.61 | 92.07/91.24/92.04/88.52/92.71 |

- Batches per epoch (360/172/247/326/351 at batch 2) equal 90% of the session pools in both seeds.
- `replay_memory_bytes` = 100,672,250 in both seeds.
- s7 has no DONE line in `queue.log` (orphaned by the v1→v2→v3 swap, logged as SWAP); its files are complete.
- `cil_cord_colar_sbr_seed123` is running (v3 pid 3338470).
- The log's "~130.1 MB fp16" bank line overcounts: it counts the newest batch before SVD. `replay_memory_bytes` is the right figure.

**B.3, prose literals.** Every literal in `main.tex` traces to a result file, the supplement, the
evidence ledger or `STATE.md`. New since R1, recomputed: 81--95, 16--19 (L-e), −89.8/−87.2/−79.1,
82.5 with 64.9/67.7, 42.0 (44.17/39.85), 61.2 (61.17), 14.8 (14.77). As in R1, the CKA profile
(1.00/0.78/0.34/0.25/0.16), the freezing numbers (41.1/0.0, 38.8/−76.8, 0.19), the ladder, the
ViT numbers and 163 MB / 80.6 are traced to the ledger and `STATE.md`, not recomputed.

**B.4, comparatives per seed** (only changed or new ones; R1's list still holds):

| Sentence | Check | Verdict |
|---|---|---|
| "plain replay comes close to joint training on every backbone" | ER − joint: −0.9/+1.9/+0.3/−0.2 | OK |
| "EWC and LwF stay at the naive level" | up to +8.2 (mean) and +12 (seed) | **N4** |
| "naive, LwF, ER, DER++ sit at 16--19" | ER s7 14.4 | L-e |
| "Only the cell with both approaches joint" | −5.2/−4.2/−2.3 | OK |
| "sessions reach 81 to 95" | 80.9 to 94.6 | OK |
| "every method with a shared softmax stays at or below CoLaR" | every baseline seed ≤ 26.9 < 28.8 | OK, but joint counts too (L-p) |
| LexSlot "within two points ... narrows the spread" | +1.7 / −0.1 / +0.8; s.d. 0.46 vs 1.97, 0.36 vs 1.25 | OK |
| "Perfect memory labels recover about half" | (61.2−28.8)/(92.8−28.8) = 0.51 | OK |

**B.5, legacy-settings table and depth profile.**
- Every row of supp `tab:legacyprotocol` agrees with `docs/LEGACY_PROTOCOL_TABLE_2026-11.md`:
  caps, precision, memory, seeds, the LwF ‡ and the first-grid * footnote including batch 16.
- `tab:depthprofile`, re-derived from `results/pilot/c*_seed*.json` (dead = final AA < 1.0; dead
  runs: `c1_text_seed123` and the three unused `c1_bert` runs), matches every cell:
  - full 273×, 0.03/0.70/0.24/0.03;
  - text+layout 361×;
  - text-only n=2, 801×, 0.03/0.34/0.51/0.12;
  - no text 247×;
  - LiLT 111×;
  - BROS 738×;
  - BERT 1608×.
- BWT: full −89.8 (−89.51/−89.90/−90.03), text+layout −87.2, text-only −79.1 (−79.41/−78.73), no text −66.7 (L-g). The `_ord210` runs form their own group (−75.2) and carry no displacement records.

### Part C: code review (`git diff ac76a6c..HEAD -- scripts/ doccl/ tests/ configs/`)

**`doccl/methods/o_lora.py` (`layers_to_transform`): inert when unset.** With the key absent,
`layers_to_transform=None` and `layers_pattern=None` are the `LoraConfig` defaults, so earlier
CL-LoRA and O-LoRA results are unaffected (cannot change a paper number).

Sites of the capacity-matched command (v3 `CLR64`: rank 64, alpha 128, targets query/key/value/dense, layers 4..11) against SBR-Recall:
- SBR-Recall (`colar_sbr_recall.py:75-81`) wraps every `nn.Linear` whose last name component is in the targets, in `_encoder_layers()[4:]`. That is 6 per layer (Q, K, V, attention.output.dense, intermediate.dense, output.dense) × 8 layers = 48.
- PEFT with `layers_pattern="layer"` and `[4..11]` hits the same 48. `tests/methods/test_olora_layers.py` pins 48 vs 72; I did not run it (rule 1).
- Rank and alpha match, so per-session capacity matches: 7.08M parameters, one delta's worth.

Differences that remain, and that the caption must state if the run is reported:
1. **one adapter shared by all sessions** vs one delta per session (Recall stores 4×);
2. **base = frozen pre-trained checkpoint**, layers 0..3 never adapted, vs Recall's base
   fine-tuned on session 1 and then frozen;
3. distillation toward the previous model (`kd_alpha` 1.0, T 2), absent in Recall;
4. orthogonality penalty (λ 0.5) against the adapter's own earlier states, absent in Recall;
5. LoRA dropout 0.1 vs none;
6. one inference pass with the current adapter vs one pass per delta;
7. PEFT-wrapped shared-softmax head vs SBR head plus per-session background logits.

Items 1 and 2 are N1/N2: the run is a sites-, rank- and alpha-matched control, **not** a
per-session-adapter control. It could change the 2×2 only as an added row.

**v3 driver (`clean_protocol_v3_2026-11.sh`).**
- Order:
  1. CoLaR-SBR ×3;
  2. Recall ×3;
  3. no-memory SBR ×3;
  4. CoLaR ×3;
  5. joint ×3 (cap 100);
  6. naive ×3;
  7. CL-LoRA ×3;
  8. CL-LoRA-r64 ×3;
  9. ER-250 s42;
  10. Recall o2/o3;
  11. BROS ×3;
  12. ER-250 s7/s123;
  13. LwF ×3;
  14. Mixed ×12;
  15. DIL ×15;
  16. Mixed Recall-c s7/s123.
- Names are prefixes of the run directories: CoLaR runs get `_k4_d50_r128` and BROS runs `_bros`. `cil_cord_cl_lora_seed*` and `cil_cord_cl_lora_r64_seed*` cannot match each other, in the driver's `ls` or in the generator's globs.
- Jobs run sequentially. A failure writes FAIL and moves on, and is retried on relaunch.
- There is no lock against a second launch (L-k).
- At the observed 1.7 to 2.3 h per CIL-CORD run, the joint/naive full-label values arrive around 20 h out and CL-LoRA-r64 around 40 h out. The full queue (about 50 runs) does not fit before the Nov 17 07:00 freeze; Mixed and DIL clean cells will likely stay †.

**`scripts/rca/make_clean_tables.py`.**
- Selection rule correct.
- Drop-zero applies to legacy sources only (`not c`).
- The `c` mark appears only when something was dropped.
- BASE rows read clean names.
- The DIL grid writer reads `results/dil_<m>_seed<s>{"", _lilt, _bros, _bert}` exactly. No suffix leak: LayoutLMv3 uses the empty suffix, so it cannot reach `_bert` and the like.
- Findings: L-i, L-j. Neither changes a printed number today.

**`pilot_depth_profile.py` / `pilot_displacement_fig.py`.**
- Same dead rule: the figure imports `dead_reason`, and `load_pilot_results`' narrower all-zero rule is a subset.
- Excluded runs are printed, not silently dropped.
- The figure filters by `condition`, which would also admit `_ord210` runs, but those have no displacement records.
- Output matches the table.

The other diffs (`analyze_results.py`, `make_image_table.py`, `build_falsification_figures.py`)
change labels and an axis limit only.

### Part D: submission hygiene

| Check | Result |
|---|---|
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10, so the body is pages 1 to 7. |
| Undefined references / citations | PASS. 0 in `main.log` and `supp.log`; 52 cited keys, all in `refs_precompiled.bbl`. |
| Overfull boxes | PASS. 0 in both logs. |
| `---` in paper inputs | PASS. 0 in `main.tex`, `supp.tex`, `fig/*.tex`, `tab_*.tex`, `vals.tex`. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and `supp.pdf`. |
| Anonymity, main.pdf | PASS. "Anonymous CVPR submission"; no names, institution, paths, script names or dates. "6 GB GPU" in the Table 1 caption is benign. |
| Anonymity, supp.pdf | PASS. Exposes in-house instance names (LexMem v3b/v5, HGT/CUBER, in the ledger only), public HF dataset IDs and "6 GB GPU" / "24 GB card". `.tex` comments carry repository paths and dates (`supp.tex` 110-114 and 153-155), so do not ship the source. |
| Every float referenced | PASS. Main: teaser, `fig:conservation`, `tab:dil`, `tab:twobytwo`, `fig:pareto`, `tab:main`, `fig:retention`. Supp: all 14 labels referenced. |
| Pointers main → supp | PASS. Sec. A (setup), D (legacy settings; displacement; `tab:grid`), C (ledger; rebalancing nulls; `tab:image`), B (`tab:image_supp`) each hold what the pointer says. Supp → main: "Sec. 3" (Setup), "Sec. 4.1" (drift), "Sec. 5.1" (CoLaR/ViT) all correct. |
| PDFs current | PASS for text. Tables lag the disk by one clean seed (L-m). |

### OPEN (not counted)
- Abstract clause "an effect that no memory can repair" (user decision; its body echo is L-o).
- CR2: † headline cells, pending data.

## R3 (2026-11-11): confirmation of P4

Scope: `git diff cfccda1..37e7df4 -- paper_cvpr/ scripts/rca/make_clean_tables.py
scripts/rca/make_retention_fig.py`, the PDFs rebuilt at 08:16, and the v4 driver (5be5c26).
Read-only. No model or dataset touched.

### Verdict: **PASS for draft upload**

N1 is fixed everywhere. No CRITICAL or HIGH finding remains besides the waived CR2 and the
user-pending abstract clause. Two MEDIUM items should go into the next Paper round:
- **N4 residual:** the new "within ten points / thirty points below" bounds fail per seed.
- **N6 (new):** the new Sec. 5.3 sentence contradicts the paper's own Related Work.

### Status table

| Item | Status | Current text / cell |
|---|---|---|
| **N1** generator label | CONFIRMED FIXED | `make_clean_tables.py`: `f"Low-rank adapters, no memory & ..."`; `tab_2x2.tex` row "Low-rank adapters, no memory". |
| **N1** `tab:twobytwo` caption | CONFIRMED FIXED | "its two cells differ in adapter design as well as in readout: our port of CL-LoRA (one rank-8 adapter on the attention projections of the frozen pre-trained encoder, shared by all sessions, trained with distillation and an orthogonality penalty against its own earlier states; fp16 or fp32 by seed, 30-epoch cap) with a shared softmax, and SBR-Recall (one rank-64 delta per session on the attention and feed-forward layers above k, base frozen after the first session, fp32, 10-epoch cap)". Matches the code. |
| **N1** Sec. 4.3 | CONFIRMED FIXED | "CL-LoRA, one low-rank adapter shared by all sessions, with a shared softmax, collapses ... SBR-Recall, one low-rank delta per session ... This row is not a controlled swap of the readout". |
| **N1** teaser | CONFIRMED FIXED | Bar label "low-rank adapters". Caption: "one low-rank delta per session with a blocked readout (SBR-Recall) is on par with it without a memory, while a single low-rank adapter shared by all sessions with a shared softmax (our CL-LoRA port) collapses". |
| **N1** leftovers | CONFIRMED FIXED | Every remaining "per-session" in main.tex and supp.tex refers to SBR, SBR-Recall, ICE, scoring or session pools; none refers to CL-LoRA. Cosmetic only: the v4 driver comment (line 40) still says "per-session isolation with a shared softmax". |
| **N2** | CONFIRMED FIXED | main:91: "Our CL-LoRA port trains a single rank-8 adapter on the attention projections of the frozen encoder, shared by all sessions, unlike the published one-adapter-per-task design, so its numbers describe this port." The supp Hyperparameters paragraph states the same for O-LoRA and CL-LoRA, with λ = 0.5 and T = 2, both matching the configs. The causal clause is gone: "Our single-adapter LoRA ports acquire but forget as much as naive training does." |
| **N3** | CONFIRMED FIXED | Limitations: "Every cell is marked by its protocol and never mixes the two, so a comparison across protocols is visible from the marks in the tables". |
| **N4** | **PARTLY (MEDIUM)** | See below. Affects the `tab:dil` caption and main:122 ("EWC and LwF stay within ten points of naive training and at least thirty points below joint training"), and the supp `tab:grid` caption ("regularisation and distillation stay within ten points of naive training"). |
| **N5** | CONFIRMED FIXED | "None of these document-level works separates the causes of forgetting." (The background claim is removed.) |
| M5 residual | PARTLY (now LOW) | The false "10 epochs" claim is gone. The new pointer misleads: "`tab:legacyprotocol` lists them for the rows shared with the main paper". `tab:legacy`'s CIL-CORD and Mixed columns are the earlier grid (`results/{cil_cord,mixed}_<m>_seed*`, legacy scoring), not the rescore runs behind the main paper's † cells; only its DIL column is shared. |
| L1 residual | CONFIRMED FIXED | supp `tab:grid`: BROS LwF `35.1±3.1`, DER++ `87.5±1.1`, BERT ER `77.1±0.9`, equal to `tab_dil_grid.tex`. The table is still typed by hand. |
| Units (L-d) | CONFIRMED FIXED for the file-backed numbers | 101 MB = 100,672,250 B; 57 MB = 4 × 7,077,888 × 2 B = 56.6e6; 60 MB = 60,403,200 B. Note: "163 MB", "about 32 MB" and the pareto labels (0.45/0.39/0.48 MB) come from the ledger and figure constants ("restored from logs"); their unit cannot be confirmed from a `metrics.json`. The logs print MB as bytes/1e6, so decimal is likely. |
| "16--19 AA on average" (L-e) | CONFIRMED FIXED | main:128. |
| IS3 s.d. (L-c) | CONFIRMED FIXED | "AA 4.4±2.2 ... 30.1±1.2". ddof 1 on 5.02/6.17/2.00 and 28.85/31.18/30.26. The run-set scoping is still open; see (a). |
| supp grid caption (L-b) | CONFIRMED FIXED | "on CIL-CORD every method stays below joint training under this scoring, and only LwF (on BROS and BERT) rises above naive training". Checked against every grid cell. Its "within ten points" clause is N4. |
| zero-drop rule (L-i) | CONFIRMED FIXED | `kept = [v for v in vs if v != 0.0] ...`. The output is unchanged (the LwF zeros are exact). |
| no-text condition (L-g) | CONFIRMED FIXED | "the model without text forgets least, −66.7 over three seeds" (−65.92/−66.15/−67.93). |
| L-n, L-p, L-f, L-o | CONFIRMED FIXED | "did not collapse in its fp32 run"; "every continual method with a shared softmax"; "On CIL-CORD and Mixed, epoch caps differ by method (the DIL caps are in the supplement)"; "no memory we tested resolved the second". |

**N4 per seed** (DIL run files; difference to naive of the same seed, and distance below joint
of the same seed):

| Backbone | Method | s42 | s7 | s123 |
|---|---|---|---|---|
| LayoutLMv3 | EWC | +0.1 / 49.5 | −0.7 / 46.7 | +0.3 / 46.3 |
| LayoutLMv3 | LwF | +2.5 / 47.1 | +1.6 / 44.4 | +3.0 / 43.7 |
| LiLT | EWC | **+11.6** / 30.6 | +1.8 / 43.5 | **+11.3** / 32.2 |
| BROS | EWC | +6.3 / 38.9 | +1.2 / 48.3 | +6.7 / 40.1 |
| BROS | LwF | −3.1 / 48.2 | −6.6 / 56.1 | −5.8 / 52.6 |
| BERT | EWC | +6.4 / 31.9 | **+10.3 / 27.2** | +4.9 / 32.8 |
| BERT | LwF | +1.0 / 37.2 | +0.8 / 36.7 | +1.5 / 36.2 |

The bounds hold on the means (largest gain over naive +8.2, LiLT EWC; smallest gap to joint
30.6, BERT EWC) but fail in three runs. Fix: write "on average" in all three places, or use
per-run bounds ("within 12 points of naive training and at least 27 below joint training in
every run").

### New findings

**MEDIUM N6. The new Sec. 5.3 sentence overclaims and contradicts Related Work.** main:187:
"It has no shared-softmax counterpart: per-session deltas need something to select them at test
time, and without blocks the only selector would be a task-identity oracle". Related Work
(main:81) says the opposite: "recent adapter methods add one module per task and need no task
identity at test time, either by combining the features of all modules (EASE) or by folding
them into the pre-trained weights (InfLoRA)". Kim et al. is cited there for task-identity
*prediction*, not an oracle. Fix: "A shared-softmax counterpart would have to combine or merge
the deltas (as EASE and InfLoRA do) or predict the session with a separate model; we did not
run one. SBR's per-session background logit is such a predictor, trained at no extra cost."

**LOW**
- (a) The supp IS3 sentence and `tab:main` use different run sets *and* different scorings.
  - Supp prose ("AA 4.4±2.2 on CIL-CORD ... 30.1±1.2", "diagonal ... on seed 123"; also `tab:legacy`): the earlier grid `results/{cil_cord,mixed}_is3_seed{42,7,123}`. Three seeds, 10-epoch cap, fp32, test-split early stopping, **legacy scoring** (no restriction; it predates c430b87).
  - `tab:main` IS3 cells (5.6/7.7, 0.0/40.6): `results/rescore/{cil_cord,mixed}_is3_seed{42,7}`. Two seeds, **100-epoch cap**, fp32 in effect, test-split early stopping, **session-restricted scoring** (log `cil_restrict_eval: true`).
  - The Mixed seed-42 runs differ (28.85 vs 0.00) although the precision is the same, so the zero is not a rescoring of the same model.
  - The supp sentence must say: "In the earlier grid (legacy scoring, 10-epoch cap, three seeds) IS3 ... gives AA 4.4±2.2 on CIL-CORD ... 30.1±1.2 on Mixed (`tab:legacy`); the session-restricted re-runs in the main comparison (100-epoch cap, two seeds) read 5.6/7.7 and 0.0/40.6."
  - The CPFD sentence in the same paragraph ("AA 2.7 and 0.0 on seeds 42 and 123"; "on Mixed it reads 6.0, 0.0 and 0.1") is also from the earlier grid and needs the same qualifier.
- (b) CoLaR-R, sliced. A row is needed in `tab:legacyprotocol`, because main:130 quotes a legacy number from it: "42.0 (two seeds)" = `results/rescore/cil_cord_colar_r_sliced_seed{42,7}` (44.17/39.85). The row: "CIL-CORD | cap 10 | fp16 (log `amp: true`, CoLaR loop wired) | 50 docs/session | 2 seeds" (test-split early stopping, restricted scoring).
  - The same sentence's oracle refresh, "61.2 (seed 42)" = `gates/.../cil_cord_colar_r_oracle_seed42_k4_d50_r128`, is fp32, cap 10, 50 docs, 1 seed, and is not listed either.
  - `docs/LEGACY_PROTOCOL_TABLE_2026-11.md` has neither.
  - `tab:legacy`'s own CoLaR-R row (25.38±2.38 / 61.30±1.91) is a third run set (`results/{cil_cord,mixed}_colar_r_sliced_seed*_k4_d50_r128`: fp32, cap 10, 3 seeds, legacy scoring). It is covered by fixing the M5 pointer.
- M5 pointer, suggested text: "Its CIL-CORD and Mixed columns are runs of the earlier grid, not the † cells of the main paper: CoLaR, CoLaR-R, LexSlot, CPFD, IS3 and LCA ran in fp32 with a 10-epoch cap; the other rows saved no configuration (grid default cap 100, 30 for CODA-Prompt). The DIL column is the main paper's."

### Regenerated tables

Only three things changed:
- the row label;
- the CoLaR-SBR full-label cell, `88.8/91.2^{n=2}` (clean s42 88.83, s7 91.19);
- `\vSbrFull` 90.0 (= 90.01; unused in prose).

Every other cell of `vals.tex`, `tab_2x2.tex` and `tab_main_clean.tex` is byte-identical to R2.
`retention_cilcord.pdf` was regenerated from the same inputs (CoLaR n=2 legacy, SBR n=3 gate,
Recall n=3 gate).

### Clean runs

There is no new finished run. `cil_cord_colar_sbr_seed123` is running (orphan of v3, pid 3395398,
30 min at 08:20, no `metrics.json`); v4 (pid 3470572) waits on it.

### Queue driver v4 (`clean_protocol_v4_2026-11.sh`)

- **Lock:** `exec 9>$P/.driver.lock; flock -n 9 || {... REFUSED ...; exit 1}` before the wait. Children get `9>&-`, so the lock dies with the driver. This closes L-k for v4-style drivers; v1 to v3 are dead.
- **Order:**
  1. CoLaR-SBR, Recall-r64, no-memory SBR, CoLaR, joint and naive on CIL-CORD;
  2. CL-LoRA;
  3. ER-250 s42;
  4. Recall o2/o3;
  5. Mixed SBR and joint;
  6. BROS;
  7. ER-250 s7/s123;
  8. LwF;
  9. the rest of Mixed;
  10. DIL;
  11. Mixed Recall-c.

  No `cl_lora_r64`.
- **Names:** unchanged from v3; every run name is a prefix of its run directory (`_k4_d50_r128`, `_bros`), and no two names prefix each other.
- **Failure:** same as v3: FAIL logged, the queue continues, and a relaunch retries.
- **Caveat:** the clean ER cell needs ER-250 s7/s123, which come after BROS, so the ER row will likely stay † at the freeze.

### Hygiene

- Body: "References" starts on page 8 of 10.
- `main.log` and `supp.log`: 0 undefined, 0 overfull.
- `---`: 0 in every paper input.
- Em-dash: 0 in both PDFs.
- Abstract: byte-identical to 0a896ef (sha1 `fa56d93a…` for both).
- `supp.pdf` is now 8 pages.

## R4 (2026-11-11): audit of the complete paper and appendix

Snapshot: `HEAD = c53cf54`; `main.pdf` 09:13 (after `main.tex` 09:12), `supp.pdf` 09:10 (after
`supp.tex` 09:10), generated `supp_*.tex` 09:09. Tracked paper files are clean against HEAD.
Recomputation: own scripts on raw `metrics.json` / `matrix.npy` / logs, `statistics.stdev`
(ddof 1). No model, dataset, torch, `doccl` import or pytest touched.

### Verdict: **PASS for draft upload**

No CRITICAL or HIGH finding remains besides the waived pending-data item (CR2) and the
user-pending abstract clause. Every generated appendix table reproduces from the run files
(no mismatch > 0.05), the supervision-plan table matches the code and the logs cell by cell,
and the new SBR-Recall "cannot drift" statement is correct in the case it is scoped to.
Six MEDIUM items should go into the next Paper round:
- two of them are descriptions that the paper's own appendix matrices contradict (R4-M1, R4-M4);
- one is a generator change needed before the clean CoLaR-SBR seed 123 lands (R4-M6).

### MEDIUM

**R4-M1. The retention descriptions are contradicted by the matrices and the figure they describe.**
- Where:
  - main `fig:retention` caption: "With a shared softmax every session decays as soon as the next arrives";
  - main "Retention per session" (new in c53cf54): "With a shared softmax each session decays as soon as the next arrives";
  - supp App. D.1: "With a shared softmax every column decays as soon as the next session arrives";
  - supp App. D.1, blocked readout: "its column loses a few points over the following sessions".
- Evidence. `supp_mat_colar.tex` (CoLaR rescore, seeds 42/7, the same matrices the figure plots):
  - column 3 goes 37.1 → **66.3** → 53.8;
  - column 4 goes 34.0 → **40.9**;
  - per seed, s42 column 3 is 22.5 → 60.6, and s7 column 1 is 95.4 → 37.1 → 55.3 (it rises again at session 3).
- Evidence. `supp_mat_sbr.tex`: column 2 goes 94.1 → 79.3 on the mean, and 94.4 → 72.8 on s42 (−21.6).
- Fix (text only):
  - shared softmax: "the first two sessions lose 30 to 50 points at the next boundary; later sessions are learned only to 19 to 37 F1 and fluctuate afterwards";
  - blocked readout: "loses between 1 and 15 points (session 2 most, 94.1 to 79.3)".
  The main paragraph's "slow loss on the earliest sessions" can stay.

**R4-M2. The Conclusion's SBR-Recall sentence is unscoped against the paper's own Mixed result.**
- Sentence: "When label growth supplies a routing key, per-session low-rank deltas (SBR-Recall)
  can replace the memory."
- Mixed has label growth and per-session background logits (the routing key, as Sec. 5.3
  defines it). Yet SBR-Recall without a memory scores 44.7 there (`mixed_colar_sbr_recall_r64b_seed42`).
- Sec. 5.3 itself gives a different failure: the all-background terms have no positive
  counterweight, not a missing key.
- Contribution 3 and Limitations are scoped correctly ("on a single-schema stream"; "does not
  replace memory ... across annotation schemas").
- Fix: "On a single-schema stream, where label growth supplies the routing key, per-session
  low-rank deltas (SBR-Recall) can replace the memory; across annotation schemas they still need it."
- The abstract clause "when label growth itself supplies the routing key" has the same gap. The
  abstract is user-agreed, so that part goes to the PM as NEEDS DECISION. I do not block on it.

**R4-M3. The Conclusion repeats the OPEN universal.** The sentence is "a better memory does not
repair a readout that supervises one background class with contradictory labels".
- It is the body form of the OPEN abstract clause ("no memory can repair"), and R2 L-o had it
  removed from the introduction.
- Evidence: one seed of true-label refresh (61.2), two fp16 seeds of replay slicing (42.0), and
  three seeds of both-sliced (62.9/69.2/49.4).
- Fix: "in our experiments even a memory with true labels did not repair ...". Resolve it together
  with the abstract clause.

**R4-M4. A typed stale-label count is wrong.** App. E: "A deployable variant that refreshes with
the model's own confident predictions rewrites none on seeds 42 and 7".
- `results/logs_vision/mixed_colar_r_pseudo_seed42_k4_d50_r128.log` (2026-10-15 13:14:55):
  "refreshed 200 docs, **84** tokens relabelled (169 candidates)". The counts are 0/0/0/84 on s42.
- Seed 7 is 0/0/0/0 and seed 123 is 7/0/0/36, as typed.
- Ledger row F10c and `STATE.md` (ticks of 2026-10-17 06:09 and 2026-10-16 11:46) carry the same error.
- Fix: "rewrites 0/0/0/84 tokens on seed 42, none on seed 7 and 7/0/0/36 on seed 123 (true-label
  refresh: 3120 to 3307 at the first refresh)". "Almost no" still holds: 84 tokens against
  6,076 oracle tokens on that seed (3120 + 2640 + 0 + 316), though 84 against 316 at that boundary. Correct the ledger
  (PM/Dev).

**R4-M5. The reason given for the all-background mode is false for SROIE and CORD.**
- Where: main Sec. 5.2 ("so none of its entities can occur in the document") and App. C, bullet
  "all-background" ("none of its entities can occur").
- Both are receipt schemas, and a CORD receipt has a total, a date and usually a company name. Those
  are SROIE entities in kind, annotated under CORD's tags.
- The rule is a convention, not a fact about the document: the document is annotated under its
  own schema, so no token carries one of b's tags, and every token is treated as a negative for b.
- Fix: "when b belongs to a different annotation schema: the document is annotated under its own
  schema, so we treat every token as a negative for b (a convention that uses the schema of each
  session at training time; SROIE and CORD share entity kinds such as totals and dates)". The
  Limitations sentence on the schema already covers the rest.

**R4-M6. `supp_slot.tex` will mix protocols at the next regeneration.**
- `write_slot` pairs `C["sbr"]` (the main-table source) with the gate LexSlot runs.
  `cil_cord_colar_sbr_seed123` is running in `clean_2026-11` (pid 3395401). When it lands,
  `choose()` switches `C["sbr"]` to clean, and the CIL-CORD row will difference clean CoLaR-SBR
  against gate LexSlot, seed by seed.
- Prose built on that row will then also compare across protocols:
  - App. D ("$\vSbr$ against $\vSlot$ on CIL-CORD");
  - main Ablations ("narrows the spread between seeds on CIL-CORD"). The clean seeds 87.8/91.3 are
    already wider than gate 87.6/86.3/90.1.
- Fix (Paper owns the generator):
  - in `write_slot`, read the CIL-CORD CoLaR-SBR side from `src(None, (GATE, "cil_cord_colar_sbr", ""))`,
    the same fp32 gate protocol as LexSlot, and label the column;
  - in the prose, replace `\vSbr` by a gate-only macro;
  - or switch the LexSlot cell only when clean LexSlot runs exist.

### LOW

- **R4-L1 (Part A.4, nuance).** "neither the base, nor its delta, nor its rows of the readout
  receive a gradient after its session: old blocks cannot drift, and routing by $e_b$ is unchanged."
  - Correct for gradients.
  - The old tag rows (and their bias entries) live in the one classifier tensor, so AdamW's
    decoupled weight decay still shrinks them. The bound is ≤ 0.5%: 9,665 steps after session 1
    on `recall_r64_seed42`, × lr·λ = 5e-7. The old $w_b$ have no gradient and are skipped.
  - Each old block's $e_b$ is unchanged, but the arg-max over blocks changes as new blocks
    compete. Old sessions' F1 moves by −0.8 to +1.3 in the r64 matrices.
  - Suggest: "each old block's $e_b$ is unchanged (up to weight decay); only competition from new
    blocks moves its tokens".
- **R4-L2.** $o_b=w_b^\top h(x)$ omits the bias of `nn.Linear(hidden, 1)`.
- **R4-L3.** Alg. 1 line "AdamW step on layers $k{+}1..12$, head, all $w_b$": in session 1 every
  layer trains (the freeze comes after). Add "(session 1: all layers)".
- **R4-L4.** App. E calls the control a "frozen trunk": the takeaway says "or on a frozen trunk",
  and a later sentence says "learned at 87.0 on the same frozen trunk". The run is `split_layer_k=11`:
  layer 12, the head and $w_b$ stay plastic (trainable 7,098,638 = one layer + 13×769 + 769).
  Write "an almost frozen trunk (only the last layer and the readout plastic)". The body, the
  conflict-table label ("layers 1..11 frozen") and App. E's own sentence are correct.
- **R4-L5 (labels_full, Paper agent's question).** See Part D.
- **R4-L6.** The deltas are counted at 2 B per parameter, but they are trained and evaluated in fp32.
  No run evaluated fp16-rounded deltas. Write "57 MB if stored in fp16 (113 MB in fp32 as run)",
  or evaluate once.
- **R4-L7.** `tab:legacyprotocol` lacks rows for the App. E gate controls, although `tab:conflict`'s
  caption points there:
  - CoLaR gate 34.1, replay-sliced gate 39.4, both-sliced ×3, + pseudo, bidirectional and the
    k=11 control are all fp32, cap 10, 50 docs (none for k=11). Verified from `training_hparams.amp`
    and `method_hparams`.
  - The rank-8 Recall runs are missing too.
  - Add one row.
- **R4-L8.** `supp_delta.tex` has no protocol mark, and the `tab:delta` and `tab:recallmix` captions
  say "fp32 gate runs" without †. Add †: these are test-split early-stopped runs.
- **R4-L9.** The main Setup now says only that legacy cells differ "in epoch cap (10 to 100) and
  precision". The first DIL grid's unconfirmed settings (batch size 16) and the per-row precision
  are only in App. G. That is acceptable with the pointer; optionally add "(App. G)" to the
  `tab:dil` caption.
- **R4-L10.** App. B, "so the restricted AA of the main comparison is not inflated by the coverage
  gap", rests on two seeds of one method. Scope it to CoLaR-SBR.
- **R4-L11.** Main Sec. 4.2 says "App. E collects these controls with their seeds and the number of
  stale labels the memory holds". The counts in App. E are Mixed (earlier grid), not CIL-CORD. Say
  so, or add the CIL-CORD oracle counts, which are in the logs:
  - s42: 101/760/1461/1402;
  - s7: 66/944/1366/1269;
  - s123: 74/943/1315/1110.
- **R4-L12.** App. C: "the receipt session reached 23.6". DIL has two receipt sessions; it is the
  SROIE one (`dil_colar_sbr_seed42` diagonal 86.7/23.6/97.9, AA 66.49).
- **R4-L13.** App. A: the held-out split is seeded by "run seed + session index"
  (`split_val(..., cfg.seed + task_idx)`; joint uses `cfg.seed`), not by the run seed alone.
- **R4-L14.** The generator has a regeneration race. `matrix.npy` is written about a minute before
  `metrics.json` (clean s42: 05:28 vs 05:29). A regeneration in that window moves a matrix (and
  `retention_cilcord.pdf`) to clean while its table cell stays legacy. Regenerate only after the
  DONE line.
- **R4-L15 (anonymity, supp.pdf).** What `supp.pdf` exposes:
  - the package path `doccl/data/scenarios.py`;
  - dated run paths `results/clean_2026-11` and `results/gates/cil_protocol_2026-11`;
  - script names (`scripts/rca/make_clean_tables.py`), run-directory names (`mixed_colar_sbr2_seed42`, ...)
    and `_CORD_CANONICAL_SESSIONS`.

  None of these identifies anyone by itself, but `doccl` could match a public repository. I
  recommend generic wording ("our scenario builder", "the generator script"), no dates and no
  package name. "RTX 2060 (6 GB)", "14 GB host memory" and "24 GB card" are acceptable.
  `main.pdf`: no paths, names, dates or package names.
- **R4-L16.** App. A, "two annotation schemas per form type", is unclear. Suggest: "three
  annotation schemas (FUNSD for forms; SROIE and CORD for receipts)".
- **R4-L17.** `alg:recall` is not `\cref`'d inside the supplement; main cites it as typed "Alg. 2",
  which is the correct number.

### Part A: the body as a finished paper

**A.1 Regressions against R1–R3.** None of substance.
- These are still fixed: H1 (16–19 on average); H2/N1 (CL-LoRA wording in the 2×2 caption, Sec. 4.3,
  and in the new Experiments sentence "those that keep a frozen encoder with a shared softmax (CODA-Prompt,
  our CL-LoRA port)"); H3 (intro); M2, M3 (82.5), M6, M7; N3, N4 (per-run 12/27), N6; L4, L-d, L-p.
- The new Conclusion reintroduces the H3/L-o pattern once (R4-M3) and drops the single-schema scope
  once (R4-M2).
- The Setup paragraph lost its per-group cap and precision list to App. G (R4-L9).

**A.2 New body claims since 37e7df4** (`git diff 37e7df4..HEAD -- paper_cvpr/main.tex`):

| Claim | Check | Status |
|---|---|---|
| Mixed: "two class-incremental FUNSD sessions, a shift to SROIE, two class-incremental sessions on CORD's super-classes and a FUNSD revisit with all its labels" | `build_mixed` sessions 0–5 | OK |
| † cells differ "in epoch cap (10 to 100) and precision" | `tab:legacyprotocol` | OK (L9) |
| "A cell switches to the clean protocol only when all three clean seeds exist, so no cell combines the two" | `choose()` | OK; `supp_slot` will break it (M6) |
| `tab:dil` caption "On average ... within ten ... thirty"; body "within 12 ... at least 27 ... in every run" | R3 per-seed table (max +11.6, min 27.2) | OK |
| 2×2 caption precision split | legacy-protocol rows; gate runs `amp=false` | OK |
| "Training in brief" | `colar_sbr.py`, `latent_replay.py` | OK; the replay term is a mean over the four documents, not a sum (trivial) |
| SBR-Recall "cannot drift" | code (A.4) | OK in its scope (L1) |
| "A session that adds no block reuses the delta of the first block it trains fully" | `before_task` | OK. Edge case: a session with neither a new block nor a fully trained block gets an unused fresh delta; it never occurs. |
| "shared-softmax counterpart ... we did not run one" | (N6 fix) | OK |
| "The table repeats the 2×2 ...": ER/DER++/CoLaR strong on DIL (87.9/88.0/87.8), far below joint on CIL-CORD (24.3/20.4/30.4); CPFD, IS3, CODA-Prompt, CL-LoRA do not recover CIL-CORD (≤18.7) | `tab_main_clean.tex` | OK |
| "Retention per session" | matrices | **M1** |
| Ablations: no-memory 20.4; k=11 control 14.8, "sessions after the second are barely learned" (diagonal 4.3/22.0/0.0); rank 64 learns sessions 3–4 on all seeds (88.8–96.2) | gate runs | OK |
| Cost: "Per-run training times are in App. H" | App. H points to `tab:perseed` (App. D) | OK |
| Conclusion | below | M2, M3 |

**A.3 Conclusion.**
- The drift claim is scoped to a fixed label space ("on a fixed label space only old-task supervision
  of the readout repaired it in our experiments"). OK.
- "survives a memory with true labels": one seed (61.2), the same wording as contribution 2. Acceptable.
- "resolved at the readout": consistent with Sec. 4.3 ("in every cell we ran").
- Recall sentence: M2.
- "a better memory does not repair": M3.
- "session-restricted scoring with a mask-free full-label F1 makes both measurable": acceptable,
  though full-label F1 exists for 2 runs only.

**A.4 "Old blocks cannot drift"** (`colar_sbr_recall.py`, `colar_sbr.py`, `latent_replay.py`, wrapper).

Mechanics:
- After task 0, `freeze_backbone()` freezes all of `layoutlmv3` (embeddings and all 12 layers),
  leaving the classifier trainable. `sbr_o` is re-enabled.
- `_train_only_current_delta` leaves only the current (A, B) trainable.
- The optimiser is rebuilt each session (`_make_optimizer`), over `trainable_parameters()`.

Per case:

| Case | What an old block b receives after its session | Paper |
|---|---|---|
| Single schema (CIL-CORD), no memory | Plans print `trains [(s,'full')]` only (logs of `recall_r64_seed42` and clean SBR s42). Base and Δ_b: frozen. $w_b$: not in the graph (grad None, skipped by AdamW). Tag rows $z_{\mathcal{T}_b}$: zero gradient, AdamW decay only (≤0.5%). | Correct as stated (L1) |
| Single schema, with memory (`recall_seed42`, rank 8, d50) | Replayed session-b documents train block b fully under Δ_b. Base and Δ_b are frozen, but $w_b$ and the $\mathcal{T}_b$ rows are trained, so the readout moves (session 4: diagonal 68.0 → final 65.1). | Not claimed; App. D's "within 1.5 points" is scoped to "without a memory" (exact: 64.94 → 66.44 = 1.50). OK |
| Across schemas, no memory (`mixed_..._r64b`) | Current documents of another schema train old blocks in "neg" mode under their own delta. $w_b$ and the rows are pushed to O, so the blocks see-saw. | App. D says exactly this. OK |
| Across schemas, with memory (`r64c`) | Neg terms plus full replay terms under own delta. | App. D. OK |
| Mixed revisit | `full=[0,1]` → reuses `block_delta[0]` = None (base). No delta is trainable; the readout only. | App. C. OK |

The pre-fix runs (CIL-CORD r8 / d0 / r64 ×3, all before fb2960c at 11-09 05:31) are unaffected,
as App. C says. In the old code, replayed documents ran under `_session_delta(s)` (own block's
delta), and current batches under the current delta with plan = own block only. That is
identical on a single schema.

**A.5 Mixed description vs `build_mixed`: OK.**
- Sessions: FUNSD {H, Q}, FUNSD {A}, SROIE (4 fields, 8 tags), CORD-super {menu, sub_total},
  CORD-super {total, void_menu, sub}, FUNSD revisit with all labels.
- The head reaches 25 tags; the revisit adds none.
- App. A pools from steps per epoch (75/74/313/400/399/75, batch 2) are 149/147–148/625–626/799–800/797–798/149. OK.

**A.6 Frozen-trunk control: OK in the body** ("every encoder layer but the last frozen after the
first session").
- `audit_gates_2026-11.sh:40` sets `method.docs_per_task=0 method.split_layer_k=11`.
- The log says "layers <11 + embeddings frozen; trainable params=7098638".
- `metrics.json` has `split_layer_k: 11`, `amp: false`.
- App. E wording: L4.

**A.7 Body → appendix pointers: all resolve.**

| Pointer | Holds |
|---|---|
| App. A | sessions and pools (`tab:cilsessions`, `tab:mixsessions`) |
| App. B | worked example and `tab:agree` |
| App. C | Alg. 1 and Alg. 2, `tab:plan`, memory and parameter accounting |
| App. D | per-seed `tab:perseed`, matrices, Mixed Recall matrices, delta capacity, LexSlot |
| App. E | conflict controls, with seeds; stale-label counts are Mixed (L11) |
| App. F | displacement / "late layers", ledger, ViT image section, rebalancing nulls |
| App. F and G | EWC sign flip (`fig:signflip`, `tab:grid` Mixed rows) |
| App. G | every † setting (L7 gap) |
| App. H | wall-clock |

The typed "Tab. 3" and "Sec. 4.1" in the supplement are the correct main-paper numbers, and so
is "Alg. 2" in main.

### Part B: the appendix, number by number

**B.1 Generated tables, recomputed: no mismatch > 0.05.**

Registry mapping checked. Every glob resolves to exactly one run directory, with one
`metrics.json` each; no top-level run directory holds two.
- **Exact-suffix sources:**
  - rescore and grid sources use `""`: `cil_cord_*_seed42` cannot reach `_bert`;
  - `dil_colar_e10` uses `_k4_d50_r128`;
  - the gate CoLaR / CoLaR-R controls use `_k4_d50_r128`.
- **No collisions between run names:**
  - `cil_cord_colar_sbr_recall_r64` does not reach `r64b`/`r64c` (those are Mixed);
  - `cil_cord_colar_r_sliced2` does not reach `sliced2_pseudo`;
  - `cil_cord_colar_sbr` (`""`) does not reach `_slot`, `_frozen`, `_noreplay` or `_recall*`;
  - clean `cil_cord_colar_sbr*` cannot reach `..._bros_seed*_bros`.
- **Methods, scenarios and seeds** are as labelled. Every row carries its protocol tag: clean,
  gate†, rescore† or grid†. Matrix titles name seeds and protocol.

Per-seed values (42/7/123), as recomputed:
- **`supp_perseed`.**
  - CIL-CORD:
    - naive 18.97/19.08/18.60;
    - no-memory SBR 20.36;
    - CoLaR 28.83/31.87;
    - SBR 87.62/86.27/90.15;
    - Recall 92.69/93.37/92.44;
    - joint 92.84/90.48/92.43;
    - ER 26.91/22.05/24.01;
    - LexSlot 90.15/89.72/89.23;
    - LwF 18.45/18.51/0.00ᶜ;
    - DER++ 21.27/18.76/21.07;
    - CODA 13.96/15.89;
    - CL-LoRA 18.67/18.59;
    - CPFD 2.72/0.00;
    - IS3 5.60/7.67;
    - LCA 13.72/6.97.
  - Mixed:
    - naive 44.40/43.68;
    - CoLaR 75.56/74.55;
    - SBR 89.77/89.62;
    - joint 86.94/88.46;
    - ER 75.58/73.33;
    - LexSlot 88.48/89.99/90.32;
    - Recall 85.94;
    - LwF 47.75/0.00ᶜ;
    - DER++ 75.14/73.95;
    - CODA 12.08/10.59;
    - CL-LoRA 45.73/43.90;
    - CPFD 0.27/15.09;
    - IS3 0.00/40.59;
    - LCA 37.96/36.42.
  - DIL:
    - naive 40.06/43.03/40.85;
    - CoLaR 87.58/88.30/87.38;
    - SBR 86.76/84.77/87.08;
    - joint 89.68/89.03/87.49;
    - ER 86.21/88.73/88.62;
    - LexSlot 87.24/87.25/86.62;
    - LwF 42.54/44.63/43.82;
    - DER++ 88.23/87.06/88.78;
    - CODA 33.40/31.11/33.02;
    - CL-LoRA 40.82/41.34/41.27;
    - CPFD 0.00/4.25/14.19;
    - IS3 38.29;
    - LCA 46.35/43.80/41.45.
  - h/run (`total_wall_time_s`) matches.
- **`supp_conflict`:**
  - CoLaR rescore 28.83/31.87;
  - CoLaR gate 34.06;
  - sliced rescore 44.17/39.85;
  - sliced gate 39.38;
  - both sliced 62.86/69.23/49.45;
  - + pseudo 48.01;
  - bidirectional 44.11;
  - true-label refresh 61.17;
  - no-memory SBR 20.36;
  - k=11 control 14.77.

  Configurations confirmed. Gate runs are `amp=false`, epochs 10, d50, k4, r128; the k=11 run is
  d0. Rescore runs are `amp=true`. `bilc` = `sliced_pseudo` + `current_sliced` + `current_pseudo_tau` 0.9.
- **`supp_agree`:** clean SBR s42 87.76 / 88.83 (+1.07); s7 91.32 / 91.19 (−0.13).
- **`supp_slot`:**
  - CIL-CORD +2.53 / +3.45 / −0.92;
  - Mixed (rescore) −1.29 / +0.37;
  - Mixed (gate 89.73) −1.25;
  - DIL +0.48 / +2.48 / −0.46.
- **`supp_delta`:** rank-8 + memory, rank-8 d0, and r64 ×3: every diagonal, final-row and AA cell
  matches `matrix.npy` (e.g. r8 d0: diagonal 95.3/91.6/64.9/67.7/90.3, final row 95.4/91.9/66.4/68.4/90.3, 82.50).
- **Six `supp_mat_*`.** Means over the stated seeds match every cell. Per seed:
  - CoLaR s42/s7 rescore;
  - SBR ×3 gate;
  - Recall r64 ×3 gate;
  - SBR Mixed s42/s7 rescore;
  - Recall Mixed r64b and r64c, s42 gate.
- **Unchanged since 37e7df4:** `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex` and `tab_dil_grid.tex`.
  They are absent from `git diff --stat 37e7df4..HEAD` and clean in the worktree.

**Protocol labels for tables that will switch.**
- Tables that label their protocol correctly today: `supp_perseed`, `supp_conflict`, `supp_slot`
  (one tag per row) and `supp_mat_*` (title).
- Will become wrong: `supp_slot`, at the next regeneration (M6).
- No protocol mark: `supp_delta` (L8).

**B.2 Typed numbers, traced.**

| Number (App.) | Source | Status |
|---|---|---|
| 62.9/69.2/49.4, "about 60 ... 20-point spread" (E) | `gates/..._sliced2_seed*` 62.86/69.23/49.45; mean 60.5, spread 19.8 | OK |
| 48.0 with pseudo-labels; 44.1 bidirectional (E) | 48.01; 44.11 | OK |
| 61.2; diagonal sessions 3–5 31.8/14.4/23.0 (E) | oracle s42 61.17; diagonal 31.82/14.43/23.01 | OK |
| CoLaR fp32 34.1; replay sliced fp32 39.4 (E) | 34.06; 39.38 | OK |
| 42.0 over two seeds (E) | 44.17/39.85 → 42.01 | OK |
| k=11 control 14.8; diagonal 4.3/22.0/0.0; session 2 87.0 (E) | 14.77; 4.33/22.04/0.00; 87.02 | OK |
| No-memory SBR 20.4, "old blocks fall to 0 to 5" (E) | 20.36; final row 0.48/0.0/1.25/4.64 | OK |
| True-label refresh counts 3120–3307 / 2640–2932 / 0 / 305–316 (E) | logs: s42 3120/2640/0/316, s7 3307/2932/0/316, s123 3231/2792/0/305 | OK |
| Pseudo-label refresh "none on seeds 42 and 7", 7/0/0/36 on 123 (E) | s42 **0/0/0/84**, s7 0/0/0/0, s123 7/0/0/36 | **M4** |
| Calibration 85.2 → 85.4 → 85.8; DIL ≤0.7 / ≤1.1 (E) | 85.16 recomputed (`rescore/cil_cord_colar_sbr_seed42`, `amp=true`); 85.4/85.8 and 0.7/1.1 from `STATE.md` ticks 2026-11-05 09:30 / 00:40 | traced, not recomputed (needs `eval_dump.pt` via torch) |
| 402,689 = 378,112 + 24,576 + 1 (C) | 2·128·(709+768) = 378,112; (512·4 bbox + 512 mask + 512 labels)·8 = 24,576; +1 | OK |
| ×250 = 100,672,250 (C) | `replay_memory_bytes` of gate SBR ×3 and clean s42/s7 | OK |
| 272.3 MB uncompressed (C) | 250·709·768·2 = 272,256,000 | OK |
| 4,096 B per document labels_full (C) | 512 × int64 | OK (Part D) |
| 7,077,888 per rank-64 delta; 48 sites (C) | per layer 4·98,304 + 2·245,760 = 884,736, × 8; log "48 LoRA sites (r=64)" | OK |
| 56,623,104 bytes, "every rank-64 run" (C) | `replay_memory_bytes`: r64 s42/s7/s123 and Mixed r64b | OK |
| 294,912 per rank-8 delta; 2.4 MB for four (C) | 8·768·2·3·8; `recall_d0` 2,359,296 | OK |
| Mixed with memory 177,429,804 = 300 × 402,689 + 56,623,104 (C) | r64c `replay_memory_bytes`; four deltas (the revisit reuses the base) | OK |
| 3.7× = (4 + 5·8)/12 (C) | arithmetic | OK |
| CIL-CORD pools 800/382/549/723/780 (A) | gate steps 400/191/275/362/390 (×2, ±1); clean 360/172/247/326/351 = 90% | OK |
| Mixed pools (A) | see A.5 | OK |
| 61 tags; 12 per session; 25 Mixed; 709 positions (A) | code; 512 + 197 | OK |
| 33.3 legacy joint (B) | 33.43/33.57/32.94 | OK |
| SBR-Recall on Mixed: 62.1 bug run; 44.7; 85.9 (D) | r64 62.07 (matrix shows the wrong-delta damage); r64b 44.66; r64c 85.94 | OK |
| See-saw: 87.6/90.8 → 0.0; SROIE 82.0 → <10; CORD 95.8/95.5; revisit 87.3/85.6, others ≤8.1 (D) | r64b matrix 87.55/90.80 → 0.00/0.00; 81.99 → 9.94; 95.77/95.54; 87.32/85.63; 0.26/8.12/0.00 | OK |
| With memory: no column at zero; CORD >96; FUNSD-2 46–48, 75.0 (D) | r64c 46.15/46.64/47.91 → 75.05; 96.71–97.66 | OK |
| Delta: "within 1.5 points"; sessions 3–4 at 61–68; r64 88.8–96.2 (D) | 1.50 max; 61.5/68.0/64.9/67.7; 88.77–96.19 | OK |
| LexSlot "within two points ... below SBR-Recall" (D) | +1.7/−0.1/+0.8; every seed < Recall | OK (until M6) |
| `dil_colar_sbr_seed42` AA 66.5, receipt diagonal 23.6 (C) | 66.49, diagonal 86.7/23.6/97.9 | OK (L12) |
| Wall-clock: 2 h 01 min vs 1.95 h; SBR 2–2.5 h; Recall 1.5 h; joint <1 h (H) | queue.log 03:28 → 05:29; 1.953 h; gate SBR 2.40, clean 1.95/2.27; Recall 1.49; joint 0.35 | OK |
| `tab:prompt_lora` (G) | L2P/DualPrompt/CODA/O-LoRA/CL-LoRA/ER+C-Flat++ on LMv3 and BERT | all 24 cells OK |
| Rebalancing base 87.6; metaplastic 85.9 → 79.5; budget-adaptive +0.6±7.5 (F) | `dil_colar` s42 87.58; `STATE.md` colar_meta 85.85/79.48; mean/sd of +4.0/−8.0/+5.7 = 0.57/7.47 | OK (the last two traced) |

**Typed numbers I could not trace, or did not re-trace in R4.** They were carried over from the
pre-P6 supplement and are not new:
- App. A: "roughly 200 GPU-hours on a 24 GB card". No file records it.
- App. F:
  - the Fisher magnitudes "0.22 against 0.005";
  - the consolidation probe 42.6/84.7/86.1/86.2 and ER 87.9 with a 200-doc buffer;
  - lexical routing +1.8/+0.4/−0.6 and uniform 86.4;
  - the three-seed rebalancing check 87.8/87.0;
  - PLaR 76.1/~7/58.9;
  - the image paragraph (42.3/21.6, 76.2/54.8, 38.0/25.8, 10.7);
  - the ridge gaps 9.5/16.1, 7.5/14.3, 2.9/15.0, 7.7.

  The image and ridge tables are generated by other scripts outside R4's scope.
- Ledger table (F): the rows for merge, LCA 43.9±2.5, HGT/CUBER, LexSlot 42.2, the input-anchored
  ledger, LexMem 66.0/63.2/13.0, and CoLaR-R 20.0/59.5 (plain 20.4/58.8, oracle 26.3/64.4).
  These are traced in R1–R3 to `docs/EVIDENCE_LEDGER_2026-10-22.md` only.
- App. G: the CPFD/IS3/CODA null paragraph. R2/R3 checked it except the ablation F1s (0.00, 2.1)
  and CODA n50 (79.5/40.6/0.0/18.9/36.8, AA 7.4).

### Part C: method descriptions against the code

**Supervision plan (`tab:plan`): matches cell by cell.** I re-derived it from `supervision_plan`
and `task_domain` (domain from the task name: cord, funsd, sroie), using the `label_set` of each
session (own-session tags only; logs print 12 tags per CIL-CORD session).
- CIL-CORD: a document trains only its own block, with all other cells "skipped" (same schema).
- Mixed:

  | Session | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 |
  |---|---|---|---|---|---|
  | 1 | F | – | N | N | N |
  | 2 | – | F | N | N | N |
  | 3 | N | N | F | N | N |
  | 4 | N | N | N | F | – |
  | 5 | N | N | N | – | F |
  | 6 | F | F | N | N | N |

- The logs agree on every current-session row:
  - `gates/.../mixed_colar_sbr2_seed42.log`: session 2 trains [(0,neg),(1,neg),(2,full)], ...,
    session 5 [(0,full),(1,full),(2,neg),(3,neg),(4,neg)];
  - `clean_2026-11/cil_cord_colar_sbr_seed42.log`: [(s,full)];
  - `mixed_colar_sbr_recall_r64c` prints the same.

**Alg. 1 vs `colar_sbr.py`.**
- Block creation from new tags: OK.
- 10% held-out split: OK.
- Current loss: Σ over plan of `_block_ce`, each a token-mean CE on [o_b, z_{T_b}], with full
  targets from `block_targets` (in-block tag → 1+pos; any other valid token, including other
  blocks' entities → 0) and neg targets all 0. The neg weight `sbr_neg_weight` is 1.0 in every
  config and run; Alg. 1 omits it and App. H states it. OK.
- Replay: 4 documents, U·V rebuilt, injected at layer k+1 (pre-hook on `layer[k]`, 0-indexed);
  per-document `_session_loss` under the storing session's plan over the current blocks; mean over
  documents; weight 1. OK.
- AdamW: L3.
- Restore best, then freeze at s=1, then capture 50 documents from the frozen state (order in
  `_after_task_impl`). OK.
- Inference (`aggregate_blocks`): $e_b$ = 1 − softmax(...)[0]; strict `> threshold` (0.5); ties
  go to the lowest block; restriction via `model._allowed` applied **after** the block decision
  (`restrict_in_method`). OK.
- Nothing the code does that matters is missing.

**Alg. 2 vs `colar_sbr_recall.py`.**
- s=1 trains the base (`_cur_delta=None`), then `freeze_backbone()`: OK.
- A new block gives a fresh delta (A Kaiming, B = 0); trainable: current delta, head, all $w_b$: OK.
- Else reuse `block_delta[full[0]]`: OK. Unstated edge case: no new block and no full block gives
  a fresh, unused delta; it never occurs.
- Loss: `_loss_under_deltas` groups by delta, one forward per distinct delta, reusing the current
  forward: OK.
- Replay: grouped by session, injected under each block's delta: stated in the App. C text, not in
  the algorithm. Acceptable.
- Inference: one forward per distinct delta, then `aggregate_blocks`: OK.
- Delta form: $\tfrac{\alpha}{r}xA^\top B^\top$ on every Linear named query/key/value/dense in
  layers ≥ k (48 sites): OK.

**Sec. 5 equations.**
- Eq. 1 (block softmax), Eq. 2 (masked sum, $y_b$) and Eq. 3 (block under θ₀+Δ_b) match the code.
- The mask rule matches `supervision_plan`, but the stated reason is M5.
- $o_b$ bias: L2.
- Per-token vs per-block token-mean normalisation: immaterial.

**Scoring (App. B) vs `train.py`.**
- `restrict_preds`: the classifier hook maps an out-of-session argmax to O (`bad` → O logit 1e4),
  never re-argmaxes, and is eval-only. For SBR, `model._allowed` is applied after the block decision.
- `cil_allowed`: the session's `label_set` (with O). It is a no-op when the label set covers the
  head (DIL), and for images.
- `add_full_label_f1`: final model, outside any restriction (`_allowed` None), on
  `scenario.full_eval_dataset` (CORD test, 100 receipts, 61-tag head). Exceptions are swallowed
  (R1 L9; both clean logs print the value).
- Worked example: correct (P 1/2, R 1, F1 0.67; cross-type confusion = 1 FP + 1 FN).
- "Charged at most once" and the coverage caveat: correct.

### Part D: generators and hygiene

| Check | Result |
|---|---|
| `choose()` | Unchanged (diff shows only context). `pick` is refactored through `_candidates` with identical semantics. `chosen_source` and `matrix_choice` call the same `choose()`, so the source and seeds always come from the same rule. `make_retention_fig.py` now imports `matrices` (same rule, same seed order). Risks: M6 (`write_slot` pairing), L14 (race). |
| Side effects | `vals.tex`, `tab_2x2.tex`, `tab_main_clean.tex`, `tab_dil_grid.tex` and `figures/retention_cilcord.pdf` are byte-identical to 37e7df4 (no git diff). |
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10 (page 8 opens with the Ablations paragraph, then Conclusion). |
| Undefined references / overfull boxes | PASS. 0 / 0 in `main.log` and `supp.log`; one benign `h`→`ht` float warning in supp. |
| `---` in paper inputs | PASS. 0 in `*.tex` and `fig/*.tex`, generated files included. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and `supp.pdf` (14 pages). |
| Abstract | Byte-identical to 0a896ef (sha1 `fa56d93a…`). Still contains the OPEN clause. |
| Anonymity | `main.pdf`: PASS (no names, paths, dates, package names; empty PDF metadata). `supp.pdf`: R4-L15. |
| Every float referenced | PASS. Main: all figures and tables `\cref`'d. Supp: every table/figure/algorithm except `alg:recall` (L17). |
| Every appendix section referenced | PASS. The roadmap `\cref`s A–H; main cites A, B, C, D, E, F, G, H. |
| New clean runs since R3 | **None.** `cil_cord_colar_sbr_seed123` has been running since 07:49 (pid 3395401, no `metrics.json`); v4 (pid 3470572) waits on it. On disk: smoke naive (18.05 / −76.09 / 12.55), SBR s42 (87.76 / −2.41 / 88.83), SBR s7 (91.32 / −1.61 / 91.19). Flags as R2: `amp` and `fp16` false, `val_frac` 0.1, `cil_restrict_eval` and `full_label_eval` true. |

**`labels_full` (Paper agent's question; my independent reading).**
- **It is stored.** On CIL-CORD (`build_cil_cord`: `_wrap("train", full_head=snapshots[-1])`) and
  on Mixed (`build_mixed`: `CIL_LabelRemapper(..., snapshots[-1])`), every training item carries
  `labels_full`, the document's labels in the *final* 61- or 25-tag space. That includes classes of
  sessions not yet seen.
- `LatentReplay._capture_task` copies `bbox`, `attention_mask` and `labels_full` into each stored
  record. `CoLaR._capture_task` keeps them (only `hidden` is replaced by factors). So every
  CoLaR, CoLaR-SBR and SBR-Recall store on these scenarios holds it: 512 × int64 = **4,096 B per
  document**.
- **It is never read.**
  - `CoLaR._stack_replay` builds the replay batch from hidden, labels, bbox, attention_mask,
    input_ids, plus session for SBR. `labels_full` is not in it.
  - The only reader is `colar_r.py` (oracle refresh). It casts the field to int16; every other
    `colar_r` mode pops it at capture.
  - No result of CoLaR, SBR or Recall can depend on it.
- **Uncounted bytes per record.** `memory_bytes` omits `labels_full` and the `n_labels` scalar
  (int64, 8 B). It counts the session id (int64 in memory) as 1 B.
  - The record as held is 378,112 + 24,576 + 4,096 + 8 + 8 = **406,800 B**, against 402,689
    counted (+1.0%).
  - CIL-CORD CoLaR-SBR: 101,700,000 B as held, against 100,672,250 counted.
  - Mixed CoLaR-SBR: 122,040,000 B against 120,806,700.
  - Mixed Recall with memory: 178,663,104 B against 177,429,804.
  - DIL is unaffected: no `full_head`, and 60,403,200 = 150 × 402,688 exactly.
- **What the paper should state.** The figures are the method's memory, so they are not
  understated as a claim about the method. They are understated by 1% as a description of what the
  code holds. App. C already discloses the field correctly. I would add two things:
  1. The field holds labels of later sessions, which a continual learner does not have. It is a
     diagnostic artefact that the methods never read and a deployment would not store.
  2. With it, the record is 406,800 B (101.7 MB on CIL-CORD).

  No change to the body's "101 MB" is needed if the sentence says "the method's memory".

### OPEN (not counted)
- Abstract clause "an effect that no memory can repair" (user decision). The Conclusion now has a
  second echo (R4-M3), and the abstract's "when label growth itself supplies the routing key" has
  R4-M2's scope gap.
- CR2: † headline cells, pending data.

## R5 (2026-11-11): confirmation of P7 and the first clean cell

Snapshot: `HEAD = 4c43de9` (commits 17f3720, b8617f1, 4c43de9 on top of c53cf54). `main.pdf` and
`supp.pdf` were built at 09:55, after `main.tex` (09:55) and `supp.tex` (09:42). The generated
files are from 09:55, after the clean s123 `metrics.json` (09:53). Tracked paper files are clean
against HEAD.

Scope: `git diff c53cf54..4c43de9 -- paper_cvpr/ scripts/rca/` and the rebuilt PDFs. My own
scripts recomputed every changed number from `metrics.json`, `matrix.npy` and the logs. No model,
dataset, torch or pytest was touched.

### Verdict: **PASS for draft upload**

Every R4 MEDIUM item is fixed, and so is every LOW item the Paper agent claimed. The switch to the
clean protocol moved exactly the cells it should and nothing else. Every sentence comparing
CoLaR-SBR with another cell is true of the current values. There is no CRITICAL, HIGH or MEDIUM
finding. Five LOW items remain; one is a stale sentence in App. G caused by the switch (R5-L1).

### Status of the R4 items

| Item | Status | Current text / evidence |
|---|---|---|
| **M1**, Fig. 4 caption | CONFIRMED FIXED | "With a shared softmax the curves fall steeply and irregularly, some sessions partly recovering later, and later sessions are learned poorly even on their own diagonal; with the blocked readout most sessions hold and the remaining losses are uneven across sessions; under SBR-Recall every curve is flat to within about a point." Mean matrices: CoLaR column 3 is 37.1 → 66.3; clean SBR losses 3.1/9.0/1.6/3.1; the largest Recall per-seed deviation is 1.34 (s7, session 3). |
| **M1**, body "Retention per session" | CONFIRMED FIXED | "... most keep nearly all of their F1, but the losses are uneven, with one early session losing the most". Clean mean: session 2 loses 9.0, the others ≤ 3.1. It is a statement about the figure (the mean), not per seed: on s123 session 2 loses 16.0 and session 3 loses 9.6. |
| **M1**, App. D.1 (`supp_vals.tex`) | CONFIRMED FIXED | "the first two sessions lose \colarDropOne\ [47] and \colarDropTwo\ [28] points ... only to [19] to [37] F1 on their own diagonal ... rises by up to [29] ... loses [2] to [9] points on the mean, session [2] the most ... within [0.5] points of its diagonal". The text says these are computed from the mean matrices shown. Recomputed below. |
| **M2** | CONFIRMED FIXED | Conclusion: "On a single-schema stream, where label growth supplies the routing key, per-session low-rank deltas (SBR-Recall) can replace the memory; across annotation schemas they still need it." Sec. 5.3: "On a single-schema stream it works without a memory because ...". |
| **M3** | CONFIRMED FIXED | Conclusion: "in our experiments even a memory with true labels did not repair a readout that supervises one background class with contradictory labels." |
| **M4** | CONFIRMED FIXED | App. E: "rewrites $0/0/0/84$ tokens on seed 42, none on seed 7 and $7/0/0/36$ on seed 123 ($84$ against the $6{,}076$ that the true-label refresh rewrites on seed 42)". 6,076 = 3120 + 2640 + 0 + 316. |
| **M5** | CONFIRMED FIXED; accurate now | Sec. 5.2: "the document is annotated under its own schema, so no token carries a tag of $b$, and we treat every token as a negative for $b$ (a training-time convention; related schemas such as SROIE and CORD share entity kinds like totals and dates, so on Mixed it can label a real entity of another schema as background)". App. C: "This is a convention ..., not a fact about the document". It is now correct about SROIE/CORD: it names the shared kinds and says the rule can label them background. Wording nit (optional): "a token that is an entity of $b$'s kind". |
| **M6** | CONFIRMED FIXED | `SLOT_PAIRS` reads both sides from the gate protocol: `cil_cord_colar_sbr` with exact suffix (cannot reach `_slot`), `mixed_colar_sbr2`, `dil_colar_sbr2`. New macros: `\vSbrGate` 88.0±2.0 (87.62/86.27/90.15), `\vSlotGate` 89.7±0.5, `\vSbrGateMix` 89.7 (s42 89.73), `\vSlotGateMix` 89.6±1.0, `\vSbrGateDil` 86.2±1.3, `\vSlotGateDil` 87.0±0.4. `supp_slot.tex`: one gate† row per scenario, differences unchanged from R4. Main Ablations: "In the fp32 gate runs, the only protocol in which both exist (App. D), ... ($\nSbrGate$ against $\nSlotGate$ on CIL-CORD)". |
| L-Alg. 1 | CONFIRMED FIXED | "AdamW step on layers $k{+}1..12$ (session 1: all layers), head, all $w_b$". |
| L1, "cannot drift" | CONFIRMED FIXED | Sec. 5.3: "each old block's $e_b$ is unchanged up to weight decay, and only competition from new blocks can move its tokens (App. C)". The new App. C paragraph gives "under $1\%$ over a run". Check: 10 epochs × (191+275+362+390) steps × 5e-7 = 0.61%. It also gives "about a point in either direction" (r64: −0.8 to +1.3). |
| L4, almost frozen trunk | CONFIRMED FIXED | App. E takeaway and both later sentences now read "almost frozen trunk (only the last layer and the readout plastic)"; also "on the same trunk". |
| L8, dagger on `supp_delta` and two captions | CONFIRMED FIXED | Seed column `42$^{\dagger}$` …. `tab:delta`: "fp32 gate runs$^\dagger$, ..., test-split early stopping". `tab:recallmix` and `tab:slot` captions carry †. |
| L7, legacy-settings rows | CONFIRMED FIXED | Three new rows: "Gate controls, tab:conflict \| CIL-CORD \| 10 \| fp32 \| 50 docs/session \| 1 or 3"; "SBR, layers 1..11 frozen \| 10 \| fp32 \| none \| 1"; "SBR-Recall, rank 8 \| 10 \| fp32 \| 50 docs/session, none \| 1, 1". All agree with `training_hparams` / `method_hparams` (R4 B.2). |
| GPU-hours line | FIXED, new LOW | Replaced by "Every run fits a single 6 GB GPU; a CoLaR-SBR run on CIL-CORD takes about two hours of training loop" (clean 1.95/2.27/2.00 h). The first clause is new and only partly true (R5-L3). |
| L15, internal names and paths in `supp.pdf` | CONFIRMED FIXED | pdftotext of `supp.pdf`: 0 hits for `doccl`, `results/`, `scripts/`, `.py`, `2026-1x`, `_seed`, `_CORD`, `clean_2026`, `cil_protocol`. Run names are paraphrased ("rank 64, seed 42"). Remaining hardware mentions (6 GB, RTX 2060) are acceptable. `main.pdf`: 0 hits. |
| L2, bias term | CONFIRMED FIXED | Sec. 5.2 "$o_b(x)=w_b^{\top}h(x)+c_b$"; App. C "a linear layer with bias". |
| L6, delta storage | CONFIRMED FIXED | Cost: "($57$ MB if stored in half precision, $113$ MB in fp32 as trained and evaluated)". App. C: "This is a reporting choice. ... no run evaluated half-precision-rounded deltas." 4 × 7,077,888 × 4 B = 113.2 MB. |
| L5, memory disclosure (PM decision) | CONFIRMED | Cost "$100.7$ MB on CIL-CORD, the method's memory as counted in App. C". App. C: "As held, a record is $406{,}800$ bytes against $402{,}689$ counted: $101.7$ against $100.7$ MB ... $122.0$ against $120.8$ MB for the $300$ of Mixed"; Mixed Recall "$177.4$ MB counted ... ($178.7$ MB as held)". Matches R4 Part D. |
| L9, L10, L11, L12, L13, L16, L17 | CONFIRMED FIXED | `tab:dil` caption "(... saved; App. G)"; App. B "so for CoLaR-SBR the restricted AA ..."; Sec. 4.2 "the number of stale labels the true-label refresh rewrites"; App. C "the SROIE session ... (one seed, AA 66.5)"; App. A "seeded by the run seed plus the session index (the run seed for joint training)"; App. A "three annotation schemas (FUNSD for forms; SROIE and CORD for receipts)"; App. C "\cref{alg:recall} summarises training and inference". |
| New CIL-CORD refresh counts (App. E) | CONFIRMED | `results/logs_vision/cil_cord_colar_r_oracle_seed{42,7,123}_k4_d50_r128.log` give 101/760/1461/1402, 66/944/1366/1269 and 74/943/1315/1110. The gate run behind the 61.2 (`gates/.../cil_cord_colar_r_oracle_seed42.log`) prints the same 101/760/1461/1402. So "Under restricted scoring it reaches 61.2" does not conflate two different label rewrites. |
| L14 (race) | not a text item; did not bite | This regeneration ran at 09:55, after the s123 `metrics.json` (09:53). |

**`supp_vals.tex`: how it is computed and my recomputation.** `write_supp_vals` averages the
matrices that `matrix_choice` returns, so it uses the same source and seeds as `tab:matrices`:
- CoLaR: rescore, s42/s7;
- CoLaR-SBR: clean, 3 seeds;
- Recall: gate r64, 3 seeds.

It then computes:
- `colarDropOne` = R[1,1] − R[2,1] = 94.66 − 48.12 = **46.5 → "47"**;
- `colarDropTwo` = R[2,2] − R[3,2] = 81.95 − 54.39 = **27.6 → "28"**;
- late diagonal (sessions 3 to 5): 37.07 / 34.00 / 18.57 → **"19" to "37"**;
- largest later rise in any column: column 3, rows 3 → 4, 37.07 → 66.31 = **29.2 → "29"**;
- SBR loss, diagonal − final, sessions 1 to 4 on the clean mean: 3.14 / **8.98** / **1.57** / 3.08 → "2" to "9", session "2";
- Recall: max |diagonal − final| of the mean = **0.54 → "0.5"**.

All match. The App. D.1 sentence says "on the mean", so the per-seed spread is out of its scope:
SBR session 2 loses 9.4 on s42, 1.6 on s7 and 16.0 on s123, and session 3 gains 6.6 on s42. The
body and Fig. 4 say "about a point" for Recall, which holds per seed (max 1.34).

### The switched cell: CoLaR-SBR on CIL-CORD (clean, three seeds)

- **Run.** `results/clean_2026-11/cil_cord_colar_sbr_seed123`:
  - flags: `amp` false, `fp16` false, `val_frac` 0.1, `cil_restrict_eval` true, `full_label_eval` true (log config dump);
  - method: epochs 10, docs 50, k 4, r 128;
  - `replay_memory_bytes` 100,672,250;
  - AA 86.16, BWT −8.55, `full_label_f1` 87.13 (log: "Full-label final F1 (unmasked test set): 87.13"; no "skipped" line);
  - training loop 2.00 h.
  - It has no DONE line in `queue.log`: v3 orphaned it at 08:14, and v4 started the next job at 09:53. The files are complete.
- **Recomputed** (42/7/123):
  - AA 87.76 / 91.32 / 86.16 → **88.41 ± 2.64**;
  - full-label 88.83 / 91.19 / 87.13 → **89.05 ± 2.04**.
  - These match `\vSbr` 88.4±2.6, `\vSbrFull` 89.1±2.0 and the 2×2 and main-table cells, now unmarked.
- **What moved.** `git diff c53cf54..4c43de9 --stat -- 'paper_cvpr/*.tex'` touches only:
  - `vals.tex`: `\vSbr`, `\vSbrFull`, plus the six new gate macros;
  - `tab_2x2.tex`: CoLaR-SBR cell;
  - `tab_main_clean.tex`: CoLaR-SBR CIL-CORD AA and full-label;
  - `supp_perseed.tex` and `supp_conflict.tex`: the CoLaR-SBR row → "clean & 87.8 & 91.3 & 86.2 & 2.1";
  - `supp_agree.tex`: + s123;
  - `supp_mat_sbr.tex`: clean mean, which I recomputed to 0.05;
  - `supp_slot.tex`: M6;
  - `supp_delta.tex`: daggers only;
  - `supp_vals.tex`: new;
  - `figures/retention_cilcord.pdf`: 21397 → 21403 bytes, regenerated at 09:55 from the same `C` registry. Only the SBR source changed, so only the middle panel changed.
- **What did not move.** Under `choose()`, no other cell has three clean seeds: clean Recall s42 is running with no `metrics.json`, and no other clean run exists. Every other generated value is byte-identical, including:
  - Mixed SBR (rescore 89.8/89.6†) and DIL SBR (gate 86.2±1.3†);
  - the supp Mixed matrices;
  - the gate CoLaR-SBR runs, which are still used for the LexSlot comparison only.

**Sentences that compare CoLaR-SBR on CIL-CORD with another cell.** Per seed against legacy joint
(92.84/90.48/92.43): −5.1 / +0.8 / −6.3. Against seed-matched naive, SBR closes 93.1% / 101% /
91.5% of the naive-to-joint gap.

| Location | Text | Verdict |
|---|---|---|
| Abstract (source byte-identical, renders 88.4 instead of 88.0) | "together they attain $\nSbr$ average F1 against $\nJoint$ for joint training" | True as stated: it is a plain report of two means and claims no closeness. The cross-protocol pairing is visible in `tab:twobytwo` (88.4 unmarked, 91.9†). The PM should tell the user that the rendered number changed. |
| Intro ¶3 | "the two together reach $\vSbr$, against $\vJoint$ for joint training" | True. |
| Contribution 3 | "closes most of the gap to joint training on class-incremental CORD and reaches it on a mixed stream" | True per seed (≥ 91% of the gap on every seed). "Reaches it" on Mixed: 89.77/89.62 vs joint 86.94/88.46, both legacy. The change from "within a few points" was right: 6.3 points on s123 is not "a few". |
| Teaser caption | "together they close most of the gap to it ... bars are the cell means of tab:twobytwo, which gives the seeds and protocol of each cell" | True, and protocol-visible. |
| Sec. 4.3 | "Only the cell with both approaches joint training ($\vSbr$ against $\vJoint$)" | True. "Approaches" is not a per-seed "within" claim. |
| Experiments | "the two blocked readouts close most or all of the gap: CoLaR-SBR $\vSbr$ and SBR-Recall $\vRecall$ against $\vJoint$" | True. CoLaR-SBR is clean, the other two are legacy; this is visible from `tab:main`'s marks. |
| Ablations | "what SBR still loses to joint training is retention of early sessions (Fig. 4)" | Consistent with the clean matrix: mean diagonal-to-final losses sum to 16.8, i.e. 3.4 AA of the 3.5-point gap. |
| Conclusion | no CoLaR-SBR number | — |
| App. D.1, D.3 | `supp_vals` macros (clean mean) | True (above). The LexSlot comparison is now gate-only (M6). |
| App. B | "differ by about one point or less on each seed ... for CoLaR-SBR ... not inflated" | True: +1.1 / −0.1 / +1.0. |
| App. E takeaway | "a session-blocked readout with the same memory moves it to $\nSbr$" | True. The row is tagged "clean" in `tab:conflict`, beside gate† and rescore† rows. |
| App. G | "The CIL-CORD cells of CoLaR-SBR, no-memory SBR, SBR-Recall and LexSlot ... are fp32 gate runs" | **Stale (R5-L1).** |

The Limitations promise ("Every cell is marked by its protocol and never mixes the two, so a
comparison across protocols is visible from the marks in the tables") holds for every table. It
does not hold for Fig. 4, whose panel titles give n but not the protocol (R5-L2). Each comparison
favours the reference rather than SBR, because the legacy joint runs early-stopped on the test
split.

### New findings (LOW only)

- **R5-L1. App. G is stale after the switch.**
  - Where: supp.tex:494, "The CIL-CORD cells of CoLaR-SBR, no-memory SBR, SBR-Recall and LexSlot, and the DIL SBR cells, are fp32 gate runs. In the $2{\times}2$ the shared-softmax cells are therefore fp16 or mixed and the blocked-readout cells fp32."
  - The CoLaR-SBR CIL-CORD cell is now clean. The main 2×2 caption is still correct: "Among the † cells, the blocked-readout ones ran in fp32".
  - Fix: drop "CoLaR-SBR" from the list, and write "In the $2{\times}2$ the shared-softmax $\dagger$ cells are fp16 or mixed and the blocked-readout $\dagger$ cells fp32". The `tab:legacyprotocol` row "CoLaR-SBR \| CIL-CORD \| fp32 \| 3" can stay, but should say "(LexSlot comparison only)".
- **R5-L2. Fig. 4 does not show protocol per panel.** The middle panel is now clean and the outer
  two are legacy (CoLaR rescore†, Recall gate†). Add "(CoLaR-SBR: clean protocol; CoLaR and SBR-Recall: $\dagger$)" to the caption, or put the protocol in the panel titles.
- **R5-L3. The new App. A sentence "Every run fits a single 6 GB GPU" is not supported for every run.**
  App. H says some first-grid rows ran on a rented GPU container, and Table 1's "--" cells did not
  fit. Write "The runs reported here ran on a single 6 GB GPU, except some first-grid rows (App. H)".
- **R5-L4. `supp_agree.tex` lists seeds in the order 123, 42, 7.** That is the `sorted(glob)` string
  order. Sort by `SEEDS`.
- **R5-L5. Note for the PM.** The clean s123 run has the largest forgetting of the three clean seeds
  (BWT −8.55; session 2 drops 92.0 → 76.0), and no text cites a CoLaR-SBR BWT. Clean BWT is
  −2.41 / −1.61 / −8.55 against gate −7.50 / −8.99 / −4.30.

### Hygiene

| Check | Result |
|---|---|
| Body ≤ 8 pages | PASS. "References" starts on page 8 of 10. |
| Undefined references / overfull boxes | PASS. 0 / 0 in `main.log` and `supp.log`. |
| `---` in paper inputs | PASS. 0 in `*.tex` and `fig/*.tex`, generated files included. |
| Em-dash in PDFs | PASS. 0 in `main.pdf` and in `supp.pdf` (now 15 pages). |
| Abstract | Byte-identical to 0a896ef (sha1 `fa56d93a…`). It renders "attain 88.4 average F1 against 91.9". |
| Anonymity | PASS for both PDFs (see L15 above). |

### Other clean runs

None has finished besides s123. `cil_cord_colar_sbr_recall_r64_seed42` started at 09:53 (v4,
pid 3470572); its directory holds only `tb/`.

### OPEN (not counted)

- Abstract clause "an effect that no memory can repair" (user).
- Abstract clause "when label growth itself supplies the routing key" (user). The Paper agent
  proposes adding "on a single-schema stream", which I support: it matches the corrected
  Conclusion and Sec. 5.3.
- CR2: the remaining † headline cells, pending data.

## R6 (2026-11-11): audit of P8, the clean SBR-Recall cell

Snapshot: `HEAD = d559dd7`. Tracked paper files are clean against HEAD. The generated files (15:13)
postdate the last clean Recall `metrics.json` (s123, 14:54). `main.pdf` and `supp.pdf` (15:14/15:15)
postdate `main.tex`. I rebuilt both from `git archive d559dd7` in a scratch directory, so no tracked
file was touched. The pdftotext of each rebuild is byte-identical to the committed PDF.

Scope: `git diff d50d267..d559dd7 -- paper_cvpr scripts/rca/make_clean_tables.py scripts/rca/make_retention_fig.py`.
My own script recomputed every number from `metrics.json`, `matrix.npy` and the logs. No generator,
model, dataset, torch or pytest was run. The clean `cil_cord_colar_sbr_noreplay_seed42` was running
during the audit (started 14:54, only `tb/` so far). No paper file reads it.

### Verdict: **PASS**

The three runs are clean and complete, and every Recall value they feed recomputes exactly. The
diff moves only the Recall cells, plus a reordering of `supp_agree`. Every comparative sentence
about SBR-Recall holds per seed. The single cross-protocol pairing (clean Recall and CoLaR-SBR
against fp16 legacy joint) is stated truthfully, in the 2×2 caption and again in App. G. R5 L1–L4
are fixed. There is no CRITICAL, HIGH or MEDIUM finding. Five LOW items follow.

### A. The three clean SBR-Recall runs

**Config.** The config dumps (log lines 1–40) are identical except for `seed`:
- `method.name colar_sbr_recall`, `lora_rank 64`, `lora_alpha 128`, targets query/key/value/dense;
- `epochs 10`, `docs_per_task 0`, `split_layer_k 4`;
- `amp false`, `fp16 false`, `val_frac 0.1`, `cil_restrict_eval true`, `full_label_eval true`;
- batch 2 with gradient checkpointing, `output_dir results/clean_2026-11`.

`metrics.json` `training_hparams` / `method_hparams` agree with the dumps. `rank_r: 128` is inert
because there is no memory.

**Completion.** Every run has a DONE line in `queue.log`:
- s42: 09:53 to 11:27;
- s7: 11:27 to 13:07;
- s123: 13:07 to 14:54.

Each log has "Saved matrix", "Saved metrics" and "Full-label final F1" lines (91.53 / 93.52 / 92.62),
with no "skipped" line and no traceback. The only ERROR lines are the HF `trust_remote_code` notice,
which every run shows (CoLaR-SBR clean s42 too).

**Other fields.** `replay_memory_bytes` is 56,623,104 on all three: the four deltas at 2 B per
parameter (4 × 7,077,888 × 2), i.e. the "57 MB if stored in half precision" of the Cost paragraph.
`matrix.npy` equals `metrics.json["matrix"]` on all three.

**Matrices (rows: after session i).** No collapsed session:
- diagonals 92.6/86.9/94.5/85.6/94.9 (s42), 95.0/89.0/90.5/91.7/94.3 (s7), 94.1/89.3/91.1/90.7/94.4 (s123);
- BWT +0.60 / +0.64 / −0.06;
- the only exact zeros are the first super-diagonal (the zero-shot reading of a session's not-yet-added labels). Clean CoLaR-SBR and gate Recall have the same zeros, so they are systemic, not a fault.

**Recomputed (42/7/123).**

| Value | Per seed | Mean ± s.d. | Generated | OK |
|---|---|---|---|---|
| AA | 91.35 / 92.62 / 91.86 | 91.94 ± 0.64 | `\vRecall` 91.9{\pm}0.6, `\nRecall` 91.9; 2×2 and main-table cell bold, unmarked | ✓ |
| full-label F1 | 91.53 / 93.52 / 92.62 | 92.56 ± 0.99 | `\vRecallFull` 92.6{\pm}1.0; main-table full-label cell, no n mark | ✓ |
| full − AA | +0.18 / +0.89 / +0.76 | | `supp_agree` +0.2/+0.9/+0.8, seeds in 42/7/123 order | ✓ |
| h/run (`total_wall_time_s`) | 1.45 / 1.53 / 1.67 | 1.55 | `supp_perseed` "clean & 91.4 & 92.6 & 91.9 & 1.6" | ✓ |
| mean matrix | | | `supp_mat_recall` (title "seeds 42/7/123, clean") equals my mean to 0.05 in every cell | ✓ |
| `\recallDevMax` = max \|diag − final\| of the mean | session 2: 88.37 → 89.68 | 1.31 | "1.3" | ✓ |
| `\nRecallJointGap` = max \|Recall − joint\| | −1.49 / **+2.14** / −0.57 | | "2.1" | ✓ |
| `\nRecallMaxLoss` = largest R[i,j] < R[j,j] | s123 session 4, 90.66 → 90.11 = 0.549 | | "0.5" | ✓ (rounding: see L1) |
| `\nRecallMaxGain` | s42 session 2, 86.86 → 88.89 = 2.03 | | "2.0" | ✓ |

The joint reference is −1.49 at s42 and +2.14 at s7; Recall is above joint at s7.

**Joint cell.** `\vJoint` 91.9{\pm}1.3 (92.84 / 90.48 / 92.43) is `results/rescore/cil_cord_joint_seed{42,7,123}`.
All three log "Mixed precision enabled" plus a `GradScaler`, i.e. fp16 on the Turing card. Their
config dumps match the clean joint recipe in every key except `amp: true` (and the absent
`full_label_eval`):
- `epochs 100`, `val_frac 0.1`, `cil_restrict_eval true`;
- batch 2, gradient checkpointing, lr 5e-5, wd 0.01, clip 1.0.

So this cross-protocol pairing differs **only in precision**, and the "(fp16)" in the 2×2 caption
is correct.

**What moved.** In the generated files, the diff touches:
- `vals.tex`: `\vRecall`, `\vRecallFull`, and the three new macros;
- `tab_2x2.tex` and `tab_main_clean.tex`: the Recall row only;
- `supp_perseed.tex`: the Recall row only;
- `supp_mat_recall.tex`;
- `supp_vals.tex`: `\recallDevMax` only;
- `supp_agree.tex`: three Recall rows plus the seed reorder;
- the figure.

Every other generated cell is byte-identical (`git diff --stat`). Rule 6 holds: Recall switched with
3/3 clean seeds. Clean no-memory SBR has 0 complete seeds and is ignored by `choose()`. No other
clean source exists.

**Generator review.**
- `recall_macros()`: the gap uses `chosen_source` of both cells, seed-matched, with abs: the right definition for "within X on every seed". The moves are `mat[i,j] − mat[j,j]` for i ≥ j over the matrices of the chosen source, on every seed. That is exactly "largest fall below / rise above its diagonal at any later boundary, on any seed". `max(0, −min)` is correct because the diagonal term contributes 0.
- AA (`metrics.json`) and matrices (`matrix.npy`) are chosen by two separate `choose()` calls. They agree here, and could only disagree if a run had one file and not the other (train.py writes the matrix first).
- Seed sort: `(scenario, method, SEEDS.index)` is correct.
- `ROW_LABEL` maps `colar_sbr_recall_r64` → "SBR-Recall" (see L5 for BROS).
- `make_retention_fig.py`: it now uses `matrix_choice` (the same rule as the tables). Its titles read "CoLaR (n=2, †)", "CoLaR-SBR (n=3, clean)", "SBR-Recall (n=3, clean)", which is correct (CoLaR = rescore s42/s7). Its output path is resolved from `__file__`.

### B. Paper audit

| Location | Rendered text | Per-seed check | Verdict |
|---|---|---|---|
| Teaser caption | "SBR-Recall comes within 2.1 points of it on every seed without a memory" | \|−1.49\|, 2.14, \|−0.57\| ≤ 2.14 | True. The bars read 18.9/20.4/30.4/88.4/18.6/91.9 from `vals.tex`, with the joint line at 91.9. |
| 2×2 caption | "The two bold cells are clean; joint training is still a † cell (fp16), so until its clean runs complete a comparison of either bold cell with it crosses protocols" | Bold = 88.4±2.6 and 91.9±0.6, both clean. The only blocked-readout † cell is no-memory SBR (gate, fp32), which matches "the blocked-readout one in fp32". | True |
| Sec. 4.3 | "SBR-Recall ... reaches 91.9±0.6" | — | True |
| Main comparison | "the two blocked readouts close most of the gap, SBR-Recall all or nearly all of it on every seed" | Gap closed against seed-matched naive (18.97/19.08/18.60): Recall 98.0% / 103.0% / 99.2%; CoLaR-SBR 93.1% / 101% / 91.5% | True |
| Fig. 4 caption | "no session falls more than 0.5 points below its diagonal on any seed, and the larger moves, up to 2.0, are gains" | Max loss 0.549. Every move > 0.55 is a gain (2.03, 1.88, 1.29, 0.63, ...). | True to one decimal (L1) |
| Retention ¶ | "the curves are flat, no session losing more than 0.5 points on any seed" | As above. It omits the gains up to 2.0 that the caption states. | True (L1) |
| Ablations | "rank 64 ... learns them on all three seeds, both in legacy runs ... and in the clean runs of Tab. 3" | Clean diagonals of sessions 3/4: 94.5/85.6, 90.5/91.7, 91.1/90.7, against 61–68 at rank 8 | True |
| Limitations | "visible from the marks in the tables and in the panel titles of Fig. 4" | Titles carry the protocol | True |
| App. B | "For the CoLaR-SBR and SBR-Recall runs listed the two differ by about one point or less on each seed" | Recall +0.2/+0.9/+0.8; SBR +1.1/−0.1/+1.0 | True |
| App. C "cannot drift" | "in the clean rank-64 runs no session falls more than 0.5 points below its diagonal at any later boundary and none rises more than 2.0" | 0.549 / 2.03 | True to one decimal (L1) |
| App. D.1 | "every column of the mean matrix ends within 1.3 points of its diagonal" | 1.31 (mean, session 2, a gain) | True to one decimal (L1) |
| App. D.3 LexSlot | "It stays below SBR-Recall on CIL-CORD on every seed, against both the gate and the clean SBR-Recall runs" | LexSlot gate 90.2/89.7/89.2 < clean 91.35/92.62/91.86 and < gate 92.69/93.37/92.44 | True. The cross-protocol side is named explicitly, so this is not a silent mix. |
| App. D.4 delta | "All rows of tab:delta are legacy gate runs ...; the clean rank-64 runs ... learn sessions 3 and 4 as well" | Clean range 85.6–94.5 vs the gate's 88.8–96.2 | Ambiguous (L2) |
| App. G | "the CIL-CORD cells of CoLaR-SBR and SBR-Recall are clean, and their fp32 gate runs appear only in the LexSlot comparison and the delta-capacity table" | The CoLaR-SBR gate run feeds `tab:slot` and `\vSbrGate`. The Recall gate run feeds `tab:delta` and the LexSlot sentence. `tab:conflict` "CoLaR-SBR" reads `C["sbr"]` = clean. The § footnote is scoped "On CIL-CORD", so the Mixed Recall gate cell (85.9†,b) is correctly not covered. | True |
| App. H wall-clock | "an SBR-Recall run without memory about 1.5 to 1.7 (clean runs)" | 1.45 / 1.53 / 1.67 h | True, but typed (L3) |

**Cross-protocol statement.** It appears once in the body (the 2×2 caption) and once in App. G, which
is where the protocols are stated. No remaining sentence says or implies that Recall and joint
share a protocol. "SBR-Recall 91.9±0.6 against 91.9±1.3" in the Main comparison relies on the table
marks, as every other cell does.

**The abstract's "reaches 91.9, on par with joint training"** (source unchanged, sha1 `163dd69a`
before and after; advice only).

The evidence:
- The means are 91.94 against 91.92.
- The seed-paired differences are −1.49 / +2.14 / −0.57: mean +0.03, s.d. 1.9, with both signs.
- The largest deficit (1.5, s42) is about one joint s.d. (1.26). The largest gap (2.1) favours Recall.
- The protocols differ only in precision (above).

My judgement is that **"on par" is defensible as rendered**: it claims neither "above" nor
"matches", and on every seed the two are within 2.1 points in a direction that changes sign. With
n = 3 it means "not distinguishable", not "equal". If the user wants wording that holds per seed
by construction and survives a reviewer reading rule-7 strictly, I would use "reaches $\nRecall$,
within $\nRecallJointGap$ points of joint training on every seed". Both macros exist, and the text
then tracks the clean joint cell when it lands. Either way, re-check this clause when clean joint
lands: if clean joint rises (fp32) by more than about 1.5 on s42, "on par" will need re-checking.

**R5 L1–L4.**

| Item | Status |
|---|---|
| L1, App. G stale | FIXED. The text quoted above; the 2×2 phrase is now "the blocked-readout † cell fp32"; § rows added. |
| L2, Fig. 4 protocol | FIXED. Panel titles give "(n=…, clean/†)", and the caption explains †. |
| L3, 6 GB sentence | FIXED. "The runs reported here ran on a single 6 GB GPU, except some first-grid rows (app:repro)". |
| L4, seed order | FIXED. 42/7/123 for both methods. |

### New findings (LOW only)

- **R6-L1. Upper bounds are rounded to nearest, so three "no more than / within" bounds are below the true maximum by 0.01 to 0.05.**
  - Where:
    - `vals.tex` `\nRecallMaxLoss` "0.5": true 0.549 (s123, session 4). Used in the Fig. 4 caption, the Retention ¶ and App. C.
    - `\nRecallMaxGain` "2.0": true 2.03 (s42, session 2). App. C reads "none rises more than 2.0".
    - `supp_vals.tex` `\recallDevMax` "1.3": true 1.31.
  - At the paper's one-decimal precision these are the right values, and no per-seed matrix is printed, so no reader can see a contradiction. Strictly, though, "falls no more than 0.5" is false for a 0.55 loss.
  - Fix, either:
    - (a) round bound macros up in `recall_macros()` / `write_supp_vals()` (`math.ceil(10*x)/10` gives 0.6 / 2.1 / 1.4); or
    - (b) word them "about 0.5", "about 2".
  - Optional in the same edit: give the body Retention ¶ the caption's "and the larger moves, up to \nRecallMaxGain, are gains", since "the curves are flat" omits a 2-point rise.
- **R6-L2. App. D.4, "the clean rank-64 runs ... learn sessions 3 and 4 as well" (supp.tex:298) can be read as "equally well".**
  - The clean diagonals of sessions 3 and 4 span 85.6 to 94.5. The s42 session-4 value is 85.6, below the gate range of 88.8 to 96.2 quoted two sentences earlier.
  - Fix: "also learn sessions 3 and 4 (\cref{tab:matrices})", or give the range from a generated macro.
- **R6-L3. App. H wall-clock "about 1.5 to 1.7 (clean runs)" is a typed run value (rule 6).**
  - It is true (1.45 / 1.53 / 1.67 h), and earlier rounds accepted the typed CoLaR-SBR "2 to 2.5".
  - Fix (optional): point to the h/run column of `tab:perseed` (1.6), which is generated.
- **R6-L4. Advisory: the 2×2 caption and App. G could say that the legacy joint cell differs from the clean protocol only in precision.**
  - Its config dumps have `val_frac 0.1`, 100 epochs and the clean recipe otherwise; `amp true` gives fp16 on all three seeds.
  - This is stronger and more useful than the generic "crosses protocols". App. G's "three ways" do not apply to joint.
  - Fix (optional): "(fp16; otherwise the clean recipe)".
- **R6-L5. Latent generator label (`write_agree`).**
  - The queued clean BROS runs have `run_alias colar_sbr_bros`. That maps to the key `sbr_bros`, which `ROW_LABEL` lacks, so the agreement table would print "sbr\_bros".
  - Fix: add `"sbr_bros": "CoLaR-SBR (BROS)"` before those runs land. No effect today.

### Hygiene

| Check | Result |
|---|---|
| Build | PASS. `latexmk -pdf main.tex` = 0, `supp.tex` = 0 (scratch rebuild of d559dd7). 0 undefined references or citations in `main.log` and `supp.log`. |
| Body ≤ 8 pages | PASS. `main.pdf` is now **11** pages (10 at d50d267). "References" starts on p. 8, right column, after the Conclusion; pp. 9–10 are references, and p. 11 holds only ref. [52]. No float or body text after p. 8. Note for the PM: Overleaf will now show 11 pp. The body itself did not cross p. 8. `supp.pdf` is 15 pp. |
| `---` | PASS. 0 in every tracked `paper_cvpr/*.tex` and `fig/*.tex`; 0 em-dashes in either PDF. |
| Anonymity | PASS. 0 hits for doccl / results/ / scripts/ / .py / clean_2026 / _seed / author or affiliation strings in either PDF. PDF Author fields are empty. |
| Abstract | Source byte-identical to d50d267. It renders "attain 88.4 average F1 against 91.9" and "reaches 91.9, on par with joint training". |
| Teaser / figure | The teaser bar values and labels come from `vals.tex` macros and match. Fig. 4 titles state n and protocol correctly. |

### OPEN (not counted)

- Clean joint and no-memory SBR on CIL-CORD are queued; when they land, re-check the abstract's "on par" (above), `\nRecallJointGap`, and the 2×2 caption sentence, which will then be stale.
- CR2: the remaining † headline cells, pending data.
